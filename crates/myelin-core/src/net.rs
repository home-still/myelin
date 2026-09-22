//! Bounded retry for the three model services.
//!
//! Every outbound call in this crate goes to a model server — the reader, the
//! reranker, the embedder — reached over a link that is not always a loopback
//! socket. In the measurement rig it is an SSH tunnel to another host, and a
//! tunnel drops.
//!
//! A dropped connection is not a wrong answer, but without a retry it is
//! indistinguishable from one: the call returns `Err`, the caller aborts, and
//! a benchmark that has spent fifty minutes producing 424 of 500 rows exits
//! having produced nothing usable. That happened twice. The services were
//! healthy both times — checked within seconds, all three answering 200 — so
//! what was lost was not a model failure but a few hundred milliseconds of
//! network, amplified by a missing five lines of code into two hours.
//!
//! # What may be retried
//!
//! Retrying is only safe because these three calls are **pure**: an embedding,
//! a rerank and a chat completion are functions of their request body with no
//! server-side effect to duplicate. Nothing here writes. That is a property of
//! the current call sites, not of HTTP, so this module is deliberately
//! `pub(crate)` and not a general-purpose client.
//!
//! # What may not
//!
//! A deterministic rejection must fail on the first attempt. A 400 for
//! `exceed_context_size_error` will be a 400 every time, and retrying it turns
//! one wasted call into four. The classifier therefore retries only:
//!
//! - transport failures — connect, timeout, and request-send errors, which is
//!   the dropped-tunnel case;
//! - `429`, and `5xx`, which on llama-swap is also how a model reports that it
//!   is still loading.
//!
//! Every other status, including all remaining `4xx`, is returned to the
//! caller as-is on the first attempt.

use std::time::Duration;

use crate::error::{MyelinError, Result};

/// Delay before each retry. The length of this slice is the number of
/// *retries*, so attempts are `RETRY_DELAYS.len() + 1`.
///
/// Sized for the failure it exists to survive: a supervised tunnel restarting.
/// The escalation spends at most ~10s before giving up, which is noise against
/// a run of hundreds of model calls but long enough to outlast a reconnect.
const RETRY_DELAYS: [Duration; 3] = [
    Duration::from_millis(500),
    Duration::from_secs(2),
    Duration::from_secs(8),
];

/// Why an attempt is worth repeating, or isn't.
enum Verdict {
    /// Usable — hand the status and body back to the caller.
    Done(reqwest::StatusCode, String),
    /// Transient. Carries the message so the last one can be reported.
    Retry(String),
    /// Deterministic; repeating it would only waste the same call again.
    Fatal(MyelinError),
}

/// Send `build()`, reading the body, retrying transient failures.
///
/// `build` is a factory rather than a single `RequestBuilder` because a
/// builder is consumed by `send`; each attempt needs its own.
///
/// `label` names the model in errors, matching what the call sites already
/// emit.
pub(crate) async fn send_retrying(
    build: impl Fn() -> reqwest::RequestBuilder,
    label: &str,
    what: &str,
) -> Result<(reqwest::StatusCode, String)> {
    send_retrying_with(build, label, what, &RETRY_DELAYS).await
}

/// `send_retrying`, with the backoff schedule injected so tests need not sleep.
async fn send_retrying_with(
    build: impl Fn() -> reqwest::RequestBuilder,
    label: &str,
    what: &str,
    delays: &[Duration],
) -> Result<(reqwest::StatusCode, String)> {
    let mut last = String::new();

    for attempt in 0..=delays.len() {
        match attempt_once(&build, label, what).await {
            Verdict::Done(status, body) => return Ok((status, body)),
            Verdict::Fatal(e) => return Err(e),
            Verdict::Retry(message) => {
                last = message;
                if let Some(delay) = delays.get(attempt) {
                    tracing::warn!(
                        model = label,
                        attempt = attempt + 1,
                        of = delays.len() + 1,
                        error = %last,
                        "transient failure talking to model service; retrying"
                    );
                    tokio::time::sleep(*delay).await;
                }
            }
        }
    }

    // Report the attempt count, so a log that ends here is not mistaken for a
    // single unlucky call.
    Err(MyelinError::Store(format!(
        "{label}: {what} failed after {} attempts: {last}",
        delays.len() + 1
    )))
}

async fn attempt_once(
    build: &impl Fn() -> reqwest::RequestBuilder,
    label: &str,
    what: &str,
) -> Verdict {
    let response = match build().send().await {
        Ok(response) => response,
        Err(e) => {
            // `is_request` covers a connection closed while the request was in
            // flight, which is precisely the dropped-tunnel signature.
            return if e.is_connect() || e.is_timeout() || e.is_request() {
                Verdict::Retry(format!("{what}: {e}"))
            } else {
                Verdict::Fatal(MyelinError::Store(format!("{label}: {what}: {e}")))
            };
        }
    };

    let status = response.status();
    let body = match response.text().await {
        Ok(body) => body,
        // The connection died mid-body. The request was pure, so re-issuing it
        // is sound.
        Err(e) => return Verdict::Retry(format!("{what}: reading body: {e}")),
    };

    if status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
        return Verdict::Retry(format!(
            "{what}: HTTP {status}: {}",
            body.chars().take(200).collect::<String>()
        ));
    }

    Verdict::Done(status, body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A server that replies with `replies[n]` to the n-th request, and closes
    /// the connection without replying where the entry is `None`.
    ///
    /// Raw HTTP/1.1 rather than a mock crate: the behaviour under test is
    /// transport-level, and a dropped connection is exactly what a mock
    /// framework abstracts away.
    async fn server(replies: Vec<Option<(u16, &'static str)>>) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let seen = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&seen);

        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let n = count.fetch_add(1, Ordering::SeqCst);

                let mut buf = [0u8; 4096];
                let _ = socket.read(&mut buf).await;

                match replies.get(n).copied().flatten() {
                    Some((status, body)) => {
                        let response = format!(
                            "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        );
                        let _ = socket.write_all(response.as_bytes()).await;
                        let _ = socket.flush().await;
                    }
                    // Hang up with no response: a dropped tunnel.
                    None => drop(socket),
                }
            }
        });

        (format!("http://{addr}/"), seen)
    }

    fn no_delay(n: usize) -> Vec<Duration> {
        vec![Duration::ZERO; n]
    }

    /// The case that cost two benchmark runs: the connection dies, and the
    /// next attempt succeeds.
    #[tokio::test]
    async fn a_dropped_connection_is_retried_and_the_run_survives() {
        let (url, seen) = server(vec![None, Some((200, "{\"ok\":true}"))]).await;
        let client = reqwest::Client::new();

        let (status, body) = send_retrying_with(
            || client.post(&url).json(&serde_json::json!({})),
            "model",
            "rerank request",
            &no_delay(3),
        )
        .await
        .expect("the second attempt succeeds");

        assert_eq!(status, 200);
        assert_eq!(body, "{\"ok\":true}");
        assert_eq!(seen.load(Ordering::SeqCst), 2, "exactly one retry was needed");
    }

    /// A deterministic rejection must cost one call, not four. `llama.cpp`
    /// returns 400 `exceed_context_size_error` for a prompt that is too long,
    /// and no amount of repetition shortens the prompt.
    #[tokio::test]
    async fn a_client_error_is_returned_immediately_without_retrying() {
        let (url, seen) = server(vec![
            Some((400, "{\"error\":\"exceed_context_size_error\"}")),
            Some((200, "unreachable")),
        ])
        .await;
        let client = reqwest::Client::new();

        let (status, body) = send_retrying_with(
            || client.post(&url).json(&serde_json::json!({})),
            "model",
            "request",
            &no_delay(3),
        )
        .await
        .expect("a 400 is a result, not a transport failure");

        assert_eq!(status, 400, "the caller still sees the status it must report");
        assert!(body.contains("exceed_context_size"));
        assert_eq!(seen.load(Ordering::SeqCst), 1, "a 400 must not be repeated");
    }

    /// llama-swap answers 5xx while it is loading a model, so this one is
    /// worth waiting out.
    #[tokio::test]
    async fn a_server_error_is_retried() {
        let (url, seen) = server(vec![Some((503, "loading")), Some((200, "ready"))]).await;
        let client = reqwest::Client::new();

        let (status, _) = send_retrying_with(
            || client.post(&url).json(&serde_json::json!({})),
            "model",
            "request",
            &no_delay(3),
        )
        .await
        .expect("the model finished loading");

        assert_eq!(status, 200);
        assert_eq!(seen.load(Ordering::SeqCst), 2);
    }

    /// Retrying is bounded; a service that is simply down still fails, and the
    /// error says how hard we tried.
    #[tokio::test]
    async fn retries_are_bounded_and_the_error_reports_the_attempts() {
        let (url, seen) = server(vec![None, None, None, None, None]).await;
        let client = reqwest::Client::new();

        let error = send_retrying_with(
            || client.post(&url).json(&serde_json::json!({})),
            "qwen3.5-9b",
            "rerank request",
            &no_delay(3),
        )
        .await
        .expect_err("a dead service is still an error");

        assert_eq!(seen.load(Ordering::SeqCst), 4, "one attempt plus three retries");
        let message = error.to_string();
        assert!(message.contains("qwen3.5-9b"), "{message}");
        assert!(
            message.contains("after 4 attempts"),
            "the operator needs to know this was not one unlucky call: {message}"
        );
    }

    /// The schedule that actually ships: three retries, escalating, bounded by
    /// roughly ten seconds so a stalled service cannot stretch a run.
    #[test]
    fn the_shipped_backoff_is_bounded() {
        assert_eq!(RETRY_DELAYS.len(), 3);
        let total: Duration = RETRY_DELAYS.iter().sum();
        assert!(
            total <= Duration::from_secs(11),
            "a per-call ceiling this small stays negligible against a run, but \
             still outlasts a supervised tunnel restart"
        );
        assert!(
            RETRY_DELAYS.windows(2).all(|w| w[0] < w[1]),
            "backoff must escalate, so a service that needs a moment gets one"
        );
    }
}
