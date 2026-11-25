//! Named stores and the agent surface, over the MCP wire.
//!
//! Every test serves a [`MyelinServer`] in-process and speaks MCP to it as a
//! client does, so what is asserted is what a caller sees: the tool list, the
//! arguments a surface accepts, and the `store` label on each hit.
//!
//! The embedder, the reranker and the model are stand-ins. What is under test
//! is which scopes a read covers and how their hits are merged and labelled;
//! a real cross-encoder would make these tests measure the cross-encoder.
//!
//! ```text
//! cargo nextest run -p myelin-mcp --features integration -E 'binary(stores)'
//! ```

#![cfg(feature = "integration")]

#[path = "../../myelin-core/tests/common/mod.rs"]
mod common;

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use common::{commit, episode, scratch_store, semantic, HashEmbedder, ScratchGuard};
use myelin_core::error::{MyelinError, Result};
use myelin_core::llm::{Completion, CompletionRequest, Llm};
use myelin_core::model::record::Scope;
use myelin_core::model::store::StoreDef;
use myelin_core::pipeline::retrieve::RetrieveConfig;
use myelin_core::rerank::Reranker;
use myelin_core::store::ledger::Ledger;
use myelin_mcp::server::{Backend, MyelinServer, Surface, AGENT_REMOVED_TOOLS};
use rmcp::model::CallToolRequestParams;
use rmcp::service::{RunningService, ServiceError};
use rmcp::{RoleClient, ServiceExt};
use serde_json::{json, Value};

/// The default store: read when a read names nothing.
const PAPERS: &str = "papers";
const PAPERS_TENANT: &str = "t/papers";
/// Not a default store: read only when named.
const NOTES: &str = "notes";
const NOTES_TENANT: &str = "t/notes";

/// The word the stand-in reranker scores highest. Only `notes` holds it.
const BEST_MARKER: &str = "QUOKKA";
const BEST_SCORE: f32 = 0.9;
const OTHER_SCORE: f32 = 0.1;

/// A word every `notes` record carries and no `papers` record does: what a
/// test reads a hit's true store from, independently of its `store` label.
const NOTES_WORD: &str = "owner";

const QUERY: &str = "what is known about attention";

/// Live records the fixture writes to each store's tenant.
const PAPERS_RECORDS: i64 = 3;
const NOTES_RECORDS: i64 = 2;

/// Buffer of the in-process pipe between the MCP client and the server.
const WIRE_BUFFER_BYTES: usize = 1 << 16;

/// Scores a document by one word, so a test states which record is the
/// better match instead of inferring it from a model.
struct MarkerReranker;

#[async_trait]
impl Reranker for MarkerReranker {
    fn id(&self) -> &str {
        "test-marker-reranker"
    }

    async fn rerank(&self, _query: &str, documents: &[String]) -> Result<Vec<f32>> {
        Ok(documents
            .iter()
            .map(|d| {
                if d.contains(BEST_MARKER) {
                    BEST_SCORE
                } else {
                    OTHER_SCORE
                }
            })
            .collect())
    }
}

/// `recall` has no model in the loop, and every refusal asserted here is made
/// before one is reached. A call is a test failure, reported as the error.
struct UnreachedLlm;

#[async_trait]
impl Llm for UnreachedLlm {
    fn id(&self) -> &str {
        "test-unreached-llm"
    }

    async fn raw_complete(&self, _req: &CompletionRequest) -> Result<Completion> {
        Err(MyelinError::Config(
            "a test in stores.rs reached the model".into(),
        ))
    }
}

type Client = RunningService<RoleClient, ()>;

/// A backend over its own scratch collection and ledger, holding the two
/// stores and their records.
async fn backend(
    label: &str,
    reranker: Option<Box<dyn Reranker>>,
) -> (Arc<Backend>, ScratchGuard) {
    let (store, guard) = scratch_store(label).await;
    let ledger = Ledger::open_memory().await.expect("ledger");

    let papers = Scope::new(PAPERS_TENANT, "myelin", "default");
    let notes = Scope::new(NOTES_TENANT, "myelin", "default");
    let papers_ep = episode(&papers, "p-ep", "A paper on attention in transformers.");
    let notes_ep = episode(&notes, "n-ep", "A note the owner wrote about attention.");
    // Episodes first: a semantic record is refused until its ancestor exists.
    let records = vec![
        papers_ep.clone(),
        notes_ep.clone(),
        semantic(
            &papers,
            "p-1",
            "Attention weighs every token against every other token.",
            vec![papers_ep.id],
        ),
        semantic(
            &papers,
            "p-2",
            "Attention layers replaced recurrence in sequence models.",
            vec![papers_ep.id],
        ),
        semantic(
            &notes,
            "n-1",
            &format!("{BEST_MARKER}: what is known about attention, as the owner noted it."),
            vec![notes_ep.id],
        ),
    ];
    commit(&ledger, &store, &HashEmbedder, &records).await;

    for (name, tenant, default_read) in [(PAPERS, PAPERS_TENANT, true), (NOTES, NOTES_TENANT, false)]
    {
        ledger
            .define_store(&StoreDef {
                name: name.into(),
                tenant: tenant.into(),
                description: format!("the {name} fixture"),
                default_read,
                agent_writes: false,
                dated: false,
            })
            .await
            .expect("define store");
    }

    let backend = Backend {
        store,
        ledger,
        embedder: Box::new(HashEmbedder),
        llm: Box::new(UnreachedLlm),
        reranker,
        config: RetrieveConfig::default(),
    };
    (Arc::new(backend), guard)
}

/// Serve `backend` on `surface` and connect an MCP client to it.
async fn connect(backend: Arc<Backend>, surface: Surface) -> Client {
    let (client_io, server_io) = tokio::io::duplex(WIRE_BUFFER_BYTES);
    let server = MyelinServer::new(backend, surface);
    tokio::spawn(async move {
        let running = server.serve(server_io).await.expect("server handshake");
        let _ = running.waiting().await;
    });
    ().serve(client_io).await.expect("client handshake")
}

async fn call(
    client: &Client,
    tool: &'static str,
    arguments: Value,
) -> std::result::Result<Value, ServiceError> {
    let Value::Object(arguments) = arguments else {
        panic!("tool arguments are a JSON object, got {arguments}");
    };
    let result = client
        .call_tool(CallToolRequestParams::new(tool).with_arguments(arguments))
        .await?;
    Ok(result
        .structured_content
        .unwrap_or_else(|| panic!("{tool} returned no structured content")))
}

/// The message of a call the server refused.
async fn refusal(client: &Client, tool: &'static str, arguments: Value) -> String {
    match call(client, tool, arguments).await {
        Err(ServiceError::McpError(e)) => e.message.to_string(),
        Err(other) => panic!("{tool} failed without a refusal: {other}"),
        Ok(value) => panic!("{tool} was not refused: {value}"),
    }
}

/// The `store` of every hit, in rank order.
fn hit_stores(result: &Value) -> Vec<String> {
    result["hits"]
        .as_array()
        .expect("hits")
        .iter()
        .map(|h| h["store"].as_str().expect("a hit names its store").to_string())
        .collect()
}

fn item_values(result: &Value) -> Vec<String> {
    result["items"]
        .as_array()
        .expect("items")
        .iter()
        .map(|i| i["value"].as_str().expect("value").to_string())
        .collect()
}

fn trace_stores(result: &Value) -> Vec<Value> {
    result["traces"]
        .as_array()
        .expect("traces")
        .iter()
        .map(|t| t["store"].clone())
        .collect()
}

async fn tool_names(client: &Client) -> BTreeSet<String> {
    client
        .list_all_tools()
        .await
        .expect("tools/list")
        .into_iter()
        .map(|t| t.name.to_string())
        .collect()
}

#[tokio::test]
async fn a_default_read_leaves_out_a_store_that_holds_the_better_match() {
    let (backend, _guard) = backend("stores_default", Some(Box::new(MarkerReranker))).await;
    let client = connect(backend, Surface::Full).await;

    let result = call(&client, "recall", json!({ "query": QUERY }))
        .await
        .expect("recall");

    let stores = hit_stores(&result);
    assert!(!stores.is_empty(), "the default store answered nothing: {result}");
    assert!(
        stores.iter().all(|s| s == PAPERS),
        "a default read returned a hit from outside the default store: {stores:?}"
    );
    // `notes` holds the one record the reranker scores highest. Had the read
    // reached it, ranking alone would have put it first.
    assert!(
        item_values(&result).iter().all(|v| !v.contains(BEST_MARKER)),
        "a default read returned the non-default store's record: {result}"
    );
    assert_eq!(trace_stores(&result), vec![json!(PAPERS)]);
}

#[tokio::test]
async fn a_read_of_two_stores_labels_each_hit_with_its_store() {
    let (backend, _guard) = backend("stores_two", Some(Box::new(MarkerReranker))).await;
    let client = connect(backend, Surface::Full).await;

    let result = call(
        &client,
        "recall",
        json!({ "query": QUERY, "stores": [PAPERS, NOTES] }),
    )
    .await
    .expect("recall");

    let stores = hit_stores(&result);
    let values = item_values(&result);
    assert_eq!(stores.len(), values.len(), "one hit per item: {result}");
    assert_eq!(
        result["record_ids"].as_array().map(Vec::len),
        Some(values.len()),
        "one record id per item: {result}"
    );
    // The merge is one ranking of both stores' hits: the best match leads
    // although its store was named second.
    assert_eq!(stores.first().map(String::as_str), Some(NOTES), "{result}");
    assert!(values[0].contains(BEST_MARKER), "{result}");
    // Every hit carries the store that holds its record, not the first one.
    for (store, value) in stores.iter().zip(&values) {
        let expected = if value.contains(NOTES_WORD) { NOTES } else { PAPERS };
        assert_eq!(store, expected, "hit {value:?} is labelled {store:?}");
    }
    assert!(stores.iter().any(|s| s == PAPERS), "no hit from {PAPERS}: {result}");
    assert_eq!(trace_stores(&result), vec![json!(PAPERS), json!(NOTES)]);
}

#[tokio::test]
async fn a_read_of_two_stores_without_a_reranker_is_refused() {
    let (backend, _guard) = backend("stores_norerank", None).await;
    let client = connect(backend, Surface::Full).await;

    let message = refusal(
        &client,
        "recall",
        json!({ "query": QUERY, "stores": [PAPERS, NOTES] }),
    )
    .await;
    assert_eq!(message, "reading more than one store needs the reranker");

    // The refusal is about the merge, not the stores: one of them reads.
    let result = call(&client, "recall", json!({ "query": QUERY, "stores": [NOTES] }))
        .await
        .expect("recall of one store");
    let stores = hit_stores(&result);
    assert!(!stores.is_empty() && stores.iter().all(|s| s == NOTES), "{result}");
}

#[tokio::test]
async fn stores_reports_each_store_and_a_rebinding_is_refused() {
    let (backend, _guard) = backend("stores_list", Some(Box::new(MarkerReranker))).await;
    let client = connect(backend, Surface::Full).await;

    let result = call(&client, "stores", json!({})).await.expect("stores");
    let listed = result["stores"].as_array().expect("stores");
    let row = |name: &str| {
        listed
            .iter()
            .find(|s| s["name"] == name)
            .unwrap_or_else(|| panic!("{name} is not listed: {result}"))
    };
    assert_eq!(listed.len(), 2, "{result}");
    assert_eq!(row(PAPERS)["default"], json!(true));
    assert_eq!(row(PAPERS)["records"], json!(PAPERS_RECORDS));
    assert_eq!(row(PAPERS)["namespaces"], json!(1));
    assert_eq!(row(NOTES)["default"], json!(false));
    assert_eq!(row(NOTES)["records"], json!(NOTES_RECORDS));
    assert_eq!(row(NOTES)["writable"], json!(false));
    // The tenant behind a store is not part of what a caller is shown.
    assert!(row(PAPERS).get("tenant").is_none(), "{result}");

    let message = refusal(
        &client,
        "define_store",
        json!({
            "name": PAPERS,
            "tenant": "t/elsewhere",
            "description": "the same name over another tenant",
            "default_read": true,
            "agent_writes": false,
            "dated": false,
        }),
    )
    .await;
    assert!(
        message.contains(PAPERS) && message.contains(PAPERS_TENANT),
        "the refusal does not name the binding it hit: {message}"
    );
}

#[tokio::test]
async fn the_agent_surface_offers_no_operator_tool_and_refuses_tenants_and_unsafe_writes() {
    let (backend, _guard) = backend("stores_agent", Some(Box::new(MarkerReranker))).await;
    let operator = connect(backend.clone(), Surface::Full).await;
    let agent = connect(backend, Surface::Agent).await;

    let operator_tools = tool_names(&operator).await;
    let agent_tools = tool_names(&agent).await;
    for removed in AGENT_REMOVED_TOOLS {
        // Present on the full surface, so its absence here is the surface's
        // doing and not a tool that was never registered.
        assert!(operator_tools.contains(removed), "{removed} is not a tool");
        assert!(
            !agent_tools.contains(removed),
            "the agent surface offers {removed}"
        );
    }
    let expected: BTreeSet<String> = operator_tools
        .iter()
        .filter(|t| !AGENT_REMOVED_TOOLS.contains(&t.as_str()))
        .cloned()
        .collect();
    assert_eq!(agent_tools, expected);

    // A store, by name, reads.
    let result = call(&agent, "recall", json!({ "query": QUERY, "stores": [NOTES] }))
        .await
        .expect("agent recall of a named store");
    let record_id = result["record_ids"][0]
        .as_str()
        .expect("a record id")
        .to_string();
    assert!(hit_stores(&result).iter().all(|s| s == NOTES), "{result}");

    // A tenant does not, on any tool that takes one.
    for (tool, arguments) in [
        ("recall", json!({ "query": QUERY, "tenant": NOTES_TENANT })),
        ("investigate", json!({ "question": QUERY, "tenant": NOTES_TENANT })),
        ("search", json!({ "tenant": NOTES_TENANT })),
        ("namespaces", json!({ "tenant": NOTES_TENANT })),
        ("remember", json!({ "text": "a fact", "tenant": NOTES_TENANT })),
    ] {
        assert_eq!(
            refusal(&agent, tool, arguments).await,
            "this surface addresses stores by name; call stores",
            "{tool} accepted a tenant on the agent surface"
        );
    }

    // Neither store accepts agent writes.
    let message = refusal(&agent, "remember", json!({ "text": "a fact", "store": NOTES })).await;
    assert!(message.contains("does not accept writes"), "{message}");

    // Erasure is not the agent's, and neither is a record of a store it may
    // not write to.
    let message = refusal(
        &agent,
        "forget",
        json!({ "record_id": record_id, "mode": "hard", "confirm": true }),
    )
    .await;
    assert!(message.contains("mode must be \"soft\""), "{message}");
    let message = refusal(
        &agent,
        "forget",
        json!({ "record_id": record_id, "mode": "soft" }),
    )
    .await;
    assert!(
        message.contains("not in a store that accepts writes"),
        "{message}"
    );

    // Both were refused before they touched the record: it still reads.
    let after = call(&agent, "recall", json!({ "query": QUERY, "stores": [NOTES] }))
        .await
        .expect("recall after the refusals");
    assert_eq!(after["record_ids"], result["record_ids"]);
}
