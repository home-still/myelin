# M41 — a derived cache you cannot rebuild is not a cache

## The incident

An arm of the dated-digest measurement died at row 424 of 500. The reader had
stopped answering:

```
store: qwen3.5-9b: request to http://127.0.0.1:5810/v1/chat/completions failed:
error sending request for url (http://127.0.0.1:5810/v1/chat/completions)
```

Both SSH tunnels to `big` had exited 255. The services were fine — checked
within a minute, reader, reranker and embedder all answering 200. Fifty
minutes of GPU time produced nothing usable because of a few hundred
milliseconds of network.

The re-run died too, at row 18, differently:

```
qdrant: Not found: Collection `myelin_longmemeval_s` doesn't exist!
```

The Qdrant log names the cause exactly:

```
11:44:18  Deleting collection myelin_lme_v2_pilot
11:44:23  Deleting collection myelin_lme_v2_small
11:44:28  Deleting collection myelin_locomo
11:44:32  Deleting collection myelin_longmemeval_s
11:44:37  gRPC /qdrant.Points/QueryBatch failed ... `myelin_longmemeval_s` doesn't exist!
11:44:41  Deleting collection myelin_lme_s_pref
```

Every `DELETE` carries `Referer: http://192.168.1.110:6333/dashboard` and
`User-Agent: qdrant-js/1.15.1`. All five `myelin_*` collections were removed by
hand through the Qdrant web dashboard, one at a time, over twenty-three
seconds — with the benchmark's failing query landing in the middle of the
sequence. Not a crash, not an eviction, not this code.

Two distinct defects, both ours, both exposed by one afternoon.

## Defect 1 — one dropped socket discards a whole run

Every outbound call in `myelin-core` goes to a model service, and none of them
retried. A transport failure was indistinguishable from a wrong answer: the
call returned `Err`, the caller aborted, the run died.

`crates/myelin-core/src/net.rs` adds `send_retrying`, used by all three
clients — `llm/openai.rs`, `embed/remote.rs`, `rerank/cross.rs`.

Retrying is only sound because these three calls are **pure**: an embedding, a
rerank and a chat completion are functions of their request body with no
server-side effect to duplicate. Nothing here writes. That is a property of
the call sites, not of HTTP, so the helper is `pub(crate)` and not a general
client.

What is retried, and what deliberately is not:

| outcome | action | why |
| --- | --- | --- |
| connect / timeout / request-send error | retry | the dropped-tunnel case |
| connection dies mid-body | retry | request was pure; re-issuing is sound |
| `429`, any `5xx` | retry | llama-swap answers `5xx` while a model loads |
| `400` and every other `4xx` | **fail at once** | `exceed_context_size_error` will be a 400 every time; retrying turns one wasted call into four |

Backoff is 0.5 s, 2 s, 8 s — four attempts, ≤ 10.5 s per call. Negligible
against a run of hundreds of model calls, long enough to outlast a supervised
tunnel restart. The error names the attempt count, so a log that ends here is
not mistaken for one unlucky call.

Five tests drive a raw `tokio` TCP listener that actually drops connections
rather than a mock that pretends to, since the behaviour under test is
transport-level and a mock framework abstracts exactly that away.

## Defect 2 — the index was unrebuildable

The ledger is the system of record; the vector index is a derived cache. That
was always the design. Nothing enforced it, and the difference only became
visible when the collections were gone.

The ledgers were untouched:

| ledger | live records | integrity |
| --- | --- | --- |
| `longmemeval_s` | **162,181** | ok |
| `lme_v2_small` | 85,979 | ok |
| `longmemeval_s_pref` | 13,536 | ok |
| `locomo` | 4,875 | ok |

Every vector was recomputable — and there was no way to recompute it.

`build` is not that way, and the reason is worth recording because it fails
*silently*. Its resume guard is `Ledger::unit_is_complete`, which reads the
`unit_complete` audit event — from the ledger. Run against a surviving ledger
it skips every unit, prints "already ingested" 500 times, exits 0, and leaves
an empty collection. Deleting the ledger to force a real rebuild is worse:
extraction is not deterministic, so it would mint *different* records and
silently break comparability with every number this project has published.

`myelin-eval reindex` is the missing path. Embed-only: no reader, no
extraction, ids preserved exactly, so runs measured before a rebuild stay
comparable after it.

```
myelin-eval reindex --corpus longmemeval-s
```

`Ledger::live_records_page` is the scan, keyset-paginated on `rowid` —
`LIMIT/OFFSET` re-walks every skipped row, which over 162k rows is quadratic.
Its predicate is `count_live`'s, so the number of records it yields is exactly
the number that count audits and a clean `reconcile` expects. A property test
asserts the two agree over a ledger holding live, quarantined and
not-yet-valid records, paged two at a time so the walk crosses a partial final
page.

`ReindexReport::is_complete` makes the silent-empty-rebuild failure loud: the
command exits non-zero unless every live record reached the index.

## Measured

| quantity | value |
| --- | --- |
| reindex throughput, CPU bge-m3 on `big` | **38 records/s** |
| LongMemEval_S rebuild | 162,181 records, ~71 min |
| retry ceiling per call | 10.5 s, 4 attempts |
| tests added | 9 (5 retry, 3 report, 1 scan property) |

## What this does not do

It does not measure `digest_dates`. That switch is built, defaulted off, and
pinned by tests — `the_dating_switch_off_reproduces_the_undated_digest`
guarantees the off path is byte-identical to M40's measured arm — but its arm
has not run, because the corpus it runs against was deleted twice mid-flight.
The pre-registration stands unchanged in `m41-dated-digest.md`; the arm is
M42's first job.

## Rule

A store whose contents cannot be reconstructed from the ledger is a store
whose contents can be lost. Any future index — a second collection, a
different embedding model, a graph projection — ships with its rebuild path or
it does not ship.
