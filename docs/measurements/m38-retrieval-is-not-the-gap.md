# M38 — on LongMemEval retrieval is solved, and a client bug was eating responses

Two premises refuted before any GPU time was spent, one transport bug found
and fixed, and the 18.8-point LongMemEval_S gap located precisely.

## The pre-registered plan died on inspection

PLAN §15 named two write-path changes for this milestone. Both are dead, and
neither needed a run to kill.

**1. Fixing entity extraction is not a retrieval lever.** M37 reported that all
37,731 LME-V2 page records carry exactly one entity, the literal token `page`.
True, and irrelevant: `entity_ids` is written to the Qdrant payload and indexed
as a keyword field, and **never read** — no filter, no query, no ranking stage
consults it. The only consumer of `record.entities` is
`pipeline::phrases::incidence_rows`, which feeds the graph/PPR channel that
`RetrieveConfig::graph` keeps off and M12 measured as a loss. Repairing the
extractor would have changed no retrieved record.

**2. Trajectory routing is worse than flat retrieval.** M37 proposed routing a
question to its trajectory via the 100 natural-language `goal:` records
(median rank 3, top-10 76.8%) and searching within it. Simulated offline
against the dense vectors already in Qdrant — so at no re-ingest cost — over
the 105 answerable web rows with a locatable gold record:

| | flat | routed@5 | routed@10 | routed@20 | routed@50 |
| --- | --- | --- | --- | --- | --- |
| dense recall@25 | **51.4%** | 44.8% | 45.7% | 49.5% | 54.3% |

Routing *loses* 5.7 points at top-10. The 23% of gold discarded by the filter
costs more than the tenfold density gain returns, and the mechanism only draws
level once it keeps half the corpus. Scope is immutable under I1, so shipping
this would have cost a full re-ingest to land a regression.

The check took 66 seconds.

## Where the LongMemEval_S gap actually is

Retrieval is not the problem. Using the benchmark's own `answer_session_ids`
— an exact annotation, not the string-matching proxy M37 had to use on LME-V2
— at the shipped operating point (`investigate`, `max_steps 2`, `select`,
k=25):

| measure | value |
| --- | --- |
| any gold session reached the reader | **93.8%** |
| **every** gold session reached the reader | **88.5%** |
| mean fraction of gold sessions covered | **91.9%** |
| mean evidence items emitted | 12.6 |

The pool before selection scores the same 93.8% at 24.0 items, so **M32's
sufficiency selector halves the evidence with zero recall loss** — the first
independent confirmation that it is doing what it claims.

Against judged accuracy of **62.00**, that leaves ~32 points sitting in
reading, not retrieval. The per-category split says the same thing and says
where:

| question_type | n | accuracy | wrong | gold sessions needed |
| --- | --- | --- | --- | --- |
| temporal-reasoning | 127 | **39.4%** | 77 | 2.20 |
| multi-session | 121 | **44.6%** | 67 | 2.59 |
| single-session-preference | 30 | 33.3% | 20 | 1.00 |
| knowledge-update | 72 | 77.8% | 16 | 2.00 |
| single-session-user | 64 | 90.6% | 6 | 1.00 |
| single-session-assistant | 56 | 96.4% | 2 | 1.00 |

`temporal-reasoning` and `multi-session` are **76% of all errors**. The
categories needing exactly one gold session score 90.6% and 96.4%; the
categories needing two or more score 39.4% and 44.6%.

The tempting reading — "we only retrieve one of the several sessions needed" —
is wrong, and the complete-coverage column above is what rules it out.
Retrieval delivers *every* gold session 88.5% of the time. The reader is
handed all the evidence and still cannot aggregate across it.

## The transport bug

While sweeping, `myelin-mcp` appeared to return truncated JSON:

```
myelin-mcp returned a body that is not JSON (Unterminated string starting at:
line 1 column 70 (char 69)): '{"jsonrpc":"2.0","id":278,"result":...
```

The server was innocent. Instrumenting the raw response:

```
raw bytes read: 89126      decoded chars: 12470
```

`adapters/myelin.py::_decode` framed SSE with `str.splitlines()`, which breaks
on VT, FF, FS, GS, RS, NEL, **U+2028** and U+2029 in addition to CR/LF. U+2028
LINE SEPARATOR appears in LongMemEval's ShareGPT-derived conversations. It
split the single `data:` line into fragments; only the first kept the prefix;
the rest were dropped by the `startswith("data:")` filter. An 88,984-character
payload decoded as 12,473 characters of unparseable JSON, and the adapter then
burned six retries with exponential backoff before failing the row.

Silent truncation of a *correct* server response, in the client. Fixed by
framing on `\r\n | \r | \n` only and stripping one space after `data:` rather
than `.strip()`, per the SSE spec.

Impact: any run over a corpus containing these characters lost rows and paid
minutes of retry backoff per affected row. Verified end-to-end — the row that
failed at index 277 now completes, 290/290 with zero failures.

### The fixture nearly failed to catch it

The first version of the regression test built its frame with
`json.dumps(...)`, whose default `ensure_ascii=True` escapes U+2028 to a
six-character `\u2028` that no line splitter can break on. That test **passed
against the unfixed decoder**. `serde_json` emits the raw character, so the
fixture has to as well (`ensure_ascii=False`); the test only became evidence
once it did. Confirmed by running both implementations side by side: the old
one raises `JSONDecodeError` on U+2028, U+2029 and U+0085, the new one round
trips all three.

VT and FF break `splitlines()` too but are **not** tested, because JSON
requires control characters to be escaped, so they cannot reach the decoder
raw and a subtest for them could not fail.

## Instrument

`adapters/recall_sweep.py` gains a `--score session` mode: did retrieval return
a record from a gold session? Exact where LongMemEval and LoCoMo annotate
evidence, unlike the verbatim-string proxy. The session a record came from is
recovered from `prov_source.doc`, so this needs no re-ingest.

It reports `any`, `complete` and mean `coverage` separately, because `any` is
what every LongMemEval baseline publishes and it is lenient by construction:
on a corpus where multi-session questions need 2.59 gold sessions, "any" calls
a question retrieved when the evidence cannot answer it.

Probe failures are now recorded and excluded from the denominator rather than
aborting the sweep — a 24-minute serial measurement should not be lost to one
bad row, and a failed row must not be scored as a miss.


## The mechanism, and its pre-registration

Written before the arm ran.

`SELECT_SYSTEM`'s first rule is:

```
- Return the indices of the FEWEST memories that TOGETHER answer the question.
```

`InvestigateConfig::select_coverage` swaps it for a prompt that asks for
**every** needed memory and says explicitly that answering often requires
combining memories from different days or conversations. Nothing else
changes — same schema, same `k`, same stable partition, same injection
defence (pinned by a test over both prompts).

**Why the clause is suspect and not merely unlucky.** `select_pool`
stable-partitions and drops no candidate; `compose`'s token budget does the
cutting. So "fewest" cannot reduce what the reader is shown — it can only
decide which records lose the race to the budget. It is an instruction with
no upside on this path, and a measured downside that scales with the number
of memories a question needs and is exactly −0.0 where one memory suffices.

**Instrument.** `recall_sweep.py --score session --mode investigate --select`,
comparing `--select-coverage` against the shipped default. Complete
gold-session coverage is the metric: exact, no reader or judge needed, and it
is the quantity the mechanism targets.

**Population.** All 500 LongMemEval_S rows, reported per `question_type`.

**Decision rule.**

- Ship `select_coverage` on if **complete coverage over the three multi-gold
  categories improves by ≥ +3.0 points**, and no single-gold category
  regresses by more than 1.0.
- A judged arm is only earned if this passes. Coverage is necessary, not
  sufficient: M38's whole point is that the reader already fails with the
  evidence in hand, so more coverage may not convert.

That last clause matters. Complete coverage on the multi-gold categories has
a ceiling of 91.4% — the pool's own figure — so the most this mechanism can
recover is the ~4.7 points the selector currently gives away. Worth having,
and nowhere near the 32-point reading gap.

## Results

**Null, and it refutes this milestone's own explanation. Default stays off.**

| arm | any | complete | coverage | mean items |
| --- | --- | --- | --- | --- |
| shipped (`FEWEST`) | 93.8% | 88.6% | 91.9% | 12.6 |
| `select_coverage` (`EVERY`) | 93.8% | 88.6% | 91.9% | 12.6 |

**500 of 500 rows byte-identical** on `gold_hit`, `complete` and `n_items`.
Not "within noise" — the same selection on every single question. The
pre-registered bar was +3.0 on the multi-gold categories; the result is
+0.000.

### The switch was live, and that is the finding

An identical result is the signature of an inert switch, so it was verified
at the wire rather than inferred, by pointing `MYELIN_LLM__URL` at a
recording proxy and reading back the system prompt the selector actually
sent:

```
--- call 0 (default) ---
- Return the indices of the FEWEST memories that TOGETHER answer the question.
--- call 1 (--select-coverage) ---
- Return the indices of EVERY memory needed to answer the question.
```

The prompt changed. The selection did not. **The 9B selector ignores the
parsimony instruction completely**, which means the −5.3-point coverage loss
measured above is *not* an instruction-following effect and the diagnosis
earlier in this document had the cause wrong. The loss is real and
reproducible; attributing it to the word "FEWEST" was an inference, and this
arm is what turns it into a refuted one.

That distinction is the whole reason the wire check exists. Without it this
would have been filed as "the prompt rewrite didn't help", leaving the false
causal story in place.

### What it implies for the next attempt

Prompt-level surgery on this selector is a dead end: the model does not read
the constraint clause. Anything that recovers the 4.7 points the selector
gives away on multi-gold questions has to be **structural** — the selection
is a judgement about which records rank highest, and on multi-gold questions
that judgement is simply worse. Candidates, in cheap-first order:

1. **Budget-aware promotion.** `compose` cuts at 10,000 tokens and the
   selector cannot see that ceiling, so it cannot know that ranking a record
   twelfth is equivalent to discarding it.
2. **Select per gold unit rather than globally.** The selector is asked one
   question over 25 candidates spanning several sessions; the categories it
   fails are exactly those where the answer is distributed across them.

Neither is measured, and neither is claimed. M38's contribution is the
localisation and the transport fix.