# M20 — the preference/persona profile layer

M19 closed by naming `single-session-preference` as the largest *relative* deficit left on either
benchmark: 26.67 judged against MemPro-15's 80.00 at the same answer-model class, 3.4 points of the
remaining 24.40-point `longmemeval_s.judge_score.n500` gap, and the one large stratum no milestone had
worked. This milestone builds the shape PERMA evaluates and MemMachine/TiMem implement — a typed profile
record written at consolidation time and composed by scope rather than by relevance — and measures it.

## Verdict

**Both switches ship off as measured nulls. `RecordKind::Profile`, the write pass, the `[profile]` block
and the MCP surface all ship; neither default flips.** No arm clears the rule fixed in §1.

| arm | mechanism | cat-3 judged (n=30) | Δ vs base | 95% CI | p | decision |
|---|---|---|---|---|---|---|
| base | *(none)* | 30.00 | — | — | — | — |
| A | `ComposeConfig::profile` — an always-composed `[profile]` block of the tenant's dispositions | 30.00 | **+0.0** | [−16.7, +16.7] | 1.0000 | **fails the rule; ships off** |
| B | `READER_PREFERENCE_CLAUSE` — tell the reader to answer from stated preferences | 33.33 | +3.3 | [−10.0, +16.7] | 0.8453 | **fails the rule; ships off** |
| A+B | both | 36.67 | +6.7 | [−10.0, +23.3] | 0.5382 | **fails the rule** |

Marginals, which decide whether both arms are needed:

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| A+B over A — the clause on top of the block | +6.7 | [−10.0, +23.3] | 0.5359 |
| A+B over B — the block on top of the clause | +3.3 | [+0.0, +10.0] | 0.7200 |

The second interval's lower bound is exactly `+0.0`, which does not *exclude* zero; at p = 0.72 it is a
null, not a marginal win. Every interval above is quoted verbatim from
`crates/myelin-eval/adapters/paired_ci.py`.

**Arm A's null has a measured cause, and it is not the record type.** The store holds a median of **124
live dispositions per tenant** (min 93, max 163, 3,750 total over 30 tenants). `PROFILE_MAX_RECORDS = 8`
selects by `t_ingested DESC`, so the block shows **6.5% of what the tenant knows**, chosen by recency.
Measured gold-content recall of the emitted block is **0.042 mean, with 9 of 30 questions at exactly
zero**. On the milestone's own headline example — *"Can you suggest some accessories that would
complement my current photography setup?"* — the block that reached the reader was about **baby
products**, because those dispositions were ingested last. The mechanism is wired correctly and reaches
the reader (§5); what failed is the selection rule, and §7 says what that implies.

**Arm B did what it was designed to do and it was not enough.** Declines on the 30 questions fall
**11 → 7**, and the four extra answers convert to +1 judged question. At n = 30 that is 3.3 points
against a ~13-point bar.

## 1. The rule, fixed before any arm

Written into this file before the category-3 store finished building and before any arm ran.

> `ComposeConfig::profile` and `READER_PREFERENCE_CLAUSE` each default **on** only if, on LongMemEval_S
> `single-session-preference` (n = 30, judged), the paired mean difference against the same-store
> baseline is positive with a 95% CI excluding zero, **and** the same arm over the full 500 loses no
> more than 1.0 point overall with a CI excluding zero. Otherwise the switch ships off as a measured
> null.

At n = 30, judged 0/1, a paired bootstrap CI excludes zero at roughly **+13 points — four questions.**
An arm that is positive but whose interval spans zero is reported as a null and its default stays off.
The population is **not** widened and the two arms are **not** pooled to manufacture significance; M12
and M13 both shipped nulls and that is the precedent this follows.

Every interval comes from the CLI, never recomputed by hand: `paired_ci.py` seeds `random.Random(seed)`
per call but draws indices in the order of the id list, and only the CLI's `sorted(set(sa) & set(sb))`
ordering reproduces the published bounds.

Token F1 is not usable on this stratum. M9 already flagged it "not scorable this way", and the M19 full
run measures it at **0.0456** against a judged **0.2667** on these same 30 rows. The judged column is
the reported one; token F1 is carried beside it and never alone. For the record, the arms' token F1 runs
base 0.0504 / A 0.0457 / B 0.0597 / A+B 0.0604 — the same ordering as the judged column and equally
uninformative about whether an answer was right.

## 2. The diagnosis, per question

Measured from `runs/rescored/m19_lme_s_full_judge/per_question.jsonl` (500 rows; category 3 is n = 30).

| quantity | value |
|---|---|
| judged score | **26.67** (8 of 30 correct) |
| token F1 | 0.0456 |
| failures | 22 |
| — declined outright (`I don't know`) | **11** |
| — answered generically | **11** |

This is not purely a retrieval miss. The clearest case is `06878be2`, *"Can you suggest some accessories
that would complement my current photography setup?"*: evidence slot [5] held `As a Sony camera user,
I've been thinking about upgrading my camera bag` and slot [4] held `compatible flash for your Sony A7R
IV`, and the reader answered `Godox V1 flash, protective cases, camera bags` against a gold of *"The
user would prefer suggestions of Sony-compatible accessories…"*. Gold-content coverage in the composed
evidence for the eleven decliners runs 0.38–0.89.

Two causes, so two arms, exactly as M19 ran mechanism-plus-reader-clause:

- **Arm A (store/compose).** Nothing in the store represents a disposition. The Sony preference is
  scattered across turns and competes for the six evidence slots on lexical merit against the question
  "suggest some accessories", which it does not match.
- **Arm B (reader).** `READER_SYSTEM` says *"Answer in as few words as possible"* and *"If the memories
  do not contain the answer, reply exactly: I don't know."* A preference question has no literal answer
  in the memories; it must be stated from the user's dispositions. The prompt instructs the exact
  failure observed.

M19's precedent for keeping the arms separate is quantitative: on LoCoMo's temporal stratum the
compose-side annotation was +37.6 and the reader-side clause +14.3, with both marginals significant
(+5.2 and +28.4). Neither arm may be assumed to subsume the other — and here, neither wins.

## 3. The cost constraint, and what the pass actually cost

`build.rs` sets `write.extract_facts = false` for LongMemEval_S because 61.2M haystack tokens through
fact extraction extrapolates to ~250 GPU-hours. A profile pass must not reintroduce that. Measured from
`data/longmemeval_s.json`:

| quantity | value |
|---|---|
| items (tenants) | 500 |
| sessions | 25,112 (≈50.2/item) |
| user turns | 122,506 — **30,751,260 chars ≈ 7.69M tokens** |
| assistant turns | 124,424 — 214,111,613 chars ≈ 53.5M tokens |
| user share of corpus | **12.6%** |
| user tokens per session | ≈306 |
| category-3 subset | 30 tenants, 1,496 sessions, ≈459k user tokens |

A disposition belongs to whoever stated it, so the pass reads **user turns only** — an 8× reduction —
and skips the model call entirely for an episode whose user text is under `PROFILE_MIN_CHARS = 32`. A
lexical prefilter was evaluated and **rejected**: a first-person preference regex matches 5.7% of user
turns (680k tokens) but recovers a matching user turn in only **9 of the 30** category-3 gold sessions.
30% recall is too lossy to build a measurement on.

**The 8× saving on input tokens did not become an 8× saving on wall time, because extraction is not the
cost — consolidation is.** Measured on the category-3 build, from the `unit_complete` audit events:

| quantity | measured |
|---|---|
| tenants | 30 |
| sessions | 1,496 |
| episodes written | 9,786 |
| profile records | **3,750 live + 151 superseded** |
| per-tenant wall | mean **224 s**, median 193 s, range 139–493 s |
| total wall | 108 min (including ~12 min of reader restarts) |
| extrapolated to 500 tenants | **≈31 GPU-hours** |

On one representative unit the 329 profile extractions finished in under a minute, and the remaining
~3 minutes went to consolidating ~125 accepted candidates — each a Qdrant neighbour search plus a
consolidator model call. The plan's `[INFERENCE]` estimate of ≈3.5 h for the full 500 was wrong by ~9×
for this reason.

**Raising concurrency does not help: it hurts.** At `MYELIN_READER_SLOTS=8 --concurrency 8` two units of
47 and 54 sessions took 371 s and 424 s; at `MYELIN_READER_SLOTS=2 --concurrency 4` three units of 49–51
sessions took 162 s, 195 s and 205 s. The 3090 is already saturated at 4 in flight, so more slots only
split the KV cache — `-c 65536 -np 8` gives 8,192 tokens per slot against 32,768 at `-np 2` — and
destroy the prompt-prefix reuse the consolidator's shared system prompt depends on. `build --concurrency`
exists so this is measurable rather than folklore; its default is unchanged at 4.

## 4. The base arm is **not** M19's baseline, and that matters

The plan expected the category-3 base arm to reproduce M19's **26.67**. It measured **30.00** — one
question. Before reading any arm that had to be explained, and it decomposes cleanly into three
independent single-question effects, none of which is an arm:

| question | M19 → base | cause |
|---|---|---|
| `505af2f5` | 0 → 1 | **judge nondeterminism.** The response string is byte-identical in both runs and was graded differently. |
| `75f70248` | 0 → 1 | **genuine retrieval change.** One evidence item swapped: an irrelevant `assistant:` turn about novels was replaced by `user: What are some simple ways to keep my living room dust-free, especially with a cat that sheds a lot?` |
| `afdc33df` | 1 → 0 | **evidence ordering.** The same six records, ranks 2 and 6 swapped by a reranker tie-break; the turn carrying the gold moved from emitted position [1] to position [5] and the reader declined instead of answering. |

The second of those exposes a false premise in the plan's step 14. Restricting the build to 30 tenants
does reproduce those tenants' *episodes* byte-for-byte — ids are v5 over `(namespace, natural key)` — but
this build **also writes 3,750 profile records, and they are indexed**. They therefore compete in the
retrieval pool even with `ComposeConfig::profile` off: **13 of the base arm's 180 evidence items (10 of
30 rows) are profile records retrieved as ordinary evidence.**

So the four arms remain internally valid — one store, paired comparisons, the differences are the
switches — but **the base arm must not be quoted against M19's 26.67**, and any future store built with
the profile pass on is a different store for every question, not only preference ones.

The third effect is worth keeping in view independently: one question flipped from correct to declined
purely because two tied records swapped rank. At n = 30 that is 3.3 points of pure position noise, which
is a further argument for why the §1 bar is as high as it is.

## 5. The mechanism reaches the reader

`bench` does not otherwise record this, and M12/M14 both lost runs to a silently unwired flag. Smoke
test on two questions with both switches on:

```
myelin-eval bench --corpus longmemeval-s --categories 3 --limit 2 \
  --profile --profile-clause --collection myelin_lme_s_pref \
  --ledger data/longmemeval_s_pref.ledger --out /tmp/m20_smoke
```

Observed — the first evidence item of every row begins `[profile] ` and names that tenant's own
dispositions:

```
8a2466db | [profile] The user avoids McDonald's healthier options.; The user prefers McDonald's
           burgers and fries.; The user is looking for recommendations for hole in the wall
           restaurants in San Francisco.; …                                     (7 items, was 6)
06878be2 | [profile] The user prefers high-quality and safe baby products.; The user is looking for
           practical and functional baby gifts.; The user is interested in personalized baby
           blankets.; The user shops at Buy Buy Baby for baby …                 (7 items, was 6)
```

The second row is the milestone's own example question — photography accessories — and the block it got
is about baby products. That is §7's finding, visible in the smoke test before a single arm was scored.

Per-arm evidence size confirms the block is the only difference: mean items 6.00 (base, B) versus 7.00
(A, A+B), mean evidence 11,174 versus 11,662 characters — **+4.4% prompt, +1 item**.

## 6. Latency

`visible_of_kind` adds one indexed SQLite read per recall. `PLAN.md` §7.1's p95 < 100 ms is a product
SLO, so the cost is measured rather than assumed:

| arm | `query_p50_seconds` | `query_avg_seconds` |
|---|---|---|
| base | 0.32 | 0.37 |
| A | 0.31 | 0.33 |
| B | 0.32 | 0.32 |
| A+B | 0.32 | 0.32 |

The kind-filtered scope read is free at this resolution.

## 7. What this milestone actually found

**A disposition store is not enough; a disposition *selection rule* is the open problem.** Three
measurements say so together, and none of them was predictable from the plan:

1. The write pass produces **~124 dispositions per tenant**, not the handful `PROFILE_MAX_RECORDS = 8`
   was sized for. The prompt is not at fault — the records are genuine and well-formed
   (`The user prefers organic produce, meat, and dairy products.`,
   `The user is interested in acoustic guitar fingerpicking techniques.`). LongMemEval_S haystacks are
   ~50 sessions of dense first-person conversation, and that is simply how much a person states.
2. Recency (`t_ingested DESC`) selects the wrong 6.5%: gold-content recall **0.042**, nine questions at
   zero.
3. Arm B moves declines 11 → 7 without arm A, so the reader *will* use a stated preference when it is
   told to and one is present. The binding constraint is that the right one is usually not present.

This is consistent with M16's finding that retrieval, not the reader, is the usual binding constraint —
and it refines it: for preferences the binding constraint is *selection among in-scope material*, a
third thing that is neither retrieval nor generation.

The obvious repair — rank the profile block by relevance to the question — is deliberately **not**
applied here. It contradicts the mechanism's own thesis ("retrieved because it is about the user, not
because it matches the question"), it is not what §1 pre-registered, and tuning a selection rule until
n = 30 moves is exactly the failure M12/M13's precedent exists to prevent. It is a hypothesis for a
future milestone with a rule fixed in advance, and the honest framing is that the null measured here is
the baseline it would have to beat.

## 8. Why the full-500 off-target arm was not run

`ComposeConfig::profile` costs tokens in every prompt and a scope read on every query, so §1's second
clause requires an arm that would ship on to lose no more than 1.0 point over the full 500. **No arm
passed the first clause, so no default flips, so there is nothing for the off-target check to gate.**
Plan step 17 says the same in advance: "Run it after step 19 has confirmed the mechanism works on 30
tenants." It did not.

The cost of running it anyway is now measured rather than estimated: **≈31 GPU-hours** to build the
500-tenant profile store (§3), for an off-target bound on a mechanism that ships off. That is not a
defensible use of the card.

`longmemeval_s.judge_score.n500` is therefore **unchanged at 56.40 against the 80.80 bar (−24.40)**,
re-confirmed by re-running `myelin-eval standing` after the arms. No number in `docs/sota` moves, which
is the correct outcome for a milestone that shipped two nulls.

## 9. What was built

| piece | where |
|---|---|
| `RecordKind::Profile`, `ALL`/`as_str`/`parse` | `crates/myelin-core/src/model/record.rs` |
| I4 binds profiles as well as semantics | `record.rs::requires_lineage` |
| `CandidateKind::Profile` + `key_prefix` | `crates/myelin-core/src/pipeline/extract.rs` |
| `PROFILE_SYSTEM`, `profile_schema`, `Extractor::extract_profile` | `extract.rs` |
| `EpisodeDraft::render_speaker` | `crates/myelin-core/src/pipeline/ingest.rs` |
| `WritePath::extract_profiles` / `profile_speaker` / `assert_one`, `WriteStats::profiles`/`profile_ms` | `crates/myelin-core/src/pipeline/write.rs` |
| `Ledger::visible_of_kind` over a shared `visible_inner` | `crates/myelin-core/src/store/ledger.rs` |
| `ComposeConfig::profile`, `profile_item`, `PROFILE_MAX_RECORDS` | `crates/myelin-core/src/pipeline/compose.rs` |
| profile fetch in `recall` and `investigate`, `RecallTrace::profile_records` | `retrieve.rs`, `investigate.rs` |
| `READER_PREFERENCE_CLAUSE`, `--profile`, `--profile-clause` | `crates/myelin-eval/src/bench.rs`, `main.rs` |
| `build --question-types`, `--concurrency`, profile pass on for LongMemEval_S | `crates/myelin-eval/src/build.rs`, `main.rs` |
| MCP tool 10 `profile`, `remember --as_profile` | `crates/myelin-mcp/src/server.rs` |

`key_prefix` is the one edit that could have invalidated an existing corpus: `record_id` is a v5 UUID
over `(namespace, natural key)`, so `Semantic` and `Procedural` keep the literal `"fact"` prefix and only
`Profile` uses `"profile"`. Changing it for `Semantic` would have changed every id in the 162k-record
LoCoMo corpus and invalidated every vector.

### The companion surface, verified live over stdio

Against a scratch ledger and collection, driven through the real MCP protocol:

| step | observed |
|---|---|
| `tools/list` | 10 tools, `profile` among them |
| `profile` on an empty store | `{"records": []}` — an empty array, not an error |
| `remember` with `as_profile: true` | `{"candidates": 1, "added": 1, "profiles": 1}` |
| `profile` | one record, `kind: "profile"`, the asserted text |
| `remember as_profile` with a contradicting preference | `{"added": 0, "updated": 1, "profiles": 1}` — a `Delta::Update`, not a second row |
| `profile` again | the new disposition only; the old one is gone |
| `profile` for another tenant | `{"records": []}` |

That is the supersession the companion story depends on, and it costs no new write path: `assert_one`
skips extraction only, and reuses the same `Consolidator`, the same four-op delta and the same audit
event as the corpus pass.

### Tests worth keeping

- `compose.rs::the_profile_block_leads_the_set_and_never_launders_trust` — placement at the head, nil
  `record_id`, and trust equal to the weakest tier among the summarised records.
- `compose.rs::the_profile_block_is_bounded_and_skipped_when_empty` — `k` still bounds records, the cap
  bounds the block, and an empty slice emits no item.
- `invariants.rs::visible_of_kind_narrows_without_loosening` — the kind filter inherits the tenant
  filter, I2, I3 and the `t_invalid` clause; a superseded and a quarantined disposition are both absent.
- `invariants.rs::i4_exempts_primary_observations_and_binds_abstractions` — a profile with no lineage is
  refused at the SQLite boundary; one derived from a real episode is admitted.
- `write.rs::the_profile_pass_never_calls_the_model_for_a_speaker_who_said_nothing` — the cost control,
  with a positive control so the assertion cannot pass by the pass being unreachable.
- `write.rs::the_profile_prompt_carries_only_the_chosen_speakers_turns`.
- `main.rs::the_two_profile_arms_never_share_a_directory`.

## 10. Reproduction

```bash
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
ssh big gpu-tenant claim coding
ssh big "MYELIN_READER_SLOTS=2 MYELIN_READER_CTX=65536 bash -s" < ops/big/serve-models.sh
ssh -N -L 5810:127.0.0.1:5810 -L 5813:127.0.0.1:5813 big &   # reader and reranker are firewalled

# the 30-tenant store (~108 min). --repair is required at the end: superseding a
# preference retracts the predecessor in the ledger, and its Qdrant point must go
# with it (151 stale_points on this build).
cargo run --release -p myelin-eval -- build --corpus longmemeval-s \
  --question-types single-session-preference \
  --collection myelin_lme_s_pref --ledger data/longmemeval_s_pref.ledger --repair

# warm the reranker: the cross-encoder breaks score ties differently on the first
# request after a cold start. --categories 3 is required, or --limit 2 picks two
# category-1 tenants that have no memory in this store and warms nothing.
cargo run --release -p myelin-eval -- bench --corpus longmemeval-s --categories 3 \
  --limit 2 --out /tmp/warm \
  --collection myelin_lme_s_pref --ledger data/longmemeval_s_pref.ledger

# four arms, service up across all of them
C="--corpus longmemeval-s --categories 3 --collection myelin_lme_s_pref --ledger data/longmemeval_s_pref.ledger"
cargo run --release -p myelin-eval -- bench $C                            --out runs/m20_pref_base
cargo run --release -p myelin-eval -- bench $C --profile                  --out runs/m20_pref_armA
cargo run --release -p myelin-eval -- bench $C --profile-clause           --out runs/m20_pref_armB
cargo run --release -p myelin-eval -- bench $C --profile --profile-clause --out runs/m20_pref_armAB

for n in base armA armB armAB; do
  cargo run --release -p myelin-eval -- judge   --run runs/m20_pref_$n --category 3
  cargo run --release -p myelin-eval -- rescore --run runs/m20_pref_$n --scorer judge
done

python3 crates/myelin-eval/adapters/paired_ci.py \
  runs/rescored/m20_pref_armA_judge runs/rescored/m20_pref_base_judge
# …and armB vs base, armAB vs base, armAB vs armA, armAB vs armB
```

## 11. Grounding

- PERMA — `10.48550/arXiv.2603.23231`. The benchmark this layer is shaped for; its three dimensions
  (Task Completion, Preference Consistency, Informational Confidence) are what a profile layer should be
  scored on, and §7's selection problem is exactly what Preference Consistency measures.
- MemMachine — `10.48550/arXiv.2604.04853`. A typed persona/profile memory written at consolidation
  time, which is the shape `RecordKind::Profile` implements.
- TiMem — `10.48550/arXiv.2601.02845`. Profile records maintained across sessions with supersession,
  which is what `Delta::Update` gives this layer for free.
- MemPro-15 — `10.48550/arxiv.2606.00619`, registry row
  `longmemeval_s.judge.mempro15.qwen3_30b`, per-category cells at L235. The 80.00 this stratum is
  measured against.
