# M21 — evidence selection, and the instrument that made it visible

M20 closed by naming the *selection* rule, not the record type, as the cause of its null: the
`[profile]` block was wired correctly, reached the reader, and showed 6.5% of what the tenant knew
because it was chosen by recency. This milestone reproduces that conclusion on the two largest
strata left on the board, builds the instrument that turns "retrieval recall" from an inference into
a reported artifact, and measures two selectors against a rule fixed in advance.

## Verdict

**All three switches ship off. The instrument ships on, and the finding is the largest unexploited
lever measured on this project: question-conditioned selection over the reranked pool is worth
+3.8 judged points over all 500 LongMemEval_S questions (95% CI [+1.0, +6.6], p = 0.0087) — and
there is currently nowhere it is allowed to run.**

| switch | mechanism | decision |
|---|---|---|
| `ComposeConfig::mmr_lambda` | maximal marginal relevance over the deduped candidates | **off — measured harm** |
| `RetrieveConfig::select_sufficient` | ask the model which candidates jointly answer, inside `recall` | **off — clears §1's bar, but `PLAN.md` §7.1 forbids an LLM in `recall`** |
| `InvestigateConfig::select_sufficient` | the same selector inside the agentic loop | **off — measured +0.0 at +1.87 s/query** |

Three results, in the order they were measured:

1. **MMR is not a null, it is a regression.** Gold-turn recall falls 0.658 → 0.550 (multi-session)
   and 0.658 → 0.471 (temporal-reasoning); judged, −5.3 ([−11.3, +0.0]) and **−9.8 ([−15.8, −3.8],
   p = 0.0018)**. The cause is measured in §6: the memories that *jointly* answer one question
   resemble **each other** 1.60× more than they resemble the rest of the composed set, so a
   redundancy penalty suppresses exactly the signal that identifies co-evidence.
2. **The sufficiency selector clears the bar, and cannot be spent.** Recall 0.658 → **0.838** and
   0.658 → **0.809**, within reach of the 0.852 ceiling that `k = 25` needs four times the items to
   touch. Judged +7.5 ([−0.8, +15.8]) and **+6.0 ([+1.5, +10.5], p = 0.0081)**; over all 500,
   **+3.8 ([+1.0, +6.6], p = 0.0087)** with no stratum regressing. §1 exempted it from `recall`
   before it was run, and the one path it was allowed to default on turned out to neutralise it.
3. **`investigate` neutralises it.** The same selector, same store, same stratum, inside the
   agentic loop: **exactly +0.0 (CI [−3.8, +3.8], p = 1.0000)** at **+1.87 s per query**. The loop
   unions its probes into a 60-record pool and re-composes, so reordering one probe's candidates
   changes which items enter a pool that was going to hold them anyway (recall 0.653 → 0.660).

Two defects were found and fixed on the way, both of which had been silently corrupting reported
numbers: `coverage` refusing pre-M19 artifacts (§3) and `standing` publishing the best *arm* rather
than the shipped configuration (§8).

## 1. The rule, fixed before any arm

Written into this file before the first bench arm was launched, as M12–M20 each did.

> A selector ships **on** only if, judged, on **both** LongMemEval_S multi-session (n = 133) and
> temporal-reasoning (n = 133): the paired mean difference against the same-store base is positive
> with a 95% CI excluding zero on at least one stratum, and has a CI lower bound ≥ −1.0 on the
> other; **and** the same arm over the full 500 loses no more than 1.0 point overall with a CI
> excluding zero. Otherwise it ships off as a measured null.
>
> `RetrieveConfig::select_sufficient` is exempt from shipping on in `recall` regardless of outcome:
> `PLAN.md` §7.1 specifies that path as "target p95 < 100 ms, **no LLM in the loop**". Its number is
> reported as a ceiling probe and, if it clears the bar, it becomes the `investigate` default only.

At n = 133, judged 0/1, a paired bootstrap CI excludes zero at roughly **+7 points**. The two strata
are **not** pooled to reach it, and the population is not widened.

A further rule, fixed at the same time, decides λ without spending a judged question on it:

> λ is chosen on LongMemEval_S `knowledge-update` (n = 78, category 6) — a stratum this milestone
> does not score — by highest mean gold-turn recall from `myelin-eval coverage`, which is
> deterministic and needs no model call. On a tie, the **larger** λ wins. If no λ beats the base
> run's recall on that stratum, the MMR arm is a measured null before any judging: record it, skip
> arms A-ms and A-tr, run arm B only, and say so in the verdict.

**Both clauses fired as written.** The selector cleared the first rule; the λ clause fired for MMR
and is honoured in §5. Every interval below is quoted verbatim from
`crates/myelin-eval/adapters/paired_ci.py`, whose `sorted(set(sa) & set(sb))` ordering is the only
one that reproduces the published bounds.

## 2. The diagnosis

Per-stratum judged scores from `runs/rescored/m19_lme_s_full_judge` and
`runs/rescored/m19_locomo_full_judge`, against MemPro-15's Qwen3-30B row in `docs/sota/registry.json`
— the same-answer-model-class comparison G2 is stated against:

| stratum | n | ours | theirs | gap | weighted |
|---|---|---|---|---|---|
| LongMemEval multi-session | 133 | **38.35** | 75.94 | −37.6 | **−10.0** |
| LongMemEval temporal-reasoning | 133 | **33.83** | 71.43 | −37.6 | **−10.0** |
| LongMemEval ss-preference | 30 | 26.67 | 80.00 | −53.3 | −3.2 |
| LongMemEval knowledge-update | 78 | 74.36 | 82.05 | −7.7 | −1.2 |
| LoCoMo multi-hop | 282 | 57.80 | 75.17 | −17.4 | −3.2 |
| LoCoMo open-domain | 96 | 29.17 | 70.83 | −41.7 | −2.6 |

The two strata losing −10.0 weighted points each are three times `single-session-preference`, which
M20 worked. And for both, **the evidence the reader needed was already retrieved and then discarded
by the top-k truncation in `compose`.**

`RetrieveConfig::rerank_depth` is 25 and `retrieve.rs:365` sets `depth = rerank_depth.max(query.budget.k)`,
so the k=6 and k=25 arms see an *identical* 25-candidate reranked pool:

| run | mean gold-turn recall | complete gold evidence composed |
|---|---|---|
| `runs/m19_lme_t_base` (k=6) | **0.662** | 70 / 132 |
| `runs/m19_lme_t_armWinv` (`--mode investigate`) | 0.653 | 69 / 132 |
| `runs/m19_lme_t_armW25` (k=25) | **0.852** | **106 / 132** |

Two alternatives were already measured and bound the design: **breadth alone** (`--k 25`) is +3.8
judged with a CI spanning zero, and **the existing agentic loop** changed the evidence on 53 of 133
rows for exactly +0.0 at 6.9× the latency. The problem is neither retrieval nor generation: it is
**selecting a small set that jointly covers the question** out of a pool that already contains it.

## 3. The instrument: `myelin-eval coverage`

`crates/myelin-eval/src/coverage.rs`. Both corpora annotate the answer-bearing material themselves —
LongMemEval_S flags the evidence turn with `has_answer`, LoCoMo cites evidence turns by `dia_id` — so
gold coverage of the composed evidence is computable **offline, deterministically, with no model
call.** It reads a `bench` run's `per_question.jsonl` and writes `<run>/coverage.json`.

Not a second `evidence-audit`: that module reads the vendored LME-V2 harness's `HarnessRow` and asks
an LLM whether evidence was *sufficient*. This is arithmetic against the dataset's own labels, and it
hard-errors on LME-V2, which carries no per-turn annotation.

Match rule, pinned by unit test: strip a leading `[YYYY-MM-DD]` stamp, then a gold unit counts as
found when its trimmed text exceeds 30 characters and its first 80 characters occur as a substring of
any stripped evidence value. The floor exists because `"Thanks!"` occurs in every evidence set ever
composed; the prefix exists because `compose` emits a consolidated record, not the verbatim turn.

It reproduces the three M19 figures exactly, and `artifact_tests` pins them:

```
runs/m19_lme_t_base    annotated 132  mean gold recall 0.662
runs/m19_lme_t_armWinv annotated 132  mean gold recall 0.653
runs/m19_lme_t_armW25  annotated 132  mean gold recall 0.852   all = 106
```

The tests are behind a `myelin-eval` feature, `artifacts`, because `runs/**/per_question.jsonl` and
`/data/` are both gitignored — the plan assumed those artifacts were committed and they are not.
Run with `cargo test -p myelin-eval --features artifacts`.

### Two things the instrument found that the plan did not predict

**A run that recorded no evidence now hard-errors instead of reporting 0.000.**
`ScoredQuestion::evidence` is `#[serde(default)]` and was added in M19, so all twelve pre-M19 run
directories parse cleanly with an empty evidence list on every row. The first LoCoMo run reported
`mean gold recall 0.000` over 1,977 annotated rows of `runs/locomo_recall` — indistinguishable from a
run that retrieved nothing, on a stratum scoring 0.699 token F1. `read_rows` now refuses any run
where no row carries evidence, naming the cause; `runs/m19_locomo_full` is the correct artifact, and
a test asserts the refusal on the other.

**LoCoMo's `dangling_evidence` is 9, not 0.** All nine are malformed strings in `data/locomo10.json`
itself: `'D8:6; D9:17'` (two ids in one field), `'D'` (truncated), `'D:11:26'` (extra colon),
`'D30:05'` (zero-padded index), and five multi-id strings in `conv-49`. Counted, excluded from the
denominator, and reported — never silently dropped, which would have overstated recall.

### LoCoMo coverage, `runs/m19_locomo_full` — the credibility check

| cat | stratum | n | recall | all | partial | none |
|---|---|---|---|---|---|---|
| 1 | multi-hop | 281 | **0.472** | 63 | 147 | 71 |
| 2 | temporal | 320 | 0.815 | 255 | 12 | 53 |
| 3 | open-domain | 89 | **0.403** | 30 | 15 | 44 |
| 4 | single-hop | 841 | 0.933 | 784 | 2 | 55 |

Multi-hop and open-domain — the two LoCoMo strata with the largest judged gaps (−17.4, −41.7) — are
the two with the lowest gold-turn recall. Single-hop, 1.3 points off SOTA, is at 0.933.

## 4. What was built

- **`crates/myelin-eval/src/coverage.rs`** + `Command::Coverage`. Offline; no Qdrant, ledger or
  reader. Writes `<run>/coverage.json` and prints a per-category table with the mean score *inside
  each coverage bucket*, which is the column that says whether coverage is the binding constraint.
- **`ComposeConfig::mmr_lambda: Option<f32>`** and `compose::mmr_select`. Relevance is the
  candidate's **rank**, not its score: `score` is a cross-encoder logit on one corpus and an RRF
  score on another, and mixing either with a cosine in [−1, 1] is the scale error `tau_abstain`
  documents. Ties break on the lower index, the first pick is `kept[0]` unconditionally, and a
  candidate with no vector is judged on relevance alone.
- **`crates/myelin-core/src/pipeline/select.rs`** — `Selector`, a constrained-decoding
  `{"keep": [...]}` call over the reranked candidates. Never fails the caller: out-of-range,
  duplicate, over-long and empty results all degrade to rank order. `EmptyCompletion` still
  propagates (R7 — a zero-byte body is a dead model, not "no memory helps").
- **`Retriever::with_llm`** + `RetrieveConfig::select_sufficient`, applied after the rerank sort and
  before the `k * 3` materialisation, as a **stable partition** — nothing is dropped, so a bad
  selection costs rank positions and never evidence. `RecallTrace::selected` / `select_ms`.
- **`InvestigateConfig::select_sufficient`**, applied through a local `Retriever` view (a stack copy
  — every field is a shared reference or a `Copy`-shaped config).
- **`--mmr` / `--select-sufficient`** bench arms, persisted on `BenchRun` and read back in
  `rescore_run`, with `_mmr70` / `_sel` run-directory segments and a `--mmr` range guard: the
  selector clamps, so `--mmr 70` would silently run the base arm under an arm's directory name.

## 5. λ, fixed on a stratum this milestone does not score

`bench --categories 6` (knowledge-update), coverage only — no judge calls.

| λ | mean gold recall (n = 72 annotated) | complete-gold rows |
|---|---|---|
| base (no MMR) | **0.933** | 63 |
| 0.9 | 0.933 | 63 |
| 0.7 | 0.907 | 60 |
| 0.5 | 0.819 | 50 |
| 0.3 | 0.646 | 27 |

**No λ beats the base.** λ = 0.9 ties it by doing nothing — it selects the identical evidence *set*
on 72 of 78 rows — while every λ that acts is strictly worse, monotonically in the diversity weight.
§1's null clause fires: MMR is a measured null before any judging.

The MMR arms were still *run and judged* so the table below has no gap, but the decision was fixed on
recall, before any verdict existed. The free mechanism check on the scored strata agrees with the
λ stratum rather than contradicting it, which is what makes the null safe to act on:

| arm | stratum | recall vs base |
|---|---|---|
| A-ms, λ = 0.5 | multi-session | 0.658 → **0.550** |
| A-tr, λ = 0.5 | temporal-reasoning | 0.658 → **0.471** |

## 6. Why MMR loses, measured

Two facts, both computed from the artifacts:

**Damage is concentrated on questions that need more than one memory.** Rows whose gold coverage MMR
reduced, by gold-unit count:

| gold units | multi-session | temporal-reasoning |
|---|---|---|
| 1 | 0 / 4 (**0.0%**) | 10 / 39 (25.6%) |
| 2–3 | 29 / 97 (**29.9%**) | 27 / 87 (**31.0%**) |
| 4+ | 3 / 24 (12.5%) | 0 / 6 (0.0%) |

**The memories that jointly answer one question resemble each other.** Over the temporal-reasoning
base run's composed evidence, mean token-Jaccard:

| pair | n | mean | median |
|---|---|---|---|
| gold ↔ gold | 58 | **0.2235** | 0.2026 |
| gold ↔ other | 471 | **0.1399** | 0.1268 |

**1.60×.** A redundancy penalty is therefore not a neutral tie-break on this task — it is a penalty
aimed at co-evidence. Diversity is the wrong objective for multi-hop evidence selection, because the
memories that together answer a question are topically coupled by construction. This is the
generalisable result of the milestone and it is why a better *heuristic* was never going to work.

## 7. The arms

LongMemEval_S, judged, store `myelin_longmemeval_s` / `data/longmemeval_s.ledger`. Bases re-run
fresh against the same store rather than reusing M19's, and they reproduce M19's full-run
per-stratum figures exactly (38.35, 33.83).

| arm | cat | n | coverage | judged | Δ vs base | 95% CI | p |
|---|---|---|---|---|---|---|---|
| base-ms | 4 | 133 | 0.658 | 38.35 | — | — | — |
| A-ms `--mmr 0.5` | 4 | 133 | **0.550** | 33.08 | −5.3 | [−11.3, +0.0] | 0.0831 |
| B-ms `--select-sufficient` | 4 | 133 | **0.838** | **45.86** | **+7.5** | [−0.8, +15.8] | 0.0769 |
| base-tr | 5 | 133 | 0.658 | 33.83 | — | — | — |
| A-tr `--mmr 0.5` | 5 | 133 | **0.471** | 24.06 | −9.8 | [−15.8, −3.8] | **0.0018** * |
| B-tr `--select-sufficient` | 5 | 133 | **0.809** | **39.85** | **+6.0** | [+1.5, +10.5] | **0.0081** * |

`*` = 95% CI excludes zero.

The selector against MMR, which is the comparison that says whether the shortfall was the heuristic:

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| B over A — multi-session | +12.8 | [+4.5, +21.1] | 0.0024 * |
| B over A — temporal-reasoning | +15.8 | [+9.0, +23.3] | <0.0001 * |

**§1's first clause is satisfied**: the CI excludes zero on temporal-reasoning, and the other
stratum's lower bound is −0.8, inside the −1.0 floor.

Token F1 beside the judged column, never alone (M9 flagged it; M19 measures it at 0.0456 against a
judged 0.2667 on a comparable stratum): base-ms 0.3445 → B-ms 0.4093; base-tr 0.2782 → B-tr 0.3044.

## 8. Off-target — the full 500

| | base | selector | Δ | 95% CI | p |
|---|---|---|---|---|---|
| **overall (n = 500)** | **56.60** | **60.40** | **+3.8** | **[+1.0, +6.6]** | **0.0087** * |
| non-abstention (470) | 54.3 | 58.3 | +4.0 | [+1.1, +7.0] | 0.0085 * |
| abstention (30) | 93.3 | 93.3 | +0.0 | [+0.0, +0.0] | 1.0000 |
| 1 single-session-user (70) | 94.29 | 94.29 | +0.0 | [−4.3, +4.3] | 1.0000 |
| 2 single-session-assistant (56) | 96.43 | 96.43 | +0.0 | [+0.0, +0.0] | 1.0000 |
| 3 ss-preference (30) | 26.67 | 30.00 | +3.3 | [−6.7, +16.7] | 0.7812 |
| 4 multi-session (133) | 38.35 | 45.86 | +7.5 | [−0.8, +15.8] | 0.0769 |
| 5 temporal-reasoning (133) | 33.83 | 39.85 | +6.0 | [+1.5, +10.5] | 0.0081 * |
| 6 knowledge-update (78) | 75.64 | 75.64 | +0.0 | [−6.4, +6.4] | 1.0000 |

Mean gold-turn recall over all 500: **0.768 → 0.873**. §1's second clause — "loses no more than 1.0
point overall" — is satisfied by a *gain* with a CI excluding zero.

**The unchanged categories are not untouched.** The selector changed the composed evidence *set* on
18.6% of category 1, 3.6% of category 2 and 28.2% of category 6, and the score did not move. That is
genuine off-target neutrality: it rearranges evidence where the reader was already right, and the
reader stays right.

### `standing` was publishing the arm

Because `runs/m21_full_sel` is a complete 500-question artifact scoring 60.40 against the default
configuration's 56.60, and `standing`'s selection rule was *complete → largest population → best
value*, the standing table selected it. Before M21 every full-population artifact was a shipped-default
run and the rule could not misfire.

Fixed: `Ours::arm` is computed from the switches `BenchRun` already records, and the order is now
*complete → **shipped configuration** → largest population → best value*. An arm is still listed
under `candidates`, and is still selected when it is the only artifact for a metric. Pinned by
`a_higher_scoring_arm_never_displaces_the_shipped_configuration`, which also asserts that an
*incomplete* default does not displace a complete arm — a partial artifact is not a configuration.

## 9. Transfer — LoCoMo multi-hop

Store `myelin_locomo` / `data/locomo.ledger`, category 1, n = 282.

| | base | selector | Δ | 95% CI | p |
|---|---|---|---|---|---|
| judged | 58.16 | 54.96 | −3.2 | [−7.8, +1.1] | 0.1825 |
| coverage | 0.478 | 0.367 | −0.111 | — | — |

**It does not transfer.** The judged difference is a null — the CI includes zero, so the
"contradiction" contingency's trigger (one stratum up, the other down *with a CI excluding zero*)
does not strictly fire — but the recall direction is unambiguous and adverse. The mechanism that
converts a reranked pool into a covering set on LongMemEval_S does the opposite on LoCoMo, whose
records are short consolidated facts rather than whole session turns. Reported as a limit on the
finding, not as a footnote.

## 10. The path it was allowed to default on, and why it is off

§1 exempted `select_sufficient` from `recall` before any arm ran, leaving `investigate` as its only
possible home. So that path was measured rather than assumed — `--mode investigate`, category 5,
same store, the two arms differing only in the switch.

| | base | selector | Δ | 95% CI | p |
|---|---|---|---|---|---|
| judged (n = 133) | 36.84 | 36.84 | **+0.0** | [−3.8, +3.8] | 1.0000 |
| coverage | 0.653 | 0.660 | +0.007 | — | — |
| latency p50 | 2.60 s | 4.47 s | **+1.87 s** | — | — |

Exactly zero, at 72% more latency. The cause is the loop's own shape: `step_k` is 10 and `max_pool`
is 60, so probe results are **unioned** across steps and re-composed by `investigate` itself.
Reordering one probe's candidates changes which items enter a pool that was going to hold them
anyway — recall moves 0.653 → 0.660, seven thousandths.

Shipping this default on the strength of the `recall`-path numbers would have shipped pure cost.
`InvestigateConfig::select_sufficient` therefore defaults **false**, with the number that killed it
in its doc comment.

### Latency, for the record

| path | base p50 / avg | selector p50 / avg | cost |
|---|---|---|---|
| LongMemEval_S `recall` | 0.43 / 0.46 s | 1.49 / 1.51 s | +1.06 s |
| LoCoMo `recall` | 0.25 / 0.26 s | 1.07 / 1.08 s | +0.82 s |
| LongMemEval_S `investigate` | 2.60 / 2.57 s | 4.47 / 4.26 s | +1.87 s |

MMR is free: 0.44 → 0.42 s p50 on multi-session, 0.43 → 0.43 s on temporal-reasoning. It costs no
model call, which is the one thing that could have recommended it.

These are end-to-end query times across an SSH tunnel to a remote GPU, not the `PLAN.md` §7.1
in-process budget; the comparison that matters here is the **delta**, and a second model call per
query is a category change for a path specified as "no LLM in the loop".

## 11. Standing

| metric | bar | ours | gap | before M21 |
|---|---|---|---|---|
| `longmemeval_s.judge_score.n500` | 80.80 (MemPro-15, Qwen3-30B) | **56.60** | −24.20 | 56.40 / −24.40 |
| `locomo.judge_score.n1540` | 77.85 (MemPro-15, Qwen3-30B) | 69.87 | −7.98 | 69.87 / −7.98 |

The shipped configuration did not move, and that is the honest reading: every mechanism this
milestone built ships off. The 0.20 change on LongMemEval_S is the fresh paired base against M19's,
a single question on a reranker tie-break (the effect M20 documented). Recomputed from the artifacts
by `myelin-eval standing`; no number here was hand-entered.

## 12. Reproduction

```bash
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
ssh big gpu-tenant claim coding
ssh big "MYELIN_READER_SLOTS=2 MYELIN_READER_CTX=65536 bash -s" < ops/big/serve-models.sh
ssh -N -L 5810:127.0.0.1:5810 -L 5813:127.0.0.1:5813 big   # second shell, leave up

# warm the cross-encoder: it breaks score ties differently on the first
# request after a cold start
cargo run --release -p myelin-eval -- bench --corpus longmemeval-s --categories 4 \
  --limit 2 --collection myelin_longmemeval_s --ledger data/longmemeval_s.ledger --out /tmp/warm

C="--corpus longmemeval-s --collection myelin_longmemeval_s --ledger data/longmemeval_s.ledger"
cargo run --release -p myelin-eval -- bench $C --categories 4 --out runs/m21_ms_base
cargo run --release -p myelin-eval -- bench $C --categories 4 --mmr 0.5 --out runs/m21_ms_mmr
cargo run --release -p myelin-eval -- bench $C --categories 4 --select-sufficient --out runs/m21_ms_sel
# …and --categories 5 into runs/m21_tr_*; the full 500 with no --categories
# into runs/m21_full_{base,sel}; --mode investigate into runs/m21_inv_*

for d in runs/m21_*; do cargo run --release -p myelin-eval -- coverage --run $d; done
cargo run --release -p myelin-eval -- judge   --run runs/m21_ms_sel --category 4
cargo run --release -p myelin-eval -- rescore --run runs/m21_ms_sel --scorer judge
python3 crates/myelin-eval/adapters/paired_ci.py \
  runs/rescored/m21_ms_sel_judge runs/rescored/m21_ms_base_judge

cargo test --workspace --features myelin-core/integration   # 219 green
cargo test -p myelin-eval --features artifacts              # pins the instrument
```

## 13. What this leaves

The finding is not "selection does not work". It is that **selection works and there is nowhere to
put it**: +3.8 over 500 questions with a CI excluding zero, from a mechanism whose only
implementation costs a model call, in a system whose fast path is specified to have none and whose
slow path dissolves the effect by pooling.

That names the next milestone precisely, and it is a choice rather than a task. Three commitments
currently hold and this measurement says they are inconsistent: `PLAN.md` §7.1 (`recall` has no LLM
in the loop), §14 (no cross-encoder training), and the +3.8 points. §6 closes the cheap escape — a
set-level judgement cannot be approximated by a distance, so any untrained, model-free selector is
a distance in disguise and inherits MMR's defect.

The narrowest option is a set-level head on the cross-encoder that already runs on every query and
already encodes (query, document) jointly: a new head on a resident model, which keeps §7.1 intact
and breaches §14 for one head rather than for the reranker. Its training data is cheap but **not**
free-standing — `bench` persists the composed evidence, not the selector's `keep[]` or the
25-candidate pool, so the labels must be regenerated by persisting `keep[]` on `RecallTrace` or by
re-running retrieval, which is deterministic against a fixed store. The population is already
defined: **1,048 selector calls over 782 distinct questions** across both corpora
(`runs/m21_{ms,tr,full,mh}_sel`).

Either way the target is fixed and the instrument to check it against exists and is free:
0.658 → 0.838 recall, +3.8 judged, at `recall` latency — and rejectable on recall alone, before a
judge call is spent, which is how this milestone killed MMR.
