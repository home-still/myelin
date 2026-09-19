# M19 — making time computable: the temporal arm on G2

M18 closed with five unsupported gates and one sentence about where to aim next. This milestone aimed there
and found that **the single largest deficit in the whole project was a read-path omission, not a capability
gap**: the store knew the date of every memory, the reader was shown that date on every item, and nothing
resolved *"last Tuesday"* against it.

## Verdict

**Three mechanisms ship on by default. On LoCoMo's temporal stratum two of them are worth +42.8 points
together — the largest single measured win in this project's history — and on LongMemEval's the third is worth
+6.8 on top of a corpus-level date repair this milestone also had to make.**

| arm | mechanism | LoCoMo cat-2 (n=321) | Δ vs 20.22 | 95% CI | p | decision |
|---|---|---|---|---|---|---|
| A | `ComposeConfig::resolve_relative` — annotate each memory's relative time references with the date they resolve to against that memory's own `t_valid` | **57.84** | **+37.6** | [+32.2, +43.2] | <0.0001 | **ships on** |
| B | the date clause in `READER_SYSTEM` — tell the reader to do the resolution itself | 34.57 | +14.3 | [+10.2, +18.7] | <0.0001 | **ships on** |
| A+B | both | **63.01** | **+42.8** | [+37.3, +48.5] | <0.0001 | **the shipped configuration** |
| D | `ComposeConfig::timeline` — a dated index of the selected records | 20.54 | +0.3 | [+0.0, +0.9] | 0.7530 | null here; gated on LongMemEval, where it **ships on** |

The pre-registered rule was Δ ≥ +5.0 with a 95% CI excluding zero, against a stratum ceiling of +31.8 (the
103 of 321 questions the diagnosis names). Both arms clear it by a wide margin, and **the effect exceeds the
ceiling**, so the mechanism also repaired questions outside the diagnosed set. The marginal tests say both
mechanisms are needed and neither is a proxy for the other:

- the prompt clause on top of the annotation: **+5.2** [+2.5, +8.1], p < 0.0001
- the annotation on top of the prompt clause: **+28.4** [+23.4, +33.7], p < 0.0001

So the finding is not "the reader was withholding the resolution" — arm B alone recovers only a third of what
arm A does. A 9B reader shown `[2023-07-20] … last Tuesday` *and told to resolve it* still gets 34.57; shown
`[2023-07-20] … last Tuesday (last tuesday = 2023-07-18)` it gets 57.84. Resolution belongs in the memory
system. Chronos (`10.48550/arXiv.2603.16862`) reaches the same conclusion from the write side, and reports the
events calendar alone as a 58.9% gain over its own baseline.

| arm | LongMemEval cat-5, judged (n=133) | Δ vs 27.07 | 95% CI | p | decision |
|---|---|---|---|---|---|
| D — `timeline` | **33.83** | **+6.8** | [+3.0, +11.3] | 0.0005 | **ships on** |
| W — `--k 25` | 30.83 | +3.8 | [−2.3, +9.8] | 0.2739 | fails the rule; `k` never ships (R4) |
| W — `--mode investigate` | 27.07 | **+0.0** | [−3.8, +3.8] | 1.0000 | exactly no change, at 6.5× the latency |
| D + `--k 25` | 37.59 | +10.5 | [+3.0, +18.0] | 0.0055 | breadth is not the mechanism — see §5 |

**And a defect found by the attempt, not by the measurement:** LongMemEval_S's session timestamps had never
parsed, so all 162,181 of its records carried the *build date* as `t_valid`. Every one of the 500 memories had
been showing the reader `[2026-09-15]` since M13, and the 133 temporal-reasoning questions were being asked of
a corpus with no time in it. Fixed, corpus re-ingested, details in §6.

## 1. Why temporal, and not something else

M18's judged columns make the G2 gap decomposable, and the decomposition was not in any document before now.
Both sides are judged 0/1 by the same rule (`standing::judged`: the local judge on answered answerable rows,
the deterministic decline rule on abstention items), so the columns are commensurate.

| stratum | n | ours (judged) | MemPro-15 @ Qwen3-30B | gap | gap × n / corpus |
|---|---|---|---|---|---|
| LoCoMo single-hop | 841 | **82.88** | 83.47 | −0.6 | 0.3 |
| LoCoMo multi-hop | 282 | 58.16 | 75.17 | −17.0 | 3.1 |
| LoCoMo open-domain | 96 | 31.25 | 70.83 | −39.6 | 2.5 |
| **LoCoMo temporal** | 321 | **23.05** | 67.60 | **−44.6** | **9.3** |
| LongMemEval ss-user | 64 | **93.75** | 92.86 | +0.9 | 0 |
| LongMemEval ss-assistant | 56 | **96.43** | 98.21 | −1.8 | 0.2 |
| LongMemEval knowledge-update | 72 | 63.89 | 82.05 | −18.2 | 2.6 |
| LongMemEval multi-session | 121 | 33.88 | 75.94 | −42.1 | 10.2 |
| **LongMemEval temporal-reasoning** | 127 | **18.11** | 71.43 | **−53.3** | **13.5** |
| LongMemEval ss-preference | 30 | 23.33 | 80.00 | −56.7 | 3.4 |

Ours: `runs/locomo_recall` and `runs/lme_s_recall` with their `judge_verdicts.json`, recomputed for this
document and reproducing cell for cell. Theirs: registry rows `locomo.judge.mempro15.qwen3_30b` and
`longmemeval_s.judge.mempro15.qwen3_30b`, per-category cells at `10.48550_arxiv.2606.00619` L235. The last
column converts each gap into points of the 15.19 / 28.80 full-set gap.

Two readings matter. First, **single-fact recall is already at parity with a SOTA system backed by a 30B
model** — 82.88 vs 83.47, 93.75 vs 92.86, 96.43 vs 98.21 — so the gap is not "our retrieval is worse". Second,
**temporal alone is 22.8 of the 44 combined gap points**, more than any other stratum on either benchmark.

## 2. The diagnosis, per question rather than in aggregate

### LoCoMo: the answer is a relative expression the gold resolves

Of the 321 temporal questions, 247 are judged wrong. In **103** of them our answer is a *bare relative
expression* carrying no absolute date while the gold names one. The rule is mechanical and reproducible: the
row is judged wrong, `response_raw` matches a relative-expression pattern, `response_raw` contains no year and
no month name, and `answer_gold` contains one.

| question | gold | our answer (M9 baseline) |
|---|---|---|
| When did Melanie paint a sunrise? | `2022` | `Last year.` |
| When did Melanie run a charity race? | `The sunday before 25 May 2023` | `Last Saturday` |
| When is Melanie planning on going camping? | `June 2023` | `Next month.` |
| When did Caroline meet up with her friends, family, and mentors? | `The week before 9 June 2023` | `Last week.` |
| When did Caroline join a new activist group? | `The Tuesday before 20 July 2023` | `Last Tuesday.` |

**90 of those 103** have a resolvable relative expression in their own gold evidence turn, and the gold is
exactly that expression resolved against the session date: evidence *"I just joined a new LGBTQ activist group
… last Tuesday"* in the session dated 20 July 2023 → gold `The Tuesday before 20 July 2023`. Across the whole
corpus, **431 of LoCoMo's 5,882 turns** carry an expression the resolver's closed grammar accepts, and **223 of
the 321** temporal questions have one in their evidence.

(The plan's pre-registration put these at 102 / 92 / 456 / 233 with a slightly wider detector. The
difference is the detector's boundary — `a few weeks ago`, `last night`, `two weekends later` are relative but
*unresolvable*, so they are outside the shipped grammar and are counted here only on the answer side. Nothing
structural depends on the boundary.)

### LongMemEval: the question asks for an interval and the reader declines

Different failure, reproduced exactly as pre-registered. Of the 127 answerable temporal-reasoning questions,
**61 are duration questions** (`how long`, `how many days/weeks/months/years`, `how much time`). **45 of the 61
are declined outright** and only **2** are judged correct. These need two dated endpoints and a subtraction;
at `k = 6` against records whose dates were all identical (§5) the reader had nothing to subtract.

## 3. What was built

All of it is read-path or prompt-side — no store rebuild was *needed* for the mechanism, which is what let
four LoCoMo arms fit in 22 minutes of GPU.

- **`myelin_core::time`** — the M14 temporal grammar moved out of the eval crate so the read path can call it
  (`DayRange`, `Temporal`, `TemporalKind`, the tokenizer, the parser, `parse_gold`, `parse_response`). The
  eval crate keeps `temporal_score` and its precision-not-F1 rationale and re-exports the types, so no caller
  changed. A grammar that exists twice is a grammar that diverges.
- **`time::resolve_relative(text, anchor)`** — the other half of what the gold grammar refuses. M14's module
  docs say open-ended and unanchored relatives return `None` because *"an unanchored offset has no reference
  day in the gold string"*; in a **record** there is one, and it is `record.validity.t_valid`. The resolver's
  grammar is closed and fails the same way: `yesterday|today|tomorrow`,
  `last|past|this|next <weekday|week|weekend|month|year>`, `<N> <unit> ago|back`, at most three per text,
  deduped by phrase, in order of occurrence. `a few weeks ago` (no count), `recently`, `earlier`,
  `since <X>`, and `two weeks before <event>` are all refused rather than guessed.
- **`time::is_interval_question(text)`** — deliberately generous (a false positive costs one evidence item, a
  false negative costs the mechanism), and deliberately excludes `when`, which 77.9% of the LoCoMo stratum
  contains and which would make the switch untargeted. It fires on 76 of the 133 LongMemEval temporal rows.
- **`ComposeConfig::resolve_relative`** (arm A) — appends ` (<phrase> = <YYYY-MM-DD>)` or
  ` (<phrase> = <from>..<to>)` after the record's text. *After*, never inside: rewriting the record's words
  would stop the evidence item quoting the memory it came from. The budget is charged on
  `approx_tokens(&candidate.record.text)`, so the annotation is free and the two arms are the same selection
  with different text — the M13 discipline that makes two runs comparable.
- **`ComposeConfig::timeline`** (arm D) — one synthetic item at the tail,
  `[timeline] 2023-05-06 +0d · rug delivered…; 2023-05-13 +7d · rearranged…  (span 28 days)`, built from the
  *selected* records ascending by `t_valid`. Its `record_id` is nil and its source is the literal doc
  `timeline` (it is a view, not a memory, and must not resolve in the ledger), and its trust is the **weakest**
  tier among the records it summarises — a synthetic view of untrusted material must not launder it upward.
  Gated per query in `Retriever::recall`, the only place that sees both the question and the compose config.
- **the date clause in `READER_SYSTEM`** (arm B).
- **`ScoredQuestion::evidence`** — the labelled evidence strings the reader was shown, on every row, always
  on. Nothing on disk recorded this before, so every claim of the form "retrieval found the record and the
  reader failed to use it" was an inference. §4 turns one into arithmetic.
- **`bench --categories`** — a stratum filter applied *before* retrieval, so a 321-question arm costs nothing
  for the 1,665 questions it skips. `BenchRun::categories` makes the artifact self-describing, and
  `standing::bench_metrics` now drops any metric whose population is empty, so a temporal-only arm cannot
  publish `locomo.abstention_accuracy.n446` as 0.00 from zero adversarial rows.
- **`rescore --scorer judge`** — reads `<run>/judge_verdicts.json` into the `score` column so
  `adapters/paired_ci.py`, which pairs on `score`, can compare judged runs. Needed because both deterministic
  scorers are wrong for LongMemEval's temporal stratum: only 4 of its 127 golds parse as durations (the order
  questions' golds are event names), and token F1 credits "Three weeks" against "Two weeks" at 0.5 — the exact
  failure M14 existed to remove. A missing verdict on an *answered* row is a hard error naming the question
  id, matching `standing::judged`'s rule; a decline scores 0.0 and an adversarial item scores by the decline
  rule.

## 4. The mechanism check, on real artifacts

Both of these are reported independently of the pass/fail rule, because an arm can move answer *shape*
without moving the score and that distinction is the finding.

**V2 — was the evidence even there?** For each of the 103 target questions, does the gold evidence turn's text
appear in the arm-A run's persisted `evidence`? **101 of 103 (98.1%)**. The two misses are `conv-44#30` and
`conv-50#17`. Over the whole 321-question stratum it is 269 of 321 (83.8%). The claim "retrieval found the
record; nobody resolved the reference" is now measured rather than asserted.

**V3 — did the answer shape change?** Of the 103, answers carrying an absolute date (a year or a month name)
instead of a bare relative expression: **90 (87.4%) under arm A, 94 (91.3%) under arm A+B**. The temporal score
on exactly those 103 rows goes **13.89 → 75.03 (arm A) → 80.58 (arm A+B)**.

| question | gold | was | now (A+B) |
|---|---|---|---|
| When did Melanie paint a sunrise? | `2022` | `Last year.` | `2022` |
| When did Melanie run a charity race? | `The sunday before 25 May 2023` | `Last Saturday` | `2023-05-20` |
| When is Melanie planning on going camping? | `June 2023` | `Next month.` | `2023-06-01..2023-06-30` |
| When did Caroline meet up with her friends, family, and mentors? | `The week before 9 June 2023` | `Last week.` | `2023-06-02..2023-06-08` |
| When is Caroline going to the transgender conference? | `July 2023` | `This month.` | `July 2023` |
| When did Caroline have a picnic? | `The week before 6 July 2023` | `Last week` | `2023-06-29..2023-07-05` |
| When did Caroline go to a pride parade during the summer? | `The week before 3 July 2023` | `Last Friday` | `2023-07-15` |
| When did Caroline join a new activist group? | `The Tuesday before 20 July 2023` | `Last Tuesday.` | `2023-07-18` |
| When did Caroline attend a pride parade in August? | `The Friday before 14 August 2023` | `Last Friday.` | `2023-08-11` |
| When did Caroline draw a self-portrait? | `The week before 23 August 2023` | `Last week.` | `2023-08-16..2023-08-22` |

Row 7 is the useful failure: the answer moved from a relative expression to an absolute date and is *still*
wrong, because the reader resolved a different expression in a different record than the gold's. The
mechanism supplies dates; it does not choose which event the question means.

## 5. LongMemEval: the dated index, and what it is *not*

The LongMemEval arms were run after the corpus repair of §6, so every one of them — including the baseline —
has correct dates, `resolve_relative` on and the date clause in the prompt. The comparison is therefore
strictly "what does the dated index add", and it is judged 0/1 by `rescore --scorer judge` because both
deterministic scorers are invalid here (only 4 of the 127 golds parse as durations, and token F1 credits
"Three weeks" against "Two weeks" at 0.5).

**The +6.8 is the index, not retrieval breadth, and the whole of it lands on the questions the switch
targets.** Three facts pin that:

- the index is worth **+6.8** at `k = 6` ([+3.0, +11.3], p = 0.0005) and **+6.8** again at `k = 25`
  ([+1.5, +12.8], p = 0.0174) — the same effect at four times the evidence;
- breadth alone is **+3.8** ([−2.3, +9.8], p = 0.27) and breadth on top of the index is **+3.8**
  ([−2.3, +10.5], p = 0.30) — a CI spanning zero both times. So arm W fails the pre-registered rule, `k`
  stays a per-call parameter (R4), and the full-set run below is at `k = 6` so M9–M18 stay comparable;
- `--mode investigate` scored **exactly** what `recall` scored (27.07 vs 27.07, Δ 0.0 [−3.8, +3.8]) at
  p50 2.58 s against 0.40 s. Two steps of an agentic loop bought nothing on this stratum at 6.5× the latency.

Split by question shape, the targeting is as clean as it gets — and the same table is the clearest statement
of what the §6 date repair was worth:

| the 61 duration questions | judged | declines |
|---|---|---|
| M9 store, every record stamped with the build date | 3.28 | **45 of 61** |
| dates repaired (this milestone's baseline) | 13.11 | 19 |
| dates repaired + dated index | **27.87** | 20 |

| the 66 other answerable temporal questions | judged |
|---|---|
| baseline | 33.33 |
| + dated index | **33.33** |

Unchanged to the digit. `is_interval_question` fires on 76 of the 133 rows, the index is appended only there,
and off-target it costs exactly nothing — which is the property that made it safe to default on despite being
a null on LoCoMo (+0.3 [+0.0, +0.9]: only 20 of those 321 questions ask for a duration).

The declines column is the mechanism in one number. 45 of the 61 duration questions were declined outright
when every record carried the same date; with real dates 19 are, and the index does not reduce that further
(20) — it converts *attempted* answers into *correct* ones. Both halves were needed: dates the reader can
subtract, and a presentation that puts the two endpoints side by side.

One contract changed and is now pinned by a test: `ComposeConfig::k` bounds **records**, not items. An
interval question at `k = 6` emits seven items — 75 of arm D's 133 rows did. The index is derived from records
already admitted, so it widens no threat surface, but a consumer sizing a buffer off `k` needs to know.

## 6. The defect the arm found: LongMemEval_S had no time in it

Arm D's first smoke run on LongMemEval produced this:

```
[timeline] 2026-09-15 +0d · assistant: What a great experience! The Museum of Mo; 2026-09-15 +0d · assistant:
Mummification played a crucial role in ancient Eg; …  (span 0 days)
```

Six records, one date, span zero. The cause is one missing format string: `parse_locomo_time` handled LoCoMo's
`1:56 pm on 8 May, 2023` and nothing else, while LongMemEval_S writes `2023/05/20 (Sat) 02:21`. The parser
returned `None`, `WritePath` fell back to the ingest time, and **every one of the 162,181 records carried the
build date**. Consequences, all of them silent until now:

- `ComposeConfig::stamp_valid_time` has defaulted on since M13, so all 500 LongMemEval memories showed the
  reader `[2026-09-15]`. The M9 and M13 LongMemEval_S numbers were measured against a corpus whose dates were
  all identical and all wrong.
- The 133 temporal-reasoning questions — the largest single gap on either benchmark, 13.5 of the 28.80 — were
  being asked of a corpus with no time in it.
- Arm D could not have worked there, and arm A would have produced *false* resolutions (anchoring
  `last Tuesday` to 2026-09-15).

The fix is `parse_session_time`, which accepts both corpora's formats, with both pinned by tests. The repair
of the existing store took three attempts and the two failures are worth recording:

1. **In-place `t_valid` correction is refused by design.** A `correct_t_valid` on the ledger fails on the
   `record_immutable` trigger — `I1: record content, scope and provenance are immutable` — which names
   `t_valid` explicitly. That is the right answer to a repair that rewrites history, so the method was
   reverted rather than the invariant weakened.
2. **`Delta::Update` is the wrong instrument too**: it writes a *new* record and retires the old one, which
   would change 162,181 ids and invalidate every vector, incidence row and link, to fix one column.

So the corpus was re-ingested from scratch, which is the sanctioned operation: 500 tenants, 162,181 episodes,
57 min wall, `reconcile: clean`, `t_valid` now spanning 2021-05-03 … 2024-02-20 and **zero** records stamped
2026. Two things came out of the attempt and stayed:

- `build_longmemeval_s` now has the resume guard `build_locomo` has had since M3. Without it a crash at unit
  341 of 500 cost a full re-walk: the write path is *safe* to repeat (episode ids are content-derived, so an
  existing record is a no-op) but not *cheap* — 29 s per already-ingested unit, four hours over the corpus.
- The crash itself was ollama's bge-m3 GPU runner dying under VRAM pressure
  (`health resp: Get "http://127.0.0.1:45487/health": EOF`). Serving ollama's own bge-m3 blob under llama.cpp
  with `CUDA_VISIBLE_DEVICES= -ngl 0 --pooling cls` costs **0 MiB of VRAM**, ~1.2 GB of host RAM, and produces
  numerically identical vectors (cosine 0.999999635, max elementwise 7.8e-05 against the GPU path), so it is
  the right embedder on a shared card.

## 7. Full-set, judged, and what it does to the standing table

The winning configuration is all three mechanisms on at `k = 6` / `recall`, so M9–M18 stay comparable. Both
corpora were run full-set and judged by the local reader with M14's rubric, and both artifacts are joined
into `runs/standing` by `myelin-eval standing`.

**LoCoMo, 1,986 questions** (`runs/m19_locomo_full`, judged in
`runs/rescored/m19_locomo_full_judge`):

| | date-aware scorer | judged |
|---|---|---|
| answerable (n=1,540) | 51.38 → **59.98**, paired **+8.6** [+7.0, +10.2] | 62.66 → **69.87**, paired **+7.2** [+5.5, +9.0] |
| abstention (n=446) | 69.96 → 69.96, +0.0 [−2.7, +2.7] | unchanged |

Per category, judged, with the date-aware scorer's paired interval beside it — the point being that the
effect is **where the diagnosis said it would be and nowhere else**:

| category | n | judged M9 | judged M19 | Δ (scorer, paired 95% CI) |
|---|---|---|---|---|
| multi-hop | 282 | 58.16 | 57.80 | +0.06 [−1.45, +1.50] |
| **temporal** | 321 | **23.05** | **60.44** | **+42.48 [+36.60, +48.08]** |
| open-domain | 96 | 31.25 | 29.17 | −1.87 [−5.19, +0.39] |
| single-hop | 841 | 82.88 | 82.16 | −0.27 [−1.41, +0.75] |
| adversarial | 446 | 69.96 | 69.96 | +0.00 [−2.69, +2.69] |

Not one off-target category moves with an interval excluding zero. Open-domain's −1.9 is the largest and its
CI spans zero at n = 96; it is reported rather than rounded away, and it is the one cell to re-check if this
mechanism is ever widened.

**LongMemEval_S, 500 questions** (`runs/m19_lme_s_full`, judged in
`runs/rescored/m19_lme_s_full_judge`): judged over the whole 500 — the population MemPro's own `Avg.`
reproduces as — **52.00 → 56.40, paired +4.4 [+1.6, +7.2], p = 0.0032**; over the 470 answerable rows
49.15 → 54.04, **+4.9** [+1.9, +7.9].

| question type | n | judged M9 | judged M19 | Δ |
|---|---|---|---|---|
| single-session-user | 70 | 94.29 | 94.29 | 0 |
| single-session-assistant | 56 | 96.43 | 96.43 | 0 |
| single-session-preference | 30 | 23.33 | 26.67 | +3.3 |
| multi-session | 133 | 39.10 | 38.35 | −0.8 |
| **temporal-reasoning** | 133 | **21.80** | **33.83** | **+12.0** |
| **knowledge-update** | 78 | **66.67** | **74.36** | **+7.7** |

Knowledge-update was not a target and gained 7.7 points. That is the date repair, not the annotation: a
question about the *current* value of a changed fact is answerable only if the versions can be ordered, and
until §6 every version carried the same date. It is the same stratum M13 found oldest-first ordering helped
(+8.0, CI [−1.7, +17.6]) and for the same reason.

The one cost: abstention accuracy on the 30 `_abs` items fell 96.67 → 93.33 (−3.3, CI [−10.0, +0.0],
p = 0.72) — one item that used to be declined is now answered. A reader with dates it can compute on is
slightly more willing to answer; at n = 30 that is one question and the interval touches zero.

**The standing table, before and after** (`runs/standing/standing.md`,
`myelin-eval standing --gate` still exits 1):

| gate metric | bar | M18 | M19 | gap before → after |
|---|---|---|---|---|
| `locomo.judge_score.n1540` | 77.85 (MemPro-15 @ Qwen3-30B) | 62.66 | **69.87** | −15.19 → **−7.98** |
| `longmemeval_s.judge_score.n500` | 80.80 (same) | 52.00 | **56.40** | −28.80 → **−24.40** |

The other three gates are untouched by this milestone and unchanged: LME-V2-Small at −34.99, the LAFS gain
tied at 0.00, and G3's defended ASR at 12.50 against a 10% bar. Two non-gate rows also moved: our LoCoMo
date-aware column is 51.64 → 59.98, and against **Mem0 as its own paper reports it** (66.88) we are now
**+2.99 ahead** — the second row in the registry where we lead a published number, though still `claim: no`
because the graders differ.

## 8. Reproducing

```bash
# the four LoCoMo arms, as run (pre-cutover, when both mechanisms were still flags)
B=~/.cargo-target-shared/global/release/myelin-eval
$B bench --corpus locomo --categories 2 --scorer temporal --resolve-dates               --out runs/m19_locomo_t_armA
$B bench --corpus locomo --categories 2 --scorer temporal --date-prompt                 --out runs/m19_locomo_t_armB
$B bench --corpus locomo --categories 2 --scorer temporal --resolve-dates --date-prompt --out runs/m19_locomo_t_armAB
$B bench --corpus locomo --categories 2 --scorer temporal --timeline                    --out runs/m19_locomo_t_armD

# the intervals, always from the CLI: its own id ordering determines the bounds
.venv/bin/python crates/myelin-eval/adapters/paired_ci.py runs/m19_locomo_t_armA  runs/rescored/locomo_recall_temporal
.venv/bin/python crates/myelin-eval/adapters/paired_ci.py runs/m19_locomo_t_armAB runs/m19_locomo_t_armA
```

After the cutover `--resolve-dates` and `--date-prompt` are gone: `ComposeConfig::resolve_relative` defaults
on and `bench` deliberately does **not** override it (a bench run that silently disabled the shipped
mechanism because a flag defaulted false would measure a configuration nobody runs — the same treatment
`stamp_valid_time` has had since M13), and the date clause is part of `READER_SYSTEM`. `--timeline` survives
because arm D is still a switch. The off state is reproducible from these artifacts and from git.

## 9. The systems this milestone is grounded in

Four systems were downloaded, converted and indexed for this milestone. Conversion was blocked for most of
the session — olmOCR needs ~14.5 GB of free VRAM and the card had 0.7–8.1 GB free while a household voice
assistant held 9.8 GB and another tenant's processes held 5.8 GB — and ran at the end in one 11-minute window
once the card cleared: Chronos 15 pages, Memanto 13, APEX-MEM 20, PERMA 38, all embedded into the local
corpus. Five registry rows were added and then upgraded from `abstract_only` to `paper` with the verbatim
quotes below.

| row | value | what the converted source says |
|---|---|---|
| `longmemeval_s.judge.chronos_high` | **95.60** | The highest LongMemEval_S number known to this project, and the population is *stated*: "the LongMemEvalS benchmark comprising 500 questions across six categories". It also reproduces arithmetically — the per-category cells for Chronos High (98.57 ss-user, 100 ss-assistant, 100 ss-preference, 88.72 multi-session, 95.50 temporal-reasoning, 100 knowledge-update) micro-average over the official 70/56/30/133/133/78 to **exactly 95.60**. "High" is the paper's own label for "advanced frontier models, such as Opus 4.6 and Gemini 3 Pro"; Chronos Low, on "GPT-4o, or similar", is 92.60. The **judge is never named** — the converted text contains no "judge" or "grader" — so that class is inferred and flagged unverified. |
| `longmemeval_s.judge.memanto` | 89.8 | n = 500 is stated: "the full 500-question suite under the standard S setting". So is the grader: **"Claude Sonnet 4 serves as the LLM judge throughout all evaluation stages."** The answer model is never named — "inference model selection" is listed as one of five ablation stages without saying which model produced the headline. |
| `locomo.judge.memanto` | 87.1 | Beats MemPro-15's 84.93. Same stated judge; the LoCoMo population is *not* stated — "LoCoMo is evaluated on its standard split" — so n = 1540 stays this registry's assumption and whether the 446 adversarial items were dropped is unverified. |
| `locomo.judge.apex_mem` | 88.88 | The highest LoCoMo number in the registry. Mechanism: a property graph of temporally grounded events, append-only temporal evolution, and query-time conflict resolution, with its `GRAPHSQL` tool doing "duration calculations" and "temporal ordering via validity intervals" — the write-time half of what M19 does at read time. |
| `longmemeval_s.judge.apex_mem` | 86.2 | Same sentence as the row above. |

The two APEX-MEM rows keep a hard caveat: the PDF converted to 20 pages but the conversion's *retrievable*
text covers the front matter only — abstract, introduction, related work, architecture — so their quote is
the abstract's results sentence rather than a table row, and the per-category cells, question counts, answer
model and judge could not be read. `standing` rates all five `caveat-judge`, none carries `gate: true`, and
against them we now stand at −39.20 (Chronos High), −33.40 / −17.23 (Memanto) and −29.80 / −19.01 (APEX-MEM).

PERMA is the fourth, and it is not a comparison row: it is the benchmark M20 will be measured against, and it
is now converted and indexed (38 pages, 50 chunks).

## 10. M20, with its arithmetic already known

The companion-facing preference layer is the next lever and it is the last large stratum that has never been
worked: **single-session-preference, 30 questions, judged 26.67 against MemPro's 80.00** — 3.4 points of the
remaining 24.40-point LongMemEval gap, the largest *relative* deficit on either benchmark, with 13 of the 30
declined and a token F1 of 4.6 that M9 already flagged as "not scorable this way". It did not move here (+3.3
on n = 30, one question) because a preference is not a dated fact: it is a disposition that accumulates and
changes across sessions, and nothing in the store represents one.

The shape M20 should build is the one PERMA (`10.48550/arXiv.2603.23231`) evaluates and MemMachine
(`10.48550/arXiv.2604.04853`) and TiMem (`10.48550/arXiv.2601.02845`) implement: a **typed profile record**
written at consolidation time and always composed, so a preference is retrieved because it is *about the
user* rather than because it lexically matches the question. PERMA is the benchmark to measure it against —
"temporally ordered interaction events spanning multiple sessions and domains, with preference-related queries
inserted over time", explicitly built because existing evaluations "interleave preference-related dialogues
with irrelevant conversations, reducing the task to needle-in-a-haystack retrieval" — which is exactly what
our 26.67 is measuring today. Its copy is converted and indexed (§9), 38 pages.

Two things M19 leaves M20 to build on. The **time machinery is now in `myelin-core`**, so a profile record
can carry *when* a preference held and `resolve_relative` will date the turn that stated it. And the
**write-path date bug is fixed**, so a preference-evolution measurement on LongMemEval_S or PERMA will not be
asking a corpus with no time in it — which, had §6 gone unnoticed, is exactly what M20 would have done.
