# What we need to research to reach SOTA

A hand-off brief for a literature-review agent. It states the target precisely, the arithmetic of what must
move, the constraints any answer has to survive, what has already been measured (so it is not re-derived),
and the questions that are genuinely open — each with the falsifiable test it would have to pass here.

Everything numeric below is from `runs/standing/standing.json` and the run artifacts it joins, generated
after M19. Sources for published claims are `docs/sota/registry.json`, every row carrying a verbatim quote.

---

## 1. What "SOTA" means for this project, exactly

Two distinct problems, and they are usually conflated:

**(a) Beating the numbers.** Three benchmark metrics with fixed populations:

| metric | population | ours | project gate | best published | who |
|---|---|---|---|---|---|
| `locomo.judge_score.n1540` | LoCoMo's 1,540 non-adversarial questions | **69.87** | 77.85 | **93.05** | EverMemOS |
| `longmemeval_s.judge_score.n500` | all 500, `_abs` items included | **56.40** | 80.80 | **95.60** | Chronos High |
| `lme_v2_small.overall_full_set` | LME-V2-Small's 451 questions | **39.91** | 74.90 | 74.90 | AgentRunbook-C |

Plus two non-accuracy gates: `lafs_gain > 0` (currently exactly **0.00** — both our operating points are
dominated by the reference frontier's fastest, 51.0 @ 0.2 s) and MINJA defended ASR ≤ 10% (currently
**12.50% [5.5–26.1]**, five attacks out of 40 at k=6 — the figure `standing` reads from
`runs/attack_live_m18/attack_live.json`; M15's earlier run of the same condition measured 6/40 = 15.0%
[7.1–29.1], and both are above the bar).

**(b) Being *allowed* to claim it.** Of 30 registry comparisons, **20 come back `caveat-judge`** — our
grader's class differs from the paper's — and **exactly one is claimable** (GAM's LoCoMo token F1, `+13.15`).
(24 of the 30 published rows are scored by a frontier-API judge; the other 20/30 figure is the *verdict*
count in `runs/standing/standing.json`, which is the number this paragraph is about.)
Our grader is a local Qwen3.5-9B; theirs are GPT-4o-mini, GPT-4.1-mini, GPT-5.2 and Claude Sonnet 4.
`standing`'s rule refuses `claim_allowed` across judge classes, and that refusal is correct — it is also why
**a perfect system here still could not claim SOTA today**. Problem (b) is at least as important as (a) and
is much cheaper to solve. It is RQ1.

---

## 2. The arithmetic: which strata must move, and by how much

Micro-averaging is linear in the per-stratum counts, so the gap decomposes exactly. "→bar" is the answers
gained by matching MemPro-15 @ Qwen3-30B's own per-category profile — the project gate. "→100%" is the total
headroom that exists.

**LoCoMo — need +122.9 correct answers (69.87 → 77.85 over n=1,540):**

| stratum | n | ours | gate profile | →gate | →100% |
|---|---|---|---|---|---|
| multi-hop | 282 | 57.80 | 75.17 | **+49** | 119 |
| temporal | 321 | 60.44 | 67.60 | +23 | 127 |
| open-domain | 96 | 29.17 | 70.83 | **+40** | 68 |
| single-hop | 841 | 82.16 | 83.47 | +11 | 150 |
| | 1540 | | | **+123** | 464 |

26% of all remaining headroom. **No single stratum perfected clears it**: multi-hop's entire headroom is 119
< 123, open-domain's is 68. It needs multi-hop *and* open-domain together, or a general mechanism.

**LongMemEval_S — need +122.0 correct answers (56.40 → 80.80 over n=500):**

| stratum | n | ours | gate profile | →gate | →100% | →Chronos High |
|---|---|---|---|---|---|---|
| ss-user | 70 | 94.29 | 92.86 | −1 | 4 | +3 |
| ss-assistant | 56 | 96.43 | 98.21 | +1 | 2 | +2 |
| ss-preference | 30 | 26.67 | 80.00 | +16 | 22 | +22 |
| multi-session | 133 | 38.35 | 75.94 | **+50** | 82 | +67 |
| temporal-reasoning | 133 | 33.83 | 71.43 | **+50** | 88 | +82 |
| knowledge-update | 78 | 74.36 | 82.05 | +6 | 20 | +20 |
| | 500 | | | **+122** | 218 | +196 |

**56% of all remaining headroom**, and multi-session + temporal-reasoning are 170 of the 218 available. This
is the harder of the two benchmarks by a wide margin and the one where the literature is furthest ahead
(Chronos High reports 88.72 multi-session and 95.50 temporal-reasoning).

**Conclusion for the reviewer:** the research target is **multi-session / multi-hop assembly** first
(99 answers across both benchmarks at the gate profile, 201 of headroom), then **open-domain retrieval**
(40), then **preference** (16), then **knowledge-update** (6). Temporal still needs +23/+50 even after M19.

---

## 3. Constraints any proposed mechanism must survive

These are not preferences. A technique that violates them is not usable here regardless of its published
numbers, and the reviewer should filter on them.

- **Reader: Qwen3.5-9B Q4_K_XL, local, open weights.** No frontier API key exists in this project. Most
  published SOTA is frontier-backed; the gate is deliberately MemPro's *own* Qwen3-30B row for that reason.
  The single cleanest comparison in the registry is **AgentRunbook-R at 58.60 on LME-V2-Small with the same
  Qwen3.5-9B reader** against our 39.91 — mechanisms demonstrated at that model scale are worth far more
  than mechanisms demonstrated on GPT-5.
- **One RTX 3090 (24 GB), shared.** A household voice assistant holds ~10 GB on a rolling keep-alive and
  another tenant intermittently holds ~5 GB; the practical working budget is **~7 GB VRAM**. Host RAM is
  31 GB with no useful swap, and the reader has been OOM-killed mid-run. **No training, no fine-tuning, no
  30B+ model, no multi-model ensemble.**
- **Cost of a write-path arm.** Read-path arms are minutes (321 questions ≈ 5 min). Write-path arms need a
  re-ingest: LongMemEval_S **57 min**, LoCoMo ~100 min, LME-V2-Small hours. A mechanism requiring three
  write-path variants to tune is not affordable; one requiring a single build is.
- **Architectural invariants** (`PLAN.md` §3): R1 fixes the evidence wire form to `[{type, value}]`; R4
  makes `k` and `mode` per-call, so nothing may hard-code them; **I1 forbids in-place record edits** — a
  SQLite trigger enforces it, and M19 confirmed the hard way that "just correct the field" is refused, so
  any versioning scheme must be append-only; C12 requires per-tenant isolation (zero cross-tenant leaks,
  held since M11).
- **Latency matters for one gate.** LAFS needs a *Pareto-improving* point, not just accuracy: currently
  39.91 @ 12.76 s and 35.70 @ 1.97 s, both dominated by 51.0 @ 0.2 s. An accuracy win that costs seconds
  per query moves G2 and does nothing for G1.

---

## 4. Already measured — do not re-propose these

Nine mechanisms have been measured against pre-registered rules with paired CIs. The nulls are as
informative as the wins and the reviewer should treat them as closed:

| mechanism | result | milestone |
|---|---|---|
| PPR / graph route as a third fusion channel | no gain in any category; −0.7 on LongMemEval_S (CI [−1.5, −0.1]) | M12 |
| chronological evidence order | coin flip: 668 answers change, 208 better / 211 worse | M13 |
| `<today>` in the prompt | +0.2 on the target stratum; real gain is +3.1 abstention | M13 |
| date-aware scorer (instrument, not mechanism) | token F1 was *inflating* LoCoMo temporal by 8.03 | M14 |
| injection adjudicator | ASR 77.5% → 12.50% [5.5–26.1] (5/40, k=6; M15 measured 6/40), still misses the ≤10% gate | M15, re-run M18 |
| wider evidence (`k = 25`) on LongMemEval temporal | +3.8, CI spans zero | M19 |
| `investigate max_steps = 2` on LongMemEval temporal | **exactly 0.0**, at 6.5× latency | M19 |
| reader-side prompt instruction to resolve dates | +14.3 alone, but only +5.2 on top of doing it in memory | M19 |
| **resolving relative dates in the evidence itself** | **+37.6 on LoCoMo temporal; shipped** | M19 |

Two structural findings to build on rather than rediscover:

- **We are retrieval-limited, not reader-limited.** M16 measured S = P(evidence sufficient | answer wrong)
  = **7.4%** at `recall` k=25 and 12.2% at `investigate`. *Insufficient + wrong* is the largest cell
  everywhere (52–57% of answerable). A perfect reader over today's evidence reaches 38.8–44.6% on LME-V2
  against a 51.0 bar. **No reader-side or prompt-side change can close these gaps.**
- **Doing the computation in memory beats telling the reader to do it.** M19's marginals: the annotation is
  worth +28.4 on top of the instruction, the instruction only +5.2 on top of the annotation. Generalise
  with care, but it is the strongest prior we have about where work belongs.

---

## 5. Open research questions

Each is stated as: the quantity that must move, what we already know, what the literature would have to
supply to be actionable, and how we would falsify it cheaply.

### RQ1 — How does anyone make a defensible SOTA claim with an open-weights judge?
**Blocks every claim regardless of quality; cheapest item on this list.**
Known: our judge is Qwen3.5-9B with M14's pinned rubric; `docs/measurements/m9-judge-panel.md` measures it
at **κ 0.8813 against a frontier model and slightly harsher**, so our numbers are conservative, not
flattering. That argument is written down but has never been tested against how the field actually
adjudicates this.
Needed from the literature: (i) what the *official* LoCoMo and LongMemEval protocols mandate, and whether
any published deterministic or open-weights-judge protocol is accepted as comparable; (ii) whether papers
report judge-sensitivity, and how much a judge swap is worth in points on these benchmarks; (iii) whether
"LoCoMo Refined" or any re-annotation resolves gold-quality disputes — we have observed gold noise directly
(gold `The sunday before 25 May 2023` where the evidence turn says "last Saturday"); (iv) whether any top
system publishes per-question outputs we could re-grade with *our* judge, which would let us compare on one
grader instead of two.
Falsifiable here: re-grade a published system's released outputs with our judge and report both columns. If
our judge scores them *below* their reported number, the harshness argument is quantified and every caveat
row gains a stated correction.

### RQ2 — Multi-session / multi-hop assembly: what actually makes an answer assemblable?
**The largest pool: +99 answers at the gate profile (49 LoCoMo multi-hop + 50 LongMemEval multi-session),
201 of headroom.**
Known: retrieval-limited (RQ2 is the main content of M16's 52–57% *insufficient + wrong* cell). Widening `k`
does not fix it (+3.8, CI spans zero). Two agentic steps do not fix it (exactly 0.0 on temporal; +8.3 on
LME-V2 web but −0.5 on enterprise). The graph route did not fix it.
Needed: what do systems reporting 75–89% multi-session do *mechanically*? Specifically — is the win in
(a) write-time aggregation that puts the multi-hop answer in a single record, (b) query decomposition into
sub-questions with separate retrievals, (c) iterative tool-calling over a structured index, or
(d) session-level summaries retrieved alongside turns? Chronos's "58.9% gain from the events calendar alone"
and APEX-MEM's multi-tool retrieval agent are the two strongest leads. Isolate **which component carries the
gain**, and whether any ablation separates mechanism from scaffolding.
Falsifiable here: any candidate must be runnable as a `--categories 1` (LoCoMo multi-hop, n=282) or
`--categories 4` (LongMemEval multi-session, n=133) stratum arm and beat a +5.0-point pre-registered rule
with a CI excluding zero. Read-path candidates cost 5 minutes; write-path candidates cost one 57-minute
re-ingest.

### RQ3 — Open-domain: why is 29.17 vs 70.83 the worst stratum gap we have?
Known: n=96, and it is the one cell M19 moved *backwards* (−1.87, CI spans zero). Nothing has ever targeted
it. LoCoMo's open-domain questions require world knowledge combined with conversation content, which may be
a reader-knowledge limit rather than a memory limit — that would make it the one stratum a 9B backbone
genuinely cannot reach, and worth knowing before spending effort.
Needed: how do papers characterise LoCoMo category 3, does the gap track backbone size across published
rows, and does any system report open-domain gains from a *memory* mechanism rather than a bigger model?
Falsifiable here: an oracle probe — hand the reader the gold evidence for all 96 and measure accuracy. If it
stays low, the stratum is a backbone ceiling and should be written off explicitly rather than researched.

### RQ4 — Write-time event structure: how much of M19's read-path win is left on the table?
Known: M19 resolved relative dates at *read* time and got +37.6 on LoCoMo temporal. Chronos, APEX-MEM and
Memanto all do the analogous thing at *write* time — SVO event tuples with resolved datetime ranges, a
property graph of temporally grounded events, typed memory with temporal versioning — and report 95.50 /
88.88 / 89.8 class numbers. Temporal still needs +23 (LoCoMo) and +50 (LongMemEval) at the gate profile.
Needed: the schema, concretely. What is an event tuple's field set; what is extracted per turn vs per
session; what does retrieval over an event index look like next to a turn index; how is extraction cost
bounded; and critically **does any of it survive a 9B extractor**, since ours is the same model that reads.
Falsifiable here: one `build` variant writing typed event records beside episodes, then the existing stratum
arms. Budget: one 57-minute re-ingest plus 5-minute arms. Must not violate I1 (append-only) or C12.

### RQ5 — Preference and persona over time (the companion goal, and +16 answers)
Known: ss-preference is 26.67 against 80.00 with 13 of 30 declined and a token F1 of 4.6 that M9 flagged as
"not scorable this way". It did not move in M19 (+3.3 = one question) because a preference is not a dated
fact — it accumulates. PERMA (`10.48550/arXiv.2603.23231`) is converted and indexed locally and is the
benchmark built for exactly this; MemMachine and TiMem implement profile layers.
Needed: how is a profile/persona record represented and *maintained* (when is it rewritten, how is
contradiction handled, is it always composed or retrieved); what does PERMA measure that LongMemEval's 30
questions do not; and is there evidence a profile layer helps the *other* strata (a persona is context for
every question) or only its own.
Falsifiable here: `--categories 3` on LongMemEval_S, n=30 — small enough that a +5.0 rule is weak, so this
one probably needs PERMA as its real instrument. Treat that as part of the deliverable.

### RQ6 — Knowledge-update: 74.36 vs 82.05, and the cheapest 6 answers on the board
Known: it gained +7.7 for free in M19 simply because record dates became correct, and M13 found oldest-first
evidence order is worth +8.0 there (CI [−1.7, +17.6]) while being a coin flip everywhere else. `ComposeConfig`
is per-call, so a query classifier could route this stratum to the ordering that suits it — untested.
Needed: how do systems with explicit versioning (Memanto's "temporal versioning", APEX-MEM's append-only
evolution, Zep's bi-temporal graph) decide *which* version to surface, and do they surface one or all?
Falsifiable here: a read-path arm, minutes. Cheapest item with a real prior behind it.

### RQ7 — The G1 latency/accuracy frontier (LAFS gain = 0.00)
Known: our points are 39.91 @ 12.76 s and 35.70 @ 1.97 s; the reference frontier's fastest is 51.0 @ 0.2 s,
`reference_lafs = 55.765`, our gain is exactly zero because both points are dominated. AgentRunbook-R
reaches 58.60 with our reader. M16: retrieval-limited, and a perfect reader over today's evidence tops out
at 38.8–44.6%.
Needed: how is the accuracy-at-low-latency point achieved in the leaderboard's top submissions — is it
caching, precomputed runbooks/notes (the "RAG slice+notes" baseline at 51.0 @ 0.2 s is *itself* the thing to
beat), or smaller retrieval? Note this is a different shape of problem from G2: **a fast 51.0 beats a slow
70.0** for this gate.
Falsifiable here: any candidate must produce a point that is not dominated, measured through the
leaderboard's own `compute_lafs.py` (already wired via `adapters/lafs_point.py`).

### RQ8 — Closing 12.5% → ≤10% ASR without a false-positive cost
Known: undefended 77.5% (MINJA reports 76.80 — we are exactly as poisonable as the literature's victims),
defended 12.50% [Wilson 5.5–26.1] (5/40 at k=6, `runs/attack_live_m18`; M15's run of the same condition
gave 6/40 = 15.0% [7.1–29.1], so the gate misses by one to two attacks depending on the run),
0/550 false positives on real LoCoMo episodes, and **every
attack is one of two surface forms** (forged audit provenance, negating redirect) whose only defect is being
false. The ungated adaptive probe — poison with no mechanic at all, only falsehood — is admitted 10/10, which
M15 argues is the ceiling of any content classifier.
Needed: is there a published defence that is *not* a content classifier — provenance/trust propagation,
retrieval-time consistency checking against existing memory, quorum across records — with a measured ASR and
a measured false-positive rate? The EHR-poisoning paper's 6.67% at k=3 is the only sub-10% figure we have and
it is at a different k.
Falsifiable here: `attack --live` with the 8-cohort partition, ASR with Wilson intervals plus the 550-episode
false-positive pass. One GPU window.

---

## 6. What a useful answer looks like

For each mechanism the reviewer proposes, we need enough to run an arm without reading the paper again:

1. **The mechanism, separated from its scaffolding.** "A property graph" is not actionable; "extract SVO
   tuples per turn with resolved datetime ranges, index them separately from turns, retrieve with these
   three tools" is.
2. **Which stratum it claims, with the paper's own per-category cells** — not just the headline average. Our
   gap is per-stratum and an average tells us nothing about where to aim.
3. **Population, backbone and judge, quoted.** Every registry row carries a verbatim quote for a reason: M18
   found four defects in our own hand-maintained table, and M19 found a paper whose headline reproduces
   exactly as a micro-average over the official category counts (which is how we confirm `n`) and another
   whose judge is named only in an appendix. State when a field is *not* in the paper.
4. **Ablation evidence that the mechanism carries the gain.** Chronos's "events calendar = 58.9% of the
   gain, everything else 15.5–22.3%" is the standard to hold others to.
5. **Whether it was demonstrated at ≤9B.** Decisive here, per §3.
6. **Its write-path cost**, since that sets whether we can afford to test it at all.

Rank the candidates by **(answers gained at the gate profile) ÷ (cost of one measurable arm)**. §2's table
gives the numerator for any stratum claim.

## 7. Where the artifacts are

- `docs/sota/registry.json` — 30 published claims, verbatim quotes, per-row `n` / judge class / backbone
  class / provenance, and the caveats that make each comparison admissible or not.
- `runs/standing/standing.md` — the joined table, regenerated by `myelin-eval standing`.
- `docs/measurements/m9…m19-*.md` — every measurement above, with its pre-registered rule and its CIs.
- Locally converted and indexed full texts for the four systems this brief leans on: Chronos
  (`10.48550_arxiv.2603.16862`), Memanto (`…2604.22085`), APEX-MEM (`…2604.14362`), PERMA (`…2603.23231`).
  APEX-MEM's conversion covers front matter only — its results section still needs recovering.
