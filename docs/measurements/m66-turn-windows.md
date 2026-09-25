# M66 — turn windows: evidence at the granularity of turns *(pre-registered 2026-09-25, before any row)*

## Why

The 2026-09-25 loss anatomy on `runs/m63_locomo_base` (strict 9B judge,
70.52) found the largest recoverable stratum in **multi-hop**:
- It loses 114 rows: 63 hold only part of their gold turns (missing ~2 of
  3.6), 43 hold none, and 8 hold all of them.
- The correct-rate is 87% with every gold turn held, 59% with part, and 36%
  with none.
- Under MemPro's own judge (M68) we still trail it there by 4.6 points.

The cause is **granularity**:
- An episode is a 512-token segment, about 13 turns (`ingest.rs`). At
  k = 6 a question reaches ~2.3 episodes plus ~3.7 facts, most of each
  episode being text around the turn that matters.
- The reranked pool holds the gold (0.959) far more often than the emitted
  set does (0.863, M65).
- The cross-encoder scores a whole episode through a 512-token input, so the
  tail of a long episode is not even seen.
- ~60 of single-hop's 102 held-but-wrong answers pick a distractor inside
  the chunk.

## The mechanism

`RetrieveConfig::turn_windows = Some(radius)`
(`crates/myelin-core/src/pipeline/turn_windows.rs`):
1. **Score every turn.** Every turn of every episode in the reranked pool
   is scored against the question by the same cross-encoder. The turns come
   from the speaker labels the episode record already stores.
2. **Walk the units best-first.** Turns and non-episodic records (facts,
   scored as before) are taken best-first across the whole pool. A new
   record takes a slot, and an episode's window opens at that turn.
3. **Count windows toward `k`.** A later turn of a taken episode opens a
   second window only while fewer than `k` units (records plus extra
   windows) are taken.
4. **Emit the window.** An episode is emitted as its windows (±radius turns
   each), with `…` where turns were cut, and is charged for that text only.
   Dates resolve on the emitted text, so an elided phrase is never
   annotated. The record is never rewritten.

`k` rises so that the tokens per question stay flat. Radius **2** is
QueryLink's c, fixed now and not swept.

**Grounds:**
- QueryLink (Hu 2026, `10.18653/v1/2026.findings-acl.765`): ±c neighbouring
  turns; c = 0 costs 11.36.
- JustMem (Chen 2026, arXiv 2609.19877): globally ranked fine units beat
  whole-session packs by 10.72.
- MemPro's focused snippets (arXiv 2606.00619, +1.47).
- RECOMP (arXiv 2310.04408).
- Du 2025 (`10.18653/v1/2025.findings-emnlp.1264`): length hurts.

## Stage 1 — retrieval, no reader (the rule fixed now)

`ablate --width --grid shipped --k K [--turn-windows 2]` on the LoCoMo dev
split, the same one M65 used:
- **Cells:** whole episodes at K ∈ {6, 10}; turn windows at
  K ∈ {6, 10, 14, 18}.
- **Metrics reported per cell:**
  - `all` (every gold turn held, through lineage);
  - the new **`turn_all`**: every gold turn's own line verbatim in the
    emitted evidence. It is an independent text check, because lineage
    would credit a trimmed episode whose window cut the gold out;
  - `evidence_tokens`;
  - latency.

**K\*** is the largest windowed K whose mean `evidence_tokens` is at or
below the whole-episode k = 6 cell's. That makes the arm length-neutral by
construction.

**Go/no-go:** the reader arm runs only if windowed `turn_all` at K\* beats
whole k = 6 by **≥ 0.03**. Otherwise M66 is recorded as failing at
retrieval.

## Stage 2 — the reader arm (pre-registered now; K\* is filled in from stage 1 before its first row)

- **Run:**
  `bench --corpus locomo --mode recall --k K* --max-steps 2 --turn-windows 2`,
  full 1,986, the shipped 9B reader.
- **Pairing:** paired against `runs/m63_locomo_base`, judged by the 9B
  seeded from it.
- **Bar:** +3.0 on the strict judge, categories 1–4, with the 95% CI
  excluding zero.
- **Veto:** adversarial (category 5) at or above the base's 67.94. M50c's
  lesson is that more sessions in view can tempt the reader on traps.
- **Also reported:** the M68 matched judge (LightMem protocol) for the arm.

**Predictions:**
- `turn_all` at K\* rises by ≥ 5 points over whole k = 6.
- Multi-hop +4 to +8.
- Single-hop 0 to +2 (less text around the answer).
- Temporal −2 to +2.
- Adversarial −2 to 0.
- **Overall +2 to +4**, at or near the bar.

**Falsifiers:**
- `turn_all` rises and multi-hop does not: the reader does not use the added
  turns.
- Single-hop falls by more than 2: ±2 turns cut context the answer needed
  (QueryLink's c = 0 failure at c = 2).

## Stage 1 result *(2026-09-25, 17:26–18:21; reranker only, on big under a lease)*

LoCoMo dev split, 997 questions, `ablate --width --grid shipped`:

| cell | all (lineage) | **turn_all** (verbatim) | evidence tokens | p50 |
|---|---|---|---|---|
| whole episodes, k = 6 (shipped) | 0.8626 | **0.7924** | 1,304 | 172 ms |
| whole episodes, k = 10 | 0.8937 | 0.8345 | 2,228 | 180 ms |
| windows ±2, k = 6 | 0.8395 | 0.6991 | 447 | 919 ms |
| windows ±2, k = 10 | 0.8776 | 0.7693 | 787 | 558 ms |
| windows ±2, k = 14 | 0.8987 | **0.8154** | 1,186 | 525 ms |
| windows ±2, k = 18 | 0.9147 | 0.8295 | 1,637 | 508 ms |

**The pre-registered rule:**
- K\* is the largest windowed K at or under 1,304 tokens, which is **k = 14**.
- Go needs turn_all ≥ 0.7924 + 0.03 = 0.8224. At K\* it is **0.8154**,
  short by 0.007.
- **No-go: M66 fails at retrieval.** The reader arm is not run.

**What the numbers do show:**
- **Windows spend tokens far more efficiently.** At the same ~1,200 tokens,
  windows hold every gold turn verbatim on 81.5% of questions. Whole
  episodes get there only at k = 10, which costs 2,228 tokens.
- **The only cell past the bar costs more tokens.** Windows at k = 18 reach
  0.830 at 1,637 tokens (+26%), which is not length-neutral.
- **`all` overstates windows.** The lineage metric rises to 0.915 at k = 18
  while the verbatim one stays at 0.83. Lineage credits a windowed episode
  whose window cut the gold turn out, which is why stage 1 was decided on
  turn_all.
- **Latency:** scoring every pooled turn costs ~0.5 s per query at the
  median, against 0.18 s.

**Instrument fix during the sweep** (PR #117): LoCoMo dia_ids restart in
every conversation. The first cell's turn_all read 0.2487 until the gold
text was keyed by (tenant, dia_id). No result from the bad metric was used.

**What next, if anyone revisits:**
- Radius 1, at the same k, would buy more windows per token.
- A second rule that windows only the top-N episodes would cut the
  latency.

Neither is scheduled. Code-first effort moves to LME-V2 (M69), where the
gap is far larger.
