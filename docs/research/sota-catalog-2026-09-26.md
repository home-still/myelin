# Research catalog, round 4 *(2026-09-26)*

This adds to [`sota-catalog-2026-09-24.md`](sota-catalog-2026-09-24.md) and
[`sota-catalog-2026-09-25.md`](sota-catalog-2026-09-25.md). It serves the
round-4 plan (BACKLOG "SOTA push"). Sources come from the home-still corpus
(`distill_search`, `paper_search`); numbers are as the papers report them, and
**derived** marks our own arithmetic.

## Where LongMemEval_S is actually lost

Before the literature, the measurement that chose it. The shipped run
(`runs/m57_bonsai_premise_s1`) graded by LongMemEval's own grader (M70),
stratum by stratum against MemPro-15 on Qwen3-30B-A3B (arXiv 2606.00619,
Table 1; question counts **derived** from its percentages):

| stratum | MemPro-15 right | myelin right | Δ questions |
|---|---|---|---|
| single-session-preference (30) | 24 | 13 | **−11** |
| single-session-user (70) | 65 | ~60 | −5 |
| single-session-assistant (56) | 55 | 50 | −5 |
| knowledge-update (78) | 64 | ~67 | +3 |
| multi-session + temporal (266) | 196 | ~205 | +9 |

myelin already leads where memory is hard: across sessions, over time, and
through updates. It loses where the evidence is already in hand. Of the 107
losses, 48 hold every gold turn. Of those 48:
- 10 are declines whose own text states the answer;
- 9 are official-grader false negatives that no memory change can recover;
- 5 are generic answers to preference questions.

The preference stratum alone is the whole 2.2-point gap. So the papers below
are grouped by what they say about preference first.

## 1. Preferences

- **PrefEval.** Zhao et al. 2025, arXiv 2502.09597
  (`10.48550/arxiv.2502.09597`).
  - Zero-shot, LLMs follow a stated preference **under 10%** of the time
    after only 10 turns (~3k tokens).
  - Retrieving the preference statement (RAG) and a "reminder" are the two
    best methods on every model. Chain-of-thought and self-critique are
    worse.
  - Its error taxonomy names our two failures:
    - *preference-unaware* violations, which are generic answers;
    - *unhelpful* responses, "refusing to answer queries due to a perceived
      lack of context".
  - Lost-in-the-middle applies to preferences too.
  - **For myelin:** put the preference *statement* in front of the reader,
    as its own item. Our 5 generic answers had it buried in a 512-token
    episode.
- **PPRO.** Jiang et al. 2026, arXiv 2607.00017.
  - Episodic memories are summarized into semantic memories, and those into
    a user profile.
  - The profile is a ranking prior *and* sits in the answer context.
  - The ablation says profile-guided ranking is load-bearing on LoCoMo and
    LongMemEval-S.
  - **For myelin:** M20 built the profile records (124 per tenant) and
    chose them by recency, reaching gold-content recall 0.042. Rank them by
    the question instead (M20b).
- **CueMem.** Wang et al. 2026, arXiv 2609.12354.
  - Extracted records are treated as *cues*, not evidence. Each points at
    its source turn, and the context is rebuilt over a turn graph.
  - It is best on LongMemEval's single-preference and single-user strata.
    Removing the graph costs 81.1 → 71.4 on LoCoMo.
  - **For myelin:** in 7 of our 8 preference losses with no gold, the right
    *session* is in evidence, but as the assistant's long replies. A
    disposition record ("the user finds turbinado adds a richer flavor")
    is the cue that reaches the short user turn.
- **REALM.** Song et al. 2026, arXiv 2609.16053. Retrieval-driven
  reconsolidation raises LongMemEval single-session-preference by +6.66 and
  knowledge-update by +4.17.
- **Memora** (Uddin et al. 2026, `10.18653/v1/2026.findings-acl.1337`) and
  **AlpsBench** (Xiao et al. 2026, `10.1145/3805712.3808634`).
  - Preferences evolve and reverse.
  - Injected profiles can bias answers ("personalization bias").
  - **For myelin:** gate the profile block to requests for advice, not every
    question.

## 2. Counting and aggregation

- **Utility Under Attack.** Karunanidhi 2026, arXiv 2608.21230.
  - On LongMemEval_S multi-session, accuracy *falls* with depth:
    0.742 / 0.677 / 0.613 at k = 15 / 30 / 50.
  - The reader over-counts from plausible extra context. "This is a
    reader-side aggregation failure."
  - **For myelin:** 4 of our 5 count errors with every gold turn held are
    over-counts. M72's depth needs an over-count rate reported beside it.
- **JustMem.** Chen et al. 2026, arXiv 2609.19877.
  - Up to two planner rewrites widen discovery, then everything is
    reranked back to a fixed top 10.
  - Aggregation questions rise 65.96 → 78.72 on LongMemEval-S, and
    Recall@10 91.8 → 96.6.
  - Global atomic cards beat session packs (69.09 → 79.81 on LoCoMo).
  - Its answer rule: "operate only over distinct evidence".
- **APEX-MEM.** Banerjee et al. 2026, `10.18653/v1/2026.acl-long.749`.
  Entity documents carry a `latest` property table, plus read-only SQL for
  aggregation.
- **EdgeMem.** Cui et al. 2026, arXiv 2609.05553. An LLM-free multi-anchor
  hypergraph (time, co-occurrence, episode): no model calls to build it.

## 3. Knowledge updates

- **JustMem:** "for latest or updated information, resolve conflicting
  evidence using temporal order."
- **CueMem:** its knowledge-update lead comes with no UPDATE operation at
  all. Temporally ordered source turns let the reader drop stale values.
- **Knowledge Conflicts for LLMs** (survey). Xu et al. 2024,
  `10.18653/v1/2024.emnlp-main.486`. Under temporal misalignment the newer
  context is correct.
- **For myelin:** 3 of our wrong values are stale, and the ledger has no
  `supersedes` links (M49). Ordering is the cheap lever.

## 4. Time

- **Test of Time.** Fatemi et al. 2024, arXiv 2406.09170.
  - Most duration errors are off by exactly one day.
  - Fact order matters: sorting by entity, then time, is best.
- The 09-24 set still holds:
  - LongMemEval's time-aware query expansion (Wu et al. 2024, arXiv
    2410.10813 §5.4) needs a strong model to extract the range. A small
    one hallucinates it, which is why M73 uses the closed grammar.
  - Chronos's event calendar (arXiv 2603.16862), and TReMu
    (`10.18653/v1/2025.findings-acl.972`).

## 5. The landscape since 09-24

New LongMemEval work in the corpus:
- JustMem;
- CueMem;
- EGMemory, "Propose, Verify, Commit" (arXiv 2609.23465; LoCoMo 73.6);
- DolphinBench (2609.24971);
- ThinkFlow (2609.17010);
- Jev-Mem (2609.23986);
- MemCalib (2609.24259);
- LSREP (2609.16730).

None of these reports a same-size open-model LongMemEval_S row above 80.80
in the text read. `docs/sota/registry.json` is unchanged.

## What we take from it

Ranked by questions within reach, for a Bonsai 27B reader under the
code-first rule:

| # | build | stratum | research |
|---|---|---|---|
| 1 | **M20b**: a relevance-ranked `[profile]` side block, gated to advice/recommendation requests | preference (17 lost) | PrefEval, PPRO, CueMem, Memora |
| 2 | **M73b**: events from the question's own past-day window | temporal, question-dated (8 targets) | LongMemEval §5.4, Chronos, M73 |
| 3 | **M71b**: re-ask a decline only when it names a value found in the evidence | gold-in-hand declines (20) | Sufficient Context, M61 |
| 4 | **M72b**: aggregation depth with the pool cap fixed, over-counts reported | multi-session counting | JustMem, Utility Under Attack |
| 5 | **M74**: the selector reads the best-matching passage, not the first 400 characters | part/none gold (57) | MemPro's focused snippets, RECOMP |

Each is validated on its own stratum. Every switch that passes goes into one
bundle arm held to +3.0 (user, 2026-09-26).
