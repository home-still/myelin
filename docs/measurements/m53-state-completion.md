# M53 — state completion: deliver the whole page state, not its best chunk *(diagnosed 2026-09-23; plan)*

## Where LME-V2 loses its answers

No gold evidence ships with LME-V2, so the diagnosis uses a deterministic
proxy: for every answerable question whose gold answer is a phrase or a list
(218 of 323; booleans, bare numbers and fragments under four characters are
excluded because their presence in 10k tokens of context says nothing), is
each answer item present, normalised, in the memory context the reader was
handed? Then, for the answers that were not: is the item present anywhere in
that domain's 100-trajectory haystack (accessibility trees, actions,
thoughts, goals, URLs)? Base runs `runs/m47_base_{web,ent}`, today's
defaults on the rebuilt store (combined 38.80).

| | web | enterprise | both |
| --- | --- | --- | --- |
| eligible answerable rows | 125 | 93 | 218 |
| answer fully delivered — reader correct | 50 of 68 (73.5%) | 31 of 56 (55.4%) | |
| **wrong, answer delivered** (reader) | 18 | 25 | **43 (36%)** |
| **wrong, answer in the haystack, not delivered** (retrieval) | 32 | 22 | **54 (45%)** |
| wrong, answer not literally anywhere (inference) | 12 | 10 | 22 (18%) |

**Retrieval is the larger loss on LME-V2**, not the reader — the opposite of
LongMemEval_S, where M44 R2 showed the reader compute-limited. It also
explains M52: thinking cannot answer from evidence it was never given.

## Where the missed answers sit

Delivered items were mapped back to ledger episodes by exact text (source
docs are `trajectory:state:chunk`), and each retrievable miss classified by
where its answer lives relative to what was delivered:

| answer location | web (wrong) | enterprise (wrong) |
| --- | --- | --- |
| **another chunk of a delivered page state** | 12 (9) | 10 (10) |
| **the adjacent state (±1) of a delivered one** | 5 (5) | 4 (4) |
| 2–3 states from a delivered one | 5 (5) | 3 (3) |
| 4+ states away | 4 (4) | 1 (0) |
| trajectory delivered only as notes / events / goal | 6 (2) | 3 (3) |
| trajectory never delivered | 9 (7) | 3 (2) |

Retrieval usually finds the right trajectory and often the right page, and
hands over the wrong 512-token slice of it: the top of the Incidents list
without the chunk that lists the filter options, or the state before the
dropdown opened rather than the one after. **28 wrong answers (14 per
domain) sit in a sibling chunk or the next state of something already
delivered.**

## The mechanism

A compose-time view, not a re-ingest (the M49 idea — JustMem's REPLAY,
`2609.19877`, "recover the original for fidelity-sensitive evidence"): when
a state chunk is composed, add the rest of that state (its sibling chunks,
by the `trajectory:state:` source prefix, which `Ledger::ids_from_source_docs`
already resolves) and the state that follows it, ranked by the reranker
against the question and admitted while the budget lasts. Chronos does the
same for dialogue: every retrieved turn is expanded with its neighbours
(`10.48550/arXiv.2603.16862` §3.3). The LME-V2 authors' coding-agent
controller wins for the same reason: it opens the whole file
(`10.48550/arXiv.2605.12493`, AgentRunbook-C 72.5 against the best RAG
system's 48.5).

The budget is the trade: ~10k tokens already hold 25 items, and a full
accessibility tree runs to several thousand. Completion must displace the
lowest-ranked items, not append.

## Pre-registration (to be completed before the arm runs)

Base and arm built **the same way** (same serve, same concurrency), the M47
lesson. Primary: combined, paired bootstrap, +3.0 with the CI excluding zero;
abstention veto. Predicted: the 28 targeted rows convert at the reader's
delivered rate (55–74%) → **+3 to +5 combined**; answerable up, abstention
flat.

Next after it, for the 12 missed answers in trajectories never delivered:
an exact-text probe the controller can issue (`grep`), then — the ambitious
version, local-only — a coding-agent controller over the trajectories as
files, on big's 27B.
