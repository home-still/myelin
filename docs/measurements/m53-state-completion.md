# M53 — where LME-V2 loses its answers, and three cheap fixes that do not reach them *(diagnosed and probed 2026-09-23; no arm)*

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

## Three cheap fixes, probed before any arm (2026-09-23)

**1. State completion, reranked** — after selection, for the top 6 records
that are page-state chunks, fetch the rest of the state and the next state
by source prefix, rerank them against the question with the cross-encoder,
admit the best 2 per seed right after it (Chronos's neighbour expansion,
`10.48550/arXiv.2603.16862` §3.3; JustMem's REPLAY, `2609.19877`). Built and
probed live on six diagnosed web misses, same settings as the base:

| question | answer items delivered, off | on | records admitted |
| --- | --- | --- | --- |
| 06a5a25f | 0/1 | 0/1 | 4 |
| 07b49858 | 2/4 | 2/4 | 6 |
| 0c6b0c60 | 2/5 | 2/5 | 4 |
| 11dac74b | 0/1 | 0/1 | 8 |
| 17d63ad8 | 0/2 | **2/2** | 6 |
| 31f146ba | 0/1 | 0/1 | 6 |

**One in six.** The answer's state usually runs to 9+ chunks and the
answer-bearing chunk (a dropdown's options, a banner) does not resemble the
question, so a question-conditioned pick of two rarely lands on it. Nine of
the thirteen diagnosed misses share a state with a top-6 item, so the seed
window is not the problem; the pick is. Not run as an arm; the code was
removed rather than merged dormant.

**2. A change view over delivered states** (offline) — for each top-6
delivered state *s*, the lines of *s* and *s+1* absent from the state before
(element ids stripped). Every inspected answer is new relative to the
previous state, and the view would contain the missing answer on **19 of 65**
retrievable misses (17 answered wrong) — worth roughly +2 to +3 combined — but
it is the whole page whenever the action navigates (p90 14–16k characters).

**3. A step-anchored change view** (offline) — rank all ~2,800 steps of the
domain's 100 trajectories by IDF-weighted overlap between the question and
the step's thought + action, and show the change after the top 3 (or 6)
steps. The agent's thoughts do name UI elements ("open the Actions menu").
It recovers **3–4** web misses and **none** on enterprise: the question's
words are too common across 2,800 steps to single out the step.

## What this means

LME-V2's gap is retrieval, and retrieval here is not a ranking tweak away:
the answer is a specific UI string in a specific transition, and finding it
takes a controller that can search for a label, open a page and compare it
with the one before. That is the LME-V2 paper's own conclusion — the
coding-agent controller over trajectory *files* reaches 72.5 against the
best RAG system's 48.5 (`10.48550/arXiv.2605.12493`, AgentRunbook-C). The
local version is M54 in BACKLOG.md.
