# SOTA catalog, 2026-09-30 — round 7: the loss blocks, and a new LoCoMo bar

**Why:** the user re-ran `/plan` and asked for four things:
- review the bottlenecks;
- catalog research in home-still;
- put resource constraints to the user;
- implement against the primary blocker.

Both headline margins were inside rerun noise: LongMemEval_S +0.73 and
LoCoMo +0.79. So this round looks for (a) the mechanisms behind the loss
blocks that remain, and (b) any same-class row we had missed.

**How:**
- A research agent searched home-still (`distill_search`, `paper_search`,
  read-only, through `hs_call.py` when the MCP connector dropped) and fetched
  arXiv HTML where a paper was not converted.
- The main session verified the two new LoCoMo rows (LeanMem and MemChain)
  in arXiv HTML and in home-still. LeanMem's PDF was downloaded and converted
  (stem `10.48550_arxiv.2608.03463`); MemChain's was downloaded
  (`10.48550_arxiv.2607.24097`).

**Confidence tags:**
- **V-hs:** seen in full text in home-still.
- **V-arx:** seen in the arXiv HTML full text.
- **R-abs:** the abstract only.
- **measured:** our own run.

---

## The finding that moved a bar: LeanMem on LoCoMo

| system | backbone | LoCoMo (1,540) | judge | tag |
|---|---|---|---|---|
| **LeanMem** (arXiv 2608.03463) | **Qwen3-8B** | **84.41** (5-run mean) | GPT-4.1-mini, "the evaluation protocol of SimpleMem" | V-arx |
| LeanMem | GPT-4.1-mini | 84.87 | same | V-arx |
| LightMem, as run by LeanMem | Qwen3-8B | 78.57 | same | V-arx |
| **myelin** (`runs/m84_locomo_base`) | Bonsai 27B | **78.64** | the same judge, re-graded by us | measured |

- **The matched re-grade.** LeanMem's App. D, which prints its prompt, is
  not in the arXiv PDF, and its code is supplementary material only. Our
  reading therefore uses SimpleMem's own LoCoMo judge:
  - verbatim from `aiming-lab/SimpleMem@9b12e8d`
    `MCP/reference/test_locomo10.py` (`llm_judge_answers`), with its
    system message and config: gpt-4.1-mini, temperature 0.3, no
    response_format;
  - `adapters/judge_matched.py --protocol simplemem-locomo`, prompt sha
    `947d5690…`, checked byte-identical with `--verify-upstream`.
- **The result:** 1,211 of 1,540 = **78.64**.
  - By category: multi-hop 71.28, temporal 76.01, open-domain 42.71,
    single-hop 86.21. Cost $0.49.
  - A 20-row pilot agreed with LightMem's grader on 18.
- **The judge is lenient in specific ways.** It passes a subset of a list,
  dates within 1–2 days, and a coarser granularity. It still fails a wrong
  entity ("Excited and proud" against "Glad").
- **User decision (2026-09-30): adopt 84.41 as LoCoMo's bar.**
  - The registry now gates on `locomo.judge_simplemem.leanmem.qwen3_8b`.
  - MemPro-15 (77.85) stays as a comparison.
  - **LoCoMo is open at −5.77** (about 89 questions), pending the
    deterministic re-measurement.
- **LeanMem's mechanism, for the next LoCoMo round:**
  - an utterance filter;
  - a scheduler that routes each topic segment to **profile**, **event** or
    **record** memory;
  - selective evolution of event memory only;
  - an LLM planner that picks memory types and per-type budgets for each
    question.
  - Its ablation on GPT-4.1-mini: without the retrieval plan 77.92, without
    the storage schedule 72.79, without the utterance filter 75.58.

## The other new row: MemChain

| system | backbone | LoCoMo | judge | verdict |
|---|---|---|---|---|
| MemChain SFT+TMPO (arXiv 2607.24097) | trained Qwen3-4B policy, frozen Qwen3-14B answers | 80.26 | LoCoMo-Refined (Qwen3-14B) | **not matched** |

- LoCoMo-Refined (`mem-eval-suite/LoCoMo_refined@8870911`) is a *stricter*
  judge. It requires complete lists, no unsupported additions and exact
  temporal granularity. It also *revises 337 gold answers*, over 1,382
  questions.
- MemChain states 1,540 questions but does not say which QA set it graded.
- Matching it needs Qwen3-14B served as the judge. Recorded, not a gate.

## A. Single-session-assistant: focusing inside a passage

| paper | mechanism | reported | tag |
|---|---|---|---|
| MemLoc, 2609.07093 | extract the relevant segment inside a unit and add evidence-id cues, **keeping the full context** | extract-only: −4.5 to −9.0; Extract +0.4–1.4, Cue +0.4–3.4 | V-hs (table OCR scrambled) |
| JustMem, 2609.19877 | compact cards, and REPLAY of the source session on demand | SSA 32.14 → 94.64 (GPT-4.1-mini) | V-hs |
| MemPro, 2606.00619 | focused evidence snippets before integration | 83.46 → 84.93 | V-hs |
| LongMemEval, 2410.10813 | JSON rendering plus Chain-of-Note | up to +10; facts or summaries in place of rounds **hurt** | V-hs |
| EXIT, 2412.12559 | a context-aware sentence classifier | Llama-3.1-8B beats full docs; RECOMP-extractive **hurt** it | V-arx |
| RECOMP, 2310.04408 | a trained extractive compressor | 2.4–3.4 EM below full docs | V-arx |
| Lost in the Middle, 10.1162/tacl_a_00638 | position effects | >20-point drops mid-context | V-hs |

**Read:**
- Our SSA losses (5 questions) are reader misreads with the whole gold turn
  in hand, so any fix here is a cue *beside* the turn, never a replacement.
- Backlog candidate only. The largest recoverable block is elsewhere (M86).

## B. Open-domain inference over personal memory

| paper | mechanism | reported | tag |
|---|---|---|---|
| LoCoMo, 2402.17753 | baseline study | open-domain "degrades in the RAG setting" with improper context | V-hs |
| REALM, 2609.16053 | retrieval-driven graph reconsolidation | open-domain +5.21 (gpt-4o-mini) | V-hs |
| TiMem, 2601.02845 | profile levels always recalled | +2.12 LoCoMo; higher levels *alone* −18.2 | V-arx |
| PPRO, 2607.00017 | profile text in the context plus a retrieval prior | F1 43.20 → 40.79 without it (Qwen2.5-7B) | V-hs |
| SGMem, 2509.21212 | sessions plus facts plus insights in a sentence graph | LME 0.676 → 0.700 | V-hs |
| Generative Agents, 2304.03442 | reflection memories | believability only | V-hs |

**Read:** profile and insight records retrieved *beside* raw turns. This is
also LeanMem's profile memory.

## C. Multi-hop over chat memory

| paper | mechanism | reported | tag |
|---|---|---|---|
| CueMem, 2609.12354 | expansion from anchor turns over temporal and semantic edges | LoCoMo 71.4 → 81.1 (Llama-3.3-70B) | V-hs |
| LongMemEval, 2410.10813 | fact-augmented keys | +9.4% recall@k, +5.4% accuracy, 8B readers included | V-hs |
| IRCoT, 2212.10509 | retrieval interleaved with CoT | recall +3.5–14.3 (Flan-T5-XXL) | V-hs |
| PREMem, 10.18653/v1/2025.findings-emnlp.1204 | links computed before storage | small readers match much larger baselines | V-hs |
| MemPro, 2606.00619 | adaptive retrieval depth | +1.34 | V-hs |

**Read:** question-gated depth for list and count questions (M87) is the
cheap step. Anchor expansion is the larger one.

## D. Preference-aware answers

| paper | mechanism | reported | tag |
|---|---|---|---|
| PrefEval, 2502.09597 | benchmark | retrieval of stated preferences is the best remedy; prompting alone hallucinates preferences | V-hs |
| TiMem, 2601.02845 | a profile layer always in context | SSP 95.71 (gpt-4o-mini) | V-arx |
| LeanMem, 2608.03463 | profile/event/record memory | −6.40 LME-S without the storage schedule | V-arx |
| AlpsBench, 10.1145/3805712.3808634 | benchmark | models over-rely on injected memories; indirect-preference recall collapses for weaker models | V-hs |

**Read:**
- Our preference losses are mostly a *segmentation split*: the reply is held
  and the user's stated preference is not. That is M86.
- A typed profile (M20/M20b) narrowed answers when it replaced the user's
  words.

## E. Other new rows (no clean same-class row above ours on LongMemEval_S)

| paper | backbone | result | note |
|---|---|---|---|
| SGMem, 2509.21212 | Qwen2.5-32B | LME 70.0 | its judge is the same model |
| SpeakerMem-R1, 2609.26780 | not stated | LoCoMo 70.85 | R-abs |
| Auditable LTM, 2609.38021 | Claude Opus | LME-S 479/500 | frontier; developed on the test set |
| MemoryLACE, 2609.03201 | Qwen3.5-4B/9B | BEAM only | V-hs |

**LongMemEval_S:**
- MemPro-15 (80.80) stays the clean same-class bar.
- LeanMem's Qwen3-8B 77.40 is below our 81.53.
- Hindsight's 83.6 stays `not-comparable` (the `answer_*` label leak,
  2026-09-29).
