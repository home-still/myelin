# M89 — typed memory for LoCoMo, in stages *(round 8, user decision 2026-09-30: "typed memory, reading fixes too"; stage A pre-registered before any row)*

## Why

LoCoMo's bar is LeanMem (Qwen3-8B, arXiv 2608.03463), which scores 84.41 under
SimpleMem's judge. Our deterministic shipped run reads 79.03 under that judge:
**−5.38**, about 83 questions of 1,540 (`det-remeasure-2026-09-30.md`).

The loss anatomy (`locomo-gap-leanmem.md`) splits the 329 losses:

| every gold turn in evidence? | losses | of which wrong / declined |
|---|---|---|
| yes | 156 | 119 / 37 |
| no | 115 | |
| partly | 58 | |

The typical miss is a nearby wrong detail taken from a ~13-turn,
~2.1k-character episode.

LeanMem's own ablation (GPT-4.1-mini, LoCoMo 84.87):

| removed | score | cost |
|---|---|---|
| typed storage schedule | 72.79 | −12.08 |
| retrieval plan | 77.92 | −6.95 |
| topic segmentation | 82.79 | −2.08 |
| memory evolution | 84.42 | −0.45 |

## Stages

Each stage is measured reader-free and gates the next. The first reader arm
is stage C. The design is from the round-8 design pass (2026-09-30).

| stage | what | instrument | go |
|---|---|---|---|
| **A** | topic episodes: an episode also ends where a session changes topic | `ablate --width` against the shipped store's episodes | `turn_all` at length-neutral K\* ≥ control + 0.03 |
| B | a write-time scheduler routes each segment to profile / event / record (gist + span pointer) | `typed-build` report: `reach_all`, ignore rate, gist grounding | `reach_all` ≥ 0.97 and the gist lane's `all`@6 ≥ stage A's |
| C | a deterministic typed read path: records lane, events lane, profile lane; records expand to their cited turns | `ablate --width --typed` with `turn_precision` | `turn_all` ≥ control + 0.03 at ≤ its tokens, or ≥ control − 0.01 at 1.3× its precision; **then the reader arm** |
| D | an LLM planner, only if the deterministic plan's oracle regret ≥ 0.03 | the plan lattice's regret | |

The reader arm (stage C) is paired against `det_locomo_m84` under SimpleMem's
judge. It ships at +3.0 with the CI excluding 0 and the adversarial veto
(strict judge, 69.51).

## Stage A — topic episodes *(pre-registered 2026-09-30, before any row)*

**Mechanism.** `pipeline::topic`, `WritePath::topics`, and `build --topics`,
which writes a new store and refuses the shipped one. Inside each session:
1. **Key utterances.** A lexical rule leaves out:
   - turns under 4 words;
   - short acknowledgements;
   - restatements of the last key utterance (≥ 80% of its content words).

   These turns are dropped from the *similarity sequence only*; every turn
   stays in its episode.
2. **Cosine similarity** of adjacent key utterances (turn text only, no
   speaker name).
3. **TextTiling depth** with a window of 2 (Hearst 1997, ACL J97-1003).
4. **Boundaries** are strict local maxima of depth at or above the session's
   `mean − σ/2` (LeanMem §3.1). Segments hold at least 2 key utterances.
5. The session, time-gap and 512-token rules are unchanged. The cap is the
   cross-encoder's input limit (M66).
6. No facts are extracted: the stage compares episodes with episodes.

Grounds:
- SeCom (arXiv 2502.05589): segment-level memory beats turn- and
  session-level on LOCOMO.
- LeanMem §3.1.
- LightMem (arXiv 2510.18866) segments by topic too.

**Instrument.** `ablate --width --grid shipped` on the LoCoMo dev split (997
questions, M65's and M66's).
- Coverage is now read from the store: an episode's turns are the
  consecutive turns from its source `dia_id` that reproduce its text.
- On the shipped store that map equals the old segmenter replay for all 550
  episodes (`store_coverage_reproduces_the_segmenter_replay_on_the_shipped_store`,
  run on 2026-09-30).
- The walk found two records the 2026-09-21 usability probe wrote through
  `remember`. One of them, "The user prefers dark mode…", sits in conv-26's
  LoCoMo tenant. It appears in no LoCoMo run's evidence
  (`det_locomo_m84`, `m84_locomo_base`, `m63_locomo_base`: 0 rows). It covers
  no gold turn and is reported by the instrument.

**Cells:**
- **Control:** the shipped store, episodes only (`--episodes-only`; later
  builds spell it `--kinds episodic`), at
  k ∈ {6, 10}.
- **Topic store** (`myelin_locomo_topics`, `data/locomo_topics.ledger`) at
  k ∈ {6, 8, 10, 12, 14}.
- **For reference only, not the rule:** the shipped store with its facts at
  k = 6, the configuration the reader sees today.

**The rule:**
- K\* is the largest topic-store k whose mean `evidence_tokens` is at or below
  the control's k = 6.
- **Go** if the topic store's `turn_all` at K\* is ≥ the control's k = 6
  `turn_all` + 0.03.
- **No-go:** stage B runs on today's 512-token episodes instead. That is a
  choice recorded here, not a runtime fallback.

**Reported beside the rule:**
- boundary counts by reason (`topic_shifts` per conversation);
- mean episode length;
- `all` (lineage);
- latency.

**Cost:** the build embeds about 5,900 turns plus about 1,000 episodes, with
no LLM calls. The sweep is 7 cells × 997 questions, reranker only. Together
well under an hour on big.

## Stage B — typed routing *(pre-registered 2026-09-30, before any row)*

**Code:**
- `pipeline::typed`: scheduler, code-side `route`, event and gist calls, and
  `typed_records` (#201).
- `typed-extract` and `typed-build` (`crates/myelin-eval/src/typed.rs`).
- Prompt version `m89b-1`. The cache is keyed by version, speakers and text.

**Store:** a copy of the stage A store, or of the shipped store's episodes if
A is a no-go, with the typed records written beside the episodes. The copy is
made by `ops/big/qdrant-copy.sh` and a ledger file copy.

**Stage B's gate:**
- **`reach_all` ≥ 0.97.** That is the share of answerable questions whose
  every gold turn lies in some stored (non-ignored) unit's span, from the
  `typed-build` report.
- **Gist lane holds its own.** On the dev split, `ablate --width --grid
  shipped --kinds gist` must reach `all`@6 at least stage A's episodes-only
  `all`@6.
  - Coverage for a typed record is **only the turns it cites**: a derived
    record with a span of one episode covers that span, never its whole
    episode.
  - No existing record carries a span (0 of 4,488 derived records in the
    shipped store), so nothing measured before moves.

**Reported beside the gate:**
- the ignore rate;
- units by route;
- gist grounding: the share of a gist's names and numbers found in its cited
  turns;
- the share of event `when`s copied verbatim;
- the share of dataset turns inside a stored span.

**Cost:** about 1,000–1,100 episodes, each one scheduler call plus at most
one event call and one gist call, on the 9B at 2 slots. That is about
2–3.5 h on big.

## Result

*(not yet run)*
