# The LoCoMo gap to LeanMem — loss anatomy under the bar's judge *(2026-09-30)*

On 2026-09-30 LoCoMo's bar moved to LeanMem (Qwen3-8B, arXiv 2608.03463):
84.41 under SimpleMem's judge (`docs/research/sota-catalog-2026-09-30.md`).
Our shipped answers read **78.64** under that judge. That is **−5.77**, about
89 questions of 1,540.

This is the anatomy of that gap on `runs/m84_locomo_base` (the shipped
recipe, before the tie-order fix). It will be refreshed on the deterministic
base, `runs/det_locomo_m84`, when that lands. The method is the one the
round-2 anatomy used:
- a gold turn counts as held when its first 60 characters appear in the
  evidence the reader saw;
- a decline is an answer that opens "I don't know" or "not mentioned".

## Where the 329 losses sit

| | every gold turn held | part held | none held | total |
|---|---|---|---|---|
| **wrong answer** | **119** | 38 | 69 | 226 |
| **decline** | 37 | 20 | 46 | 103 |
| **total** | **156** | 58 | 115 | 329 |

| category (n) | ours (SimpleMem judge) | wrong | declined |
|---|---|---|---|
| multi-hop (282) | 71.28 | 62 | 19 |
| temporal (321) | 76.01 | 37 | **40** |
| open-domain (96) | 42.71 | 35 | 20 |
| single-hop (841) | 86.21 | **92** | 24 |

**Read:** the largest block is **reading**. In 156 losses, 47% of all losses,
the reader held every gold turn and still missed:
- 119 answered wrong;
- 37 declined.

Retrieval (none held) is 115. LeanMem's reader is a Qwen3-8B, no bigger than
ours (Qwen3.5-9B), so the difference is in what the reader is shown.

## What the reading failures look like

Sampled from the 119 wrong answers with every gold turn held (mean evidence
about 5,500 characters: k = 6 items, 1–3 of them 13-turn episodes):

- **A nearby wrong detail from the same chunk (single-hop, most of the
  77):**
  - "How do Audrey's dogs react to snow?" Gold: confused. Answer: "They hate
    it."
  - "What did John share with the person he skyped?" Gold: Harry Potter
    characters. Answer: "A photo of a basketball game."
- **The wrong granularity:**
  - "In what country did Jolene's mother buy her the pendant?" Gold: France.
    Answer: Paris.
  - Bogota for Colombia; Toronto for Canada.
- **An unresolved date (temporal):**
  - "When did John start boot camp?" Gold: April 2023. Answer: "Last month".
  - "What day did Tim get into his study abroad program?" Answer: "Friday".
- **A judge miss:** "He relies on it for his active lifestyle and road
  trips" was marked wrong against a gold that says the same. SimpleMem's
  judge samples at 0.3.

## What LeanMem does differently (read in home-still, stem `10.48550_arxiv.2608.03463`)

- **Write:**
  - key-utterance filtering removes greetings and acknowledgements;
  - topic segments are cut at valleys of adjacent-utterance similarity;
  - an LLM scheduler routes each segment to **profile**
    (attribute-value pairs), **event** (topic, temporal anchor, state) or
    **record** (a retrieval gist plus GLiNER entities plus a pointer to the
    source span), or ignores it.
- **Maintain:** only event memory evolves, merged by topic and ordered by
  its temporal anchors.
- **Read:** an LLM planner turns each question into constraints, a
  granularity, memory types, per-type weights and per-type k. Records expand
  to their source span only "when detailed evidence is required".
- **Cost:** about 3.0K inference tokens per LoCoMo question on Qwen3-8B.
- **Its ablation (GPT-4.1-mini, LoCoMo):**

  | configuration | score |
  |---|---|
  | full | 84.87 |
  | without the storage schedule | 72.79 |
  | without the utterance filter | 75.58 |
  | without the retrieval plan | 77.92 |
  | without topic segmentation | 82.79 |
  | without memory evolution | 84.42 |

## The directions this points to

1. **Typed memory at write time,** LeanMem's storage schedule. It carries
   the largest ablation effect (−12.1), and LoCoMo's ten conversations
   rebuild in about an hour. It is an architecture change: a new write-time
   router and memory kinds.
2. **Reading-side fixes for the 156 held-gold losses:**
   - answer granularity that the question's own shape asks for (country,
     month);
   - relative dates resolved against the evidence item's date (M64's
     machinery);
   - the 37 held-gold declines.

   Each is a post-pass measured in hours.
3. **Per-question evidence planning,** LeanMem's retrieval plan (−7.0 in
   its ablation). M87's question-gated depth is its first, cheapest slice.

The choice among them is the user's (an architecture decision), and it is
taken after round 7's arms report.
