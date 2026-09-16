# M16 — retrieval or reader? An evidence-sufficiency audit of G1's answerable gap

`PLAN.md` M16. Measured 2026-09-15 against the artifacts `m6-g1-breakeven.md` produced, plus two
new full-set runs at M7's operating point (web and enterprise). Judge and reader are the same local
`qwen3.5-9b` UD-Q4_K_XL
(`enable_thinking: false`), reranker `bge-reranker-v2-m3`, embedder `bge-m3` via ollama, Qdrant
1.19.1, one shared RTX 3090.

M6 measured LME-V2-Small at **36.7% / 34.6% overall** against a break-even bar of **51.0**, and
`m6-abstention-gate.md` closed the abstention half: all three memory-side levers measured and
rejected, the last one costing −28.3 points. Four parameter attempts had been spent guessing at the
answerable half. This milestone does not make a fifth. It splits every *wrong answerable* question
by one fact already on disk — **was the answer in the evidence the reader was shown?**

## Verdict

**Retrieval-limited, decisively, at both operating points, and the reader is not the binding
constraint.** The audit was run twice: over the `recall` k=25 artifacts `m6-g1-breakeven.md`
measured, and over a new full-set run at M7's `investigate max_steps=2` point — 646 sufficiency
judgements over 4 runs and 902 questions.

|quantity|`recall` k=25, pooled|`investigate` max_steps=2, pooled|
|---|---|---|
|answerable / wrong|323 / 190|323 / 172|
|**S = P(sufficient \| wrong)**|**7.4%** [4.4, 12.0] (14/190)|**12.2%** [8.1, 17.9] (21/172)|
|P(correct \| sufficient)|81.8% [71.8, 88.8] (63/77)|77.9% (74/95)|
|P(correct \| insufficient)|28.5% [23.2, 34.4] (70/246)|33.8% (77/228)|
|measured overall|35.7%|39.9%|
|**reader-fix ceiling**|**38.8%**|**44.6%**|
|**retrieval-fix ceiling**|**67.6%**|**66.0%**|

Per domain, at the `recall` point:

|quantity|web|enterprise|
|---|---|---|
|S|8.3% [4.3, 15.6] (8/96)|6.4% [3.0, 13.2] (6/94)|
|P(correct \| sufficient)|82.6% (38/46)|80.6% (25/31)|
|reader-fix ceiling|40.0%|37.4%|
|retrieval-fix ceiling|67.0%|68.2%|

The rule fixed before the measurement selects the **S ≤ 0.40** branch on every run measured, and
the ceiling arithmetic says the same thing twice:

- **A perfect reader over today's evidence cannot clear the bar.** Flip every *sufficient + wrong*
  row to correct, change nothing else, and the `recall` point reaches **38.8%** (40.0% web / 37.4%
  enterprise) and the `investigate` point **44.6%** — both below **51.0**, and both are *upper*
  bounds. The reader is worth at most **+3.1 points** (35.7% → 38.8%) at the `recall` point and
  **+4.7** at the `investigate` point. Even web alone at its best measured configuration tops out
  at **50.8%**, 0.2 points short. That is the same shape of answer `m6-abstention-gate.md` reached
  for abstention, now proven for the answerable half: **no reader-side or prompt-side change can
  reach G1 from here.**
- **Retrieval has the headroom.** Put the answer in front of the reader on the 176 *insufficient +
  wrong* rows and it converts them at its measured 81.8%, giving **67.6%** — 16.6 points *above*
  the bar. Even at half that conversion the bar is cleared.

**And the diagnosis was tested, not just stated.** The `investigate max_steps=2` run is an
independent retrieval-side change made for other reasons, so it is an out-of-sample check on the
audit's mechanism — and the audit's own conversion rate predicts its accuracy:

|domain|*insufficient + wrong*|predicted answerable gain|observed|
|---|---|---|---|
|web|88 → 68 (−20)|−20 × 82.6% = **+16.5**|**+14** (72 → 86)|
|enterprise|88 → 83 (−5)|−5 × 80.6% = **+4.0**|**+4** (61 → 65)|

The cell the audit named as the loss shrank, and accuracy rose by very nearly the amount
P(correct | sufficient) predicts, on both domains independently. A diagnosis that predicts the
effect of a change it did not observe is the strongest evidence available here that the split is
real.

So **M17 runs the pre-registered width arm**: `k = 50`, `budget_tokens = 20000`,
`rerank_depth = 50`, `investigate max_steps = 2`, full web set. The latency headroom exists —
11.06 s measured on web against the 26.9 s LAFS breakpoint.

**The pre-registered instrument-reliability clause fired, and the verdict survives it.** M16's plan
said: if *insufficient + correct* exceeds 20% of answerable rows, treat the judge as unreliable and
report it. It is **21.7%** at the `recall` point (20.2% web, 23.2% enterprise) and 23.8% at the
`investigate` point. The clause is honoured in full below — the cell is decomposed, the judge's
discrimination is measured, and the worst case is computed — and the conclusion does not move:
under the most adversarial assumption available (*every* one of those 70 rows is a judge false
negative, and the same false-negative rate holds on the wrong rows) S rises to **56.1%**, which is
the rule's *both* branch, and that branch's instruction is **"M17 runs the width arm first"**. No
assumption consistent with the data reaches the reader-limited branch on the pooled set. The
decision is robust to the instrument.

Total GPU cost: one window, 2 h 20 min. 646 sufficiency judgements, issued one at a time
(`evidence_audit.rs` judges the docket sequentially so a failure costs one row), at ~4.6 s each for
~50 min of wall clock, plus the two full-set runs (~46 min web, ~38 min enterprise).

**Not leaderboard-comparable.** `submission_utils.py:202` requires the evaluator to be `gpt-5.2`;
every number here is scored by the local Qwen3.5-9B, as in M6. This caveat travels with every
figure in this document.

## The decision rule, fixed before the measurement

|S|reading|what M17 does|selected|
|---|---|---|---|
|S ≥ 0.60|reader-limited|no further retrieval tuning for G1; record the ceiling and stop| |
|**S ≤ 0.40**|**retrieval-limited**|**run the width arm on the full web set**|**← 7.4%**|
|0.40 < S < 0.60|both|width arm first, it is cheaper|← 56.1% worst case|

## The 2×2, and what each cell means

At the `recall` k=25 point — the configuration `m6-g1-breakeven.md` reports. The same tables for
the `investigate max_steps=2` point are further down; every cell moves in the same direction and
the verdict does not change.

**web** (240 questions, 168 answerable, 72 correct):

| |wrong|correct|total|
|---|---|---|---|
|**insufficient**|**88**|34|122|
|**sufficient**|8|38|46|
|total|96|72|168|

**enterprise** (211 questions, 155 answerable, 61 correct):

| |wrong|correct|total|
|---|---|---|---|
|**insufficient**|**88**|36|124|
|**sufficient**|6|25|31|
|total|94|61|155|

- *insufficient + wrong* — **retrieval's loss, and the largest cell in both domains**: 88/168
  (52.4%) of web's answerable questions and 88/155 (56.8%) of enterprise's. The answer was never
  put in front of the reader.
- *sufficient + wrong* — the reader's loss. **8 and 6 questions.** This is the whole of the
  reader-side opportunity on the answerable half.
- *insufficient + correct* — the row was scored right anyway: pretraining, a lucky multiple-choice
  guess, or a judge false negative. Bounds the judge's error in the direction that matters, and is
  decomposed below.
- *sufficient + correct* — the system working. 38 and 25.

Per category, S is uniformly low — the diagnosis is not a stratum artefact:

|category|web insuf/wrong|insuf/right|suf/wrong|suf/right|S|
|---|---|---|---|---|---|
|static|30|12|2|16|6.2%|
|dynamic|32|5|3|11|8.6%|
|procedure|18|12|2|10|10.0%|
|gotchas|8|5|1|1|11.1%|

|category|ent. insuf/wrong|insuf/right|suf/wrong|suf/right|S|
|---|---|---|---|---|---|
|static|42|18|3|11|6.7%|
|dynamic|21|3|2|9|8.7%|
|procedure|15|11|1|5|6.2%|
|gotchas|10|4|0|0|0.0%|

`static` is the biggest single block of retrieval loss in both domains (30 and 42 questions), which
is consistent with `datasets/lmev2.rs`: a static-environment answer is one line of one
accessibility tree among ~450-token chunks of thousands, and k=25 either contains that line or the
question is lost before the reader sees anything.

## The ceiling arithmetic

`ceiling = (answerable_correct + credited_cell + abstention_correct) / n_questions`, with the
measured abstention column unchanged (16/72 web, 12/56 enterprise — `m6-abstention-gate.md` settled
it, and this milestone never judged an abstention row).

|counterfactual|web|enterprise|combined 451|
|---|---|---|---|
|measured|36.7%|34.6%|35.7%|
|**reader-fix** — every *sufficient + wrong* row correct|**40.0%**|**37.4%**|**38.8%**|
|**retrieval-fix** — every *insufficient + wrong* row retrieved, read at P(correct \| sufficient)|**67.0%**|**68.2%**|**67.6%**|
|retrieval-fix with a perfect reader|73.3%|76.3%|74.7%|
|break-even bar (average query ≤ 26.9 s)|51.0|51.0|51.0|

**The reader-fix row is the decisive one.** It is below 51.0 in both domains and it is an *upper*
bound: it assumes a reader that never misses an answer that is present, on top of the measured
abstention column. G1 is therefore **bounded from the reader side at 38.8%** on the combined set,
and the only lever with enough headroom is the one that changes what the evidence contains.

## The judge's validity, measured rather than asserted

The sufficiency judge is the same `qwen3.5-9b` that produced the answers — no `gpt-5.2` key exists.
Four internal checks, plus the pre-registered clause that fired. The strongest evidence is external
and already stated in the verdict: the audit's own conversion rate predicted the accuracy of a
retrieval change it never saw, on both domains.

### 1. The label predicts the harness's own verdict by 53 points

|domain|P(correct \| sufficient)|P(correct \| insufficient)|discrimination|
|---|---|---|---|
|web|82.6% [69.3, 90.9]|27.9% [20.7, 36.4]|**+54.7**|
|enterprise|80.6% [63.7, 90.8]|29.0% [21.8, 37.6]|**+51.6**|
|pooled|81.8% [71.8, 88.8]|28.5% [23.2, 34.4]|**+53.4**|

The judge never sees the model's answer or the harness's score — only the question, the reference
answer, and the evidence. Its label nonetheless moves the probability that the harness scored the
row correct from 28.5% to 81.8%, with non-overlapping intervals. **An instrument that does not
measure sufficiency cannot do that.** It is also the check that holds across all four runs
(+54.7, +51.6, +39.6, +49.2).

### 2. The pre-registered 20% clause fired — and here is the cell it fired on

*insufficient + correct* is 34/168 = **20.2%** (web) and 36/155 = **23.2%** (enterprise), against a
threshold of 20%. Decomposing it by the harness's own `eval_function`:

|domain|cell|of which multiple-choice|MC accuracy on insufficient evidence|phrase-match accuracy on insufficient evidence|
|---|---|---|---|---|
|web|34|13 + 1 set-match|14/30 = **46.7%** (chance on 4 options: 25%)|15/80 = 18.8%|
|enterprise|36|12|12/28 = **42.9%**|20/82 = 24.4%|

Roughly 40% of the cell is `mc_choice_match`, where the reader picks A/B/C/D or true/false and beats
chance without needing the answer in the evidence — 46.7% and 42.9% against a 25% floor is what
partial cues plus guessing look like, not what a blind judge looks like. The remainder is
phrase-match at 18.8% / 24.4%, which is where genuine pretraining knowledge of Magento and
ServiceNow conventions lives, alongside whatever judge error there is. The cell is *structurally*
inflated: it is an upper bound on judge false negatives, as M16's plan said, and it is not a
false-negative rate.

**The clause is applied, not argued away**: the verdict line above reports the trigger, and the
sensitivity analysis below carries the worst case rather than the point estimate.

### 3. Worst case, if the whole cell is judge error

Assume every *insufficient + correct* row is a judge false negative, and the same false-negative
rate holds among the *wrong* rows:

|domain|FN rate among correct rows|S worst case|branch|
|---|---|---|---|
|web|34/72 = 47.2%|51.6%|both|
|enterprise|36/61 = 59.0%|61.6%|reader-limited|
|**pooled**|70/133 = 52.6%|**56.1%**|**both**|

Pooled and on web the worst case lands in the *both* band, whose instruction is the width arm
first. Only enterprise alone, under the maximally adversarial assumption, crosses 60% — and that
assumption also requires believing that a judge with +51.6 points of discrimination is wrong on
every row it called insufficient and right. **The action M17 takes is the same across the whole
range.**

### 4. The deterministic proxy, and the one direction it can witness

Gold-token recall ≥ 0.8 (multiset intersection over `bench::normalize` tokens), on the
`norm_phrase_set_match*` families only — 118 web rows and 108 enterprise rows where a gold string is
literal text:

|domain|agreement|proxy sufficient, judge insufficient|proxy insufficient, judge sufficient|**decisive direction**|
|---|---|---|---|---|
|web|84/118 (71.2%)|33|1|**47/48 (97.9%)**|
|enterprise|47/108 (43.5%)|58|3|**24/27 (88.9%)**|

The disagreement is almost entirely one-sided, and that is expected rather than alarming: a
four-token gold scattered across ~9,000 tokens of accessibility tree satisfies token containment
without the evidence *stating* the answer. Token recall over-calls sufficiency, so it is a
trustworthy witness in exactly one direction — where the gold's tokens are **absent**, the evidence
cannot contain the answer. In that direction the judge agrees **47/48** and **24/27**. The raw
43.5% on enterprise is the proxy's failure, not the judge's; it is reported because pretending the
proxy is a second opinion would be worse than publishing its asymmetry.

## Five labels a reader can check

Verbatim, from `myelin-eval evidence-audit` (which dumps up to five per diagnostic cell, evidence
truncated to 1,200 characters).

**`sufficient + wrong` — the reader's loss, and it looks exactly like M7's failure mode.**
`2b1a8fc6`: *"Which module should I use? Give me the parent and child module names…"*, gold
`Marketing, Cart Price Rules`. The evidence says, verbatim, *"in the Marketing > Promotions menu the
'Cart Price Rules' link is available (bid '243')"* and carries the
`.../admin/sales_rule/promo_quote/new/` page. The reader writes ~800 words, reasons *"the most
direct path is often cited as Marketing → Promotions → Cart Price Rules"*, cycles through five
candidate readings of "parent and child", and boxes **`Marketing, Promotions`** — the intermediate
menu, not the module, argued for from pretraining against the evidence in front of it. The judge's
`sufficient` label is right and the harness's 0.0 is right.

**`sufficient + wrong`, counting.** `36fe48ef`: *"how many links are there in the Toolbox section"*,
gold `six`, boxed `two`. The evidence carries `[1119] heading 'Toolbox'` and the `[1120] list`
beneath it; the reader miscounts it.

**`insufficient + wrong` — retrieval's loss.** `06a5a25f`: *"What does the confirmation pop-up
banner … say when I update an attribute and click Save?"*, gold `Message is added to queue`. The
evidence is the *Update Attributes* form page and the step text *"the admin shows a scheduled task
message ('Update attributes for 16 selected products')"*; the gold banner string appears nowhere in
the 38,032 characters retrieved. The
reader answers from the closest thing it has — the scheduled-task text — and is wrong. No prompt
fixes this.

**`insufficient + wrong`, arithmetic over unretrieved state.** `07a0145f`: *"After clicking downvote
on the first five posts, what will be the sum of their points?"*, gold `-5`, boxed `0`. The five
posts' point values are not in the evidence set.

**`insufficient + correct` — the judge's false-negative bound at work.** `06e965cf` is a
multiple-choice item whose gold is `B` (`Hot, New, Active, Top, Controversial, Most Commented`). The
evidence says only *"The sort menu is currently expanded (showing Hot/New/Active/Top/etc.)"* — a
prefix, not the list. The reader matches the prefix to option B and scores 1.0. `insufficient` is
the defensible label for that evidence, and the row is still correct: precisely the structural
inflation decomposed in §2 above.

## The operating point, at full set

`m7-step-value-curve.md` chose `investigate max_steps=2` on a 60-question `web` subset and said
plainly that *"the levels are provisional until run on the full 240"*. This milestone needed
artifacts at the point we would actually ship, so it ran it — both domains, full set. **This is not
a fifth parameter attempt**: no level was searched, the point was already selected by M7.

|point|domain|overall|answerable|abstention|`memory_query` avg|
|---|---|---|---|---|---|
|`recall` k=25|web|36.7% (88/240)|42.9% (72/168)|22.2% (16/72)|1.83 s|
|**`investigate` max_steps=2**|web|**45.0%** (108/240)|**51.2%** (86/168)|**30.6%** (22/72)|11.06 s|
|`recall` k=25|enterprise|34.6% (73/211)|39.4% (61/155)|21.4% (12/56)|2.12 s|
|**`investigate` max_steps=2**|enterprise|**34.1%** (72/211)|41.9% (65/155)|**12.5%** (7/56)|14.69 s|
|`recall` k=25|**combined 451**|35.7%|—|—|1.97 s|
|**`investigate` max_steps=2**|**combined 451**|**39.9%**|—|—|**12.76 s**|

Paired bootstrap (20,000 resamples, `adapters/paired_ci.py`, `investigate max_steps=2` − `recall
k=25`, same question set so the pairing is real):

|domain|stratum|n|A|B|A−B|95% CI|p|
|---|---|---|---|---|---|---|---|
|web|overall|240|45.0%|36.7%|**+8.3**|**[+2.9, +13.8]**|**0.0026**|
|web|answerable|168|51.2%|42.9%|**+8.3**|**[+2.4, +14.9]**|**0.0104**|
|web|abstention|72|30.6%|22.2%|+8.3|[−2.8, +19.4]|0.1625|
|enterprise|overall|211|34.1%|34.6%|−0.5|[−6.6, +5.7]|0.9423|
|enterprise|answerable|155|41.9%|39.4%|+2.6|[−5.2, +10.3]|0.5621|
|enterprise|abstention|56|12.5%|21.4%|**−8.9**|**[−17.9, −1.8]**|**0.0095**|

**M7's web level holds at n=240, and the point does not transfer to enterprise.** M7 measured
43.3% overall on its 60-question web subset; the full 240 gives **45.0%**, and the gain over
`recall` is significant on both the overall and answerable strata. On enterprise the same
configuration is **−0.5 points overall** with a **significant −8.9-point abstention loss** — the
persistence-manufactures-false-confidence mechanism M7 identified at `max_steps` 3–4 on web appears
at `max_steps=2` on enterprise. M7's caveat is therefore settled in both directions: the *shape* is
real on web and the *levels are domain-specific*, so `max_steps=2` cannot be defended as a single
global default from these data.

Latency is safe on both: **11.06 s** and **14.69 s** average, against the 26.9 s frontier
breakpoint above which the bar jumps 51.0 → 58.6 (LAFS uses the average — `m6-g1-breakeven.md`'s
correction). Enterprise's p95 is 32.60 s and its max 43.15 s, which does not matter to LAFS but is
worth recording: the tail is twice the median.

**Neither point clears 51.0.** The best measured configuration is 45.0% on web and 39.9% combined,
6.0 and 11.1 points short, which is precisely the gap the audit above assigns to retrieval.

## The audit at the shipped operating point

Re-run over the `investigate max_steps=2` artifacts, because a diagnosis of the configuration we
would *not* ship is worth less than one of the configuration we would.

**web** (168 answerable):

| |wrong|correct|total|
|---|---|---|---|
|insufficient|**68**|40|108|
|sufficient|14|46|60|
|total|82|86|168|

**enterprise** (155 answerable):

| |wrong|correct|total|
|---|---|---|---|
|insufficient|**83**|37|120|
|sufficient|7|28|35|
|total|90|65|155|

|quantity|web|enterprise|pooled|
|---|---|---|---|
|S|**17.1%** [10.5, 26.6] (14/82)|**7.8%** [3.8, 15.2] (7/90)|**12.2%** [8.1, 17.9] (21/172)|
|P(correct \| sufficient)|76.7% (46/60)|80.0% (28/35)|77.9% (74/95)|
|P(correct \| insufficient)|37.0% (40/108)|30.8% (37/120)|33.8% (77/228)|
|discrimination|**+39.6**|**+49.2**|**+44.1**|
|S worst case (whole cell is judge error)|55.6%|60.3%|**57.0%**|
|reader-fix ceiling|**50.8%**|37.4%|44.6%|
|retrieval-fix ceiling|66.7%|65.6%|66.0%|

Four observations, and the verdict is not among the things that move.

1. **The retrieval loss shrinks and the verdict holds.** *insufficient + wrong* falls 88 → 68 on
   web and 88 → 83 on enterprise, S rises to 17.1% and 7.8%, and both stay inside the ≤ 0.40
   retrieval-limited branch. The agentic loop is a retrieval-side change and it moved the
   retrieval-side cell, by less than a third of the way.
2. **The reader's share grows but stays small.** *sufficient + wrong* goes 8 → 14 (web) and 6 → 7
   (enterprise): the loop hands the reader more answers it then loses. This is the same mechanism
   `m7-step-value-curve.md` named — more evidence, more confident wrong answers — and it is why the
   reader-fix ceiling rises only to 50.8% on web while the retrieval-fix ceiling stays at 66.7%.
3. **Web's reader-fix ceiling lands at 50.8%, 0.2 points under the bar.** The single most
   favourable reading available — the best configuration measured, its best domain, and a reader
   that never misses an answer it is shown — still does not clear 51.0. Nothing about that is
   marginal in practice: it is an upper bound nobody knows how to reach, and the retrieval-fix
   ceiling beside it is 66.7%.
4. **The judge behaves consistently across four independent runs.** Discrimination is +54.7, +51.6,
   +39.6 and +49.2 points; P(correct | sufficient) is 82.6%, 80.6%, 76.7% and 80.0%; the decisive
   proxy direction is 97.9%, 88.9%, 97.6% and 84.6%. The *insufficient + correct* share sits in a
   20.2–23.9% band on all four. None of these were tuned.

## What was built

|piece|where|
|---|---|
|the audit: harness-row reader, judge, 2×2, ceilings, examples|`crates/myelin-eval/src/evidence_audit.rs`|
|the subcommand|`myelin-eval evidence-audit --run <dir> [--limit N]`|
|the verdict cache, flushed per row so an interrupted pass resumes|`<run>/evidence_audit.json`|
|the cell arithmetic as a free function over `(&[HarnessRow], &BTreeMap<_, Sufficiency>)`|`evidence_audit::tally`|
|the deterministic proxy and its disagreement direction|`evidence_audit::{gold_token_recall, proxy_agreement}`|
|Wilson intervals|reused from `attack_live::wilson`|

`judge.rs` grades *answers* and hard-errors on a vendored-harness directory; this module grades
*evidence* and hard-errors on a `bench` directory
(`tests::a_bench_run_directory_is_rejected`). Neither number ever enters a reported accuracy.

### The one prompt revision, and why it was a format change

The plan allowed one documented revision. Rev 1 ended `Reply with exactly one character, 1 or 0,
and nothing else.` On row 29 of the web pass the judge replied `"The evidence provided describes"`,
which the audit refused to coerce into a verdict — correctly, since a defaulted `insufficient` is
indistinguishable from retrieval's loss and moves S in the direction that licenses more tuning.

Rev 2 keeps **every judgement clause byte-identical** and changes only the output channel: a JSON
Schema (`{"sufficient": boolean}`, `additionalProperties: false`) through
`CompletionRequest::with_schema`, i.e. the mechanism `extract`, `consolidate` and
`pipeline::adjudicate` already use where a free-form answer is unusable. Format compliance is
enforced by constrained decoding instead of hoped for. The 28 rev-1 verdicts were **discarded**;
every label in this document comes from rev 2. No judgement criterion was touched, so there are no
two label distributions to compare — that is the point of fixing the channel rather than the rubric.

## Reproduce

```bash
ssh big gpu-tenant claim coding
ssh big "MYELIN_READER_SLOTS=4 MYELIN_READER_CTX=65536 bash -s" < ops/big/serve-models.sh
ssh -N -L 5810:127.0.0.1:5810 -L 5813:127.0.0.1:5813 big &

myelin-eval evidence-audit --run runs/myelin_k25_web_small --limit 5   # cost probe; idempotent
myelin-eval evidence-audit --run runs/myelin_k25_web_small
myelin-eval evidence-audit --run runs/myelin_k25_enterprise_small

# the operating point, both domains (needs myelin-mcp and the reranker too)
myelin-mcp --serve 127.0.0.1:7447 --collection myelin_lme_v2_small --ledger data/lme_v2_small.ledger
PYTHONPATH=crates/myelin-eval/vendor/longmemeval-v2:crates/myelin-eval/adapters \
  .venv/bin/python crates/myelin-eval/adapters/run_myelin.py \
  --data-root /tmp/lmev2 --domain web --tier small --mcp-url http://127.0.0.1:7447/mcp \
  --mode investigate --max-steps 2 --k 25 --budget-tokens 10000 \
  --output-dir runs/myelin_inv2_web_small \
  --evaluator-base-url http://127.0.0.1:5810/v1 --evaluator-model Qwen/Qwen3.5-9B
# ... and --domain enterprise --output-dir runs/myelin_inv2_enterprise_small

myelin-eval evidence-audit --run runs/myelin_inv2_web_small
myelin-eval evidence-audit --run runs/myelin_inv2_enterprise_small
.venv/bin/python crates/myelin-eval/adapters/paired_ci.py \
  runs/myelin_inv2_web_small runs/myelin_k25_web_small
```

The reader context override is **mandatory**: audit prompts carry the whole evidence set (median
9,510 tokens on web, max 13,423), and the default `-c 16384 -np 4` gives each slot 4,096 tokens and
returns HTTP 400 `exceed_context_size_error`. Overrides must be on the **remote** side of the `ssh`
command; ssh does not forward the environment.

`/tmp/lmev2` is a scratch materialisation and does not survive a reboot. Re-fetched here with
`huggingface_hub.snapshot_download('xiaowu0162/longmemeval-v2', repo_type='dataset')` restricted to
`questions.jsonl`, `trajectories.jsonl`, `haystacks/lme_v2_small.json` and `question_screenshots/*`
(1.1 GB; the two `trajectory_screenshots` tarballs are 5.9 GB and `evaluation/harness.py` never
reads them). All three files verified against the release's own `checksums.sha256`, and the
re-materialised `questions.json`/`haystack.json` are **byte-identical** (SHA-256) to the ones M6's
runs used, which is what makes these audits and M6's numbers the same measurement.

## For the record

**This milestone spent zero new parameter attempts.** M6's three-attempt cap stands: no retrieval
or reader level was searched here. The two full-set runs use the levels M7 had already selected,
and the width arm (`k = 50`, `budget_tokens = 20000`, `rerank_depth = 50`) is M17's, taken only
because the rule fixed in advance selected it.
