# M74 — the selector reads the line that answers *(pre-registered 2026-09-26, before any row)*

## Why

Every shipped LongMemEval_S row passes through the sufficiency selector
(`--select-sufficient`, M21):
- once per probe;
- once over the pooled candidates.

It decides which records enter the evidence. **It reads only the first 400
characters of each candidate** (`select.rs`, `CANDIDATE_CHARS`), and a
candidate is an episode of up to 512 tokens, about 2,000 characters.

LongMemEval plants its facts as asides late in a turn. The M73 example is
"…by the way, I just got a smoker today", deep inside a BBQ-sauce
conversation. The head the selector reads is usually not the part that
answers.

The loss anatomy (2026-09-26) counts **57 answerable losses that hold part
or none of their gold turns**:
- multi-session: 18 part;
- temporal: 12 part, 12 none;
- preference: 8 none.

Some of those gold turns were in the 25-candidate pool and lost at
selection.

Research:
- MemPro's evolved pipeline gained +1.47 from "focused evidence snippets
  before integration" (Liu et al. 2026, arXiv 2606.00619, App. A.1).
- RECOMP (Xu et al. 2023, arXiv 2310.04408) is the extractive form.

## The mechanism

`RetrieveConfig::select_focus` → `turn_windows::focus_views`:
- Before each selection, every line of every candidate is scored by the
  cross-encoder against the question, in one call.
- The selector is shown each candidate's best line, marked `…` when it is
  not the first. A one-line record is shown whole.
- It applies in `recall`'s per-probe selection and in `investigate`'s pooled
  selection alike.
- The records, their order after selection, and what the reader sees are
  otherwise unchanged. Only the selector's view differs.

## The arm

Selection runs on every row, so **all 500 rows are re-run**, with M57's exact
command plus `--select-focus`:
- Bonsai PTQ1_0;
- thinking at 1,024 tokens;
- reader seed 1;
- the premise clause.

Output: `runs/m74_focus_s1`. It is judged by the strict 9B (seeded from M57)
and the official grader.

**Gate (pre-registered, the user's stratum rule adapted to a mechanism that
touches every row).** The switch enters the bundle only if all three hold:
1. **Retrieval, reader-free:** the share of the 470 answerable rows holding
   every gold turn (`coverage.rs::is_found`) rises, with a paired 95% CI
   excluding zero.
2. The strict paired difference over all 500 is positive.
3. Abstention is not below 29/30.

**Predictions.**
- All-gold-held rises by 2–5 points of share, mostly on multi-session and
  temporal "part" rows.
- Strict overall +0.4 to +1.2, too small to ship alone, which is why it is
  gated for the bundle.
- The official grader moves the same direction.

**Falsifier.** All-gold-held flat or down: the head was not what was hiding
the gold, or the line-level cross-encoder picks decoys. Report which, from
the selector's views on the lost rows.

**Diagnostics:**
- selector call failures and degradations, which must stay at M57's 0.0%;
- mean cross-encoder latency added per row.
