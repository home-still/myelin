# M73b — events from the day the question names *(pre-registered 2026-09-26, before any row)*

## Why

M73 (`m73-question-dates.md`) found that a question's own date phrase ("what
did I buy **10 days ago**?") resolves, through M19's closed grammar, to a
window that holds the gold session.
- 13 of 18 past-point phrases resolve to such a window.
- Inside the window, the gold **turn** ranks 3rd to 90th, because
  LongMemEval plants facts as asides ("by the way, I just got a smoker
  today"). The top-ranked turn states the answer in 0 of 8 targets.
- The top-ranked M50 **event** states it in 7 of 10.

Research:
- The time-range filter is LongMemEval's own time-aware query expansion
  (Wu et al. 2024, arXiv 2410.10813, §5.4). It raised temporal recall@10
  from 0.550 to 0.722 with GPT-4o. With an 8B extractor the gain vanished,
  because small models hallucinate ranges. Here the grammar fails closed.
- The event calendar is Chronos's (Sen et al., `10.48550/arXiv.2603.16862`).

**A defect fixed first.** Every M50, M50b and M50c event reached the reader
with a second, contradicting date (PR #135). This arm is the first with
events dated once.

## The mechanism

`pipeline/side_block.rs`, `SideKind::Events`:
1. When `time::question_window` finds one past-point window strictly before
   the question's date, read the tenant's events dated inside it from a
   side ledger.
2. Rank them all with the cross-encoder.
3. Append the top 3 after the evidence under an `[events]` header, with
   their own 512-token budget.

What gets no window:
- a period ("last month", which M73 measured as usually meaning the past 30
  days);
- the future ("this weekend");
- today;
- two different days;
- no phrase at all.

With no window, nothing is appended. This replaces M50c's ungated vector
block, so there is one events path.

## The arm

**Population.** The gate fires on **18** questions (14 temporal, 2
single-session-user, 2 multi-session); 11 of them are currently lost under
the official grader.
- Events for all 18 haystacks: 10 from M50/M73, and 8 extracted on
  2026-09-26 with M73's extractor unchanged (Qwen3.5-9B UD-Q4_K_XL on bmb).
- They are built into `data/longmemeval_s_events.ledger`. Every `semantic`
  record in it is an event, and `bench` checks that before the first row.
- The other 482 rows are copied from `runs/m57_bonsai_premise_s1`: the gate
  is shut, so their prompts are byte-identical.

**Configuration.**
- M57's exact command, plus
  `--events-ledger data/longmemeval_s_events.ledger`.
- Output: `runs/m73b_events_s1`.
- Judged by the strict 9B (seeded) and the official grader.

**Stratum gate (the user's rule).**
1. On the 18 rows, the paired difference against M57 under the strict judge
   has a 95% CI excluding zero.
2. The official difference is positive.
3. The 30 abstention rows are unchanged.

At n = 18 that needs roughly +4 net questions.

**Predictions.**
- Of the 11 lost rows, **5 to 7** turn right, which is M73's 7 of 10 top-1
  events, less reader loss.
- At most 1 of the 7 won rows turns wrong.

**Falsifier.** A dated event that is right but distracts the reader: won rows
turning wrong, or declines rising on the 18.

**Diagnostics reported either way:**
- rows with a block, and events per block;
- whether the top event states the answer (M73's measure, on the arm's
  actual blocks);
- `gpt4_59149c78`, whose gold sits 3 days outside its window. It must get
  no block.
