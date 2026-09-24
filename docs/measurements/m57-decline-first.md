# M57 — "I don't know" first, then the correction *(pre-registered 2026-09-24)*

## Why

M55 put Bonsai 27B at **82.20** on LongMemEval_S, against the shipped 9B's
**78.40**, but abstention fell 28 → 25/30 and the veto fired. The three rows
Bonsai lost:

| question | 9B | Bonsai |
|---|---|---|
| How often do I see Dr. Johnson? | I don't know. | You see Dr. Smith, not Dr. Johnson. |
| How long have I been living in my apartment in Shinjuku? | I don't know. | You live in Harajuku, not Shinjuku. |
| How many engineers do I lead … as Software Engineer Manager? | I don't know. | 4 |

The first two are correct corrections that the string-rule scorer cannot
read. The third accepts a false premise.

M56 showed that an external checker cannot fix this on this data: Jev's
false-premise signal fired on 100 answers the judge had right. So the fix
goes in the reader's own instructions. (QA)² (Kim et al., ACL 2023,
`10.18653/v1/2023.acl-long.472`) finds that questions with questionable
assumptions "require a distinct answer strategy": address the assumption,
do not answer through it.

## The arm

`bench --reader-premise-clause` appends to `READER_SYSTEM`:

> If the question assumes something the memories do not state or that they
> contradict — a person, place, event or detail that does not appear as the
> question describes it — begin your reply with "I don't know." and then say
> in a few words what the memories do state.

The reply shape is the one `is_abstention` already reads as a decline ("I
don't know. You see Dr. Smith, not Dr. Johnson."), so the correction
survives for the user and the row scores as the abstention it is. Nothing
else changes.

**Run.**
- Big serves Bonsai (`MYELIN_READER_MODEL=bonsai-27b`, 2 × 16k, thinking
  budget 1,024, no projector).
- The command is M55's full-run command plus the one switch, in two shards
  of 250, merged and closed with `bench --resume` while Bonsai still
  serves:
  `bench --corpus longmemeval-s --mode investigate --k 6 --budget-tokens 4096 --max-steps 2 --select-sufficient --item-digest --digest-dates --reader-thinking --reader-seed 1 --reader-premise-clause`
  → `runs/m57_bonsai_premise_s1`.
- Judged by the 9B, seeded from `runs/m55_bonsai_s1`, then
  `rescore --scorer judge`.
- M54's full run yields big between chunks and resumes afterwards.

**Two comparisons, both paired over the 500:**
1. **To ship:** against the shipped `runs/m44_r2_s1_judged` (9B, 78.40).
   The bar is +3.0 with the 95% CI excluding zero, and abstention no lower
   than 28/30 (the veto). If it clears, Bonsai plus this clause becomes
   LongMemEval_S's shipped point: per-corpus shipped model, and the clause
   on for LongMemEval_S.
2. **Attribution:** against `runs/m55_bonsai_s1_judged` (Bonsai without the
   clause, 82.20). This measures the clause alone.

**Predictions.** Abstention 25 → ≥ 28/30: both premise corrections become
"I don't know. …", and possibly the "4". The answerable rows move less than
−1 against M55 Bonsai. Overall ≈ 82, which is +3.5 to +4 over the shipped
9B.

**Falsifier.** The clause fires on answerable questions whose wording
differs from the memories, such as paraphrases or renamed entities. That
would show as answerable declines rising past M55 Bonsai's 64, and
answerable accuracy falling more than 1.5 against it.
