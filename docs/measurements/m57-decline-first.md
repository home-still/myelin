# M57 — "I don't know" first, then the correction *(pre-registered 2026-09-24; **shipped**: 83.40, +5.0 [+2.2, +7.8], abstention 29/30)*

> **Correction (2026-09-25):** 83.40 was inflated by 21 stale seeded verdicts on declined answers (`defect-2026-09-25-stale-verdicts.md`). The corrected score is **79.20**; vs the 9B it is **+4.20 [+1.20, +7.40]** (was +5.0), so it still ships. It does **not** pass MemPro-15's 80.80: the 2026-09-24 SOTA claim was wrong.

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

---

## Result *(2026-09-24 13:47)* — **clears; ships; first same-size SOTA row passed**

`runs/m57_bonsai_premise_s1` (500 rows, Bonsai PTQ1_0 recorded as the served
model, `reader_premise_clause: true`), judged by the 9B seeded from M55 (151
fresh verdicts; 270 reused only on byte-identical answers).

**1. To ship — against the shipped 9B (`m44_r2_s1_judged`, 78.40):**

| stratum | n | shipped 9B | M57 | Δ | 95% CI |
|---|---|---|---|---|---|
| **overall** | 500 | 78.4 | **83.4** | **+5.0** | **[+2.2, +7.8]** |
| answerable | 470 | 77.4 | 82.6 | +5.1 | [+2.1, +8.1] |
| **abstention** | 30 | 28/30 | **29/30** | +1 | [+0.0, +10.0] |
| multi-session | 133 | 69.2 | 78.2 | +9.0 | [+2.3, +15.8] |
| temporal-reasoning | 133 | 78.2 | 83.5 | +5.3 | [+1.5, +9.0] |
| knowledge-update | 78 | 84.6 | 89.7 | +5.1 | [−2.6, +12.8] |
| single-session-preference | 30 | 26.7 | 36.7 | +10.0 | [−6.7, +26.7] |
| single-session-assistant | 56 | 100.0 | 96.4 | −3.6 | [−8.9, +0.0] |

Past the bar (+3.0, CI excluding zero), and abstention is above the veto
line (29 ≥ 28). 38 rows gained and 13 lost, 18 of the gains multi-session.
**It ships:** Bonsai 27B plus the premise clause is LongMemEval_S's shipped
point, per corpus. LoCoMo and LME-V2 keep the 9B.

**2. Attribution — against M55, Bonsai without the clause (82.20):**
overall +1.2 [−0.6, +3.0], answerable +0.4 [−1.3, +2.1], **abstention
25 → 29/30, +13.3 [+3.3, +26.7]**. The clause does what it was built for
and nothing else measurable. Preference +13.3 (n = 30) came along with it,
not significant.

**Predictions.**
- ✓ Abstention ≥ 28/30: 29.
- ✓ Answerable within −1 of M55: +0.4.
- Overall came in at 83.4, above the predicted ≈ 82.

**Falsifier: half fired, the conjunction did not.**
- The clause *does* fire on some answerable questions. Declines rose
  from 64 to 78 in total, of which 4 are the abstention rows it was meant
  to win. On answerable rows alone they rose from 39 to 49 (a leading
  "I don't know").
- 17 answerable rows were newly declined, and M55 had 12 of them right.
- Answerable accuracy did not fall. It rose 0.4, because gains elsewhere
  (5 preference, 4 knowledge-update, 4 multi-session) outweighed those
  12. That is a real, measured cost hidden inside a net gain. It is the
  first thing to look at if the clause is ever tried on another corpus.

**Against the literature (`myelin-eval standing`):**
- **83.40 vs MemPro-15 (Qwen3-30B) 80.80, the finish line the user set:
  ahead by 2.60.**
- EverMemOS 83.00: +0.40. APEX-MEM 86.20: −2.80.
- Every such row carries `caveat-judge`: they were judged by a frontier
  API, we by a local 9B. So `standing` keeps the gate open, by design. A
  gate closes only on a comparable row. The ratchet floor rises 78.40 →
  83.40 (token F1 56.37 → 60.82).
