# M63 — a fact and its episode are one memory *(L1; pre-registered 2026-09-24, before any row)*

## Why

The bottleneck review (`docs/research/sota-catalog-2026-09-24.md`, BACKLOG
"SOTA push") read `runs/m19_locomo_full`, the shipped LoCoMo run at 69.87:
- **2,806 of 6,651** emitted facts (42%) came from an episode already in
  the same evidence set, so on **1,424 of 1,986** rows (72%) at least one
  of the six slots restated another;
- multi-hop held every gold turn on only 63 of 282 rows (22%), and it
  scored 86% when it did against 36% when it held none.

LoCoMo's multi-hop is list aggregation across sessions ("What exercises has
John done?"), so every distinct memory counts. Compose already drops equal
text and near-equal vectors, keeping the higher-ranked copy. It did not
know that a fact abstracted from an episode (`derived_from`, I4) repeats
that episode.

## The mechanism

`ComposeConfig::dedupe_lineage` (`compose.rs`). In step 1's dedupe, a record
and one it was derived from count as duplicates, and the higher-ranked copy
stays. No model call, no new store, and the budget loop is unchanged: the
freed slot goes to the next distinct candidate. It is the redundancy half of
maximal marginal relevance (Carbonell & Goldstein 1998,
`10.1145/290941.291025`), with the redundancy read from lineage instead of
estimated. A unit test shows the set unchanged with the switch off.

## The arm

- LoCoMo, full 1,986, at the shipped LoCoMo settings: Qwen3.5-9B, `recall`,
  k = 6, plain reader.
- `bench --corpus locomo --mode recall --k 6 --max-steps 2 --dedupe-lineage`
  → `runs/m63_locomo_dedupe`.
- **A fresh base in the same window, on the same binary with the switch
  off** → `runs/m63_locomo_base`. It removes the drift found in M50c's
  diagnostics: only 75% of rows retrieve the same six items as m19 on
  today's code.
- Both judged by the 9B, with the arm seeded from the fresh base.
- It runs on big between M54 chunks (`STOP_big` window).

**Comparisons, paired over 1,986:**
1. **To ship:** against the fresh base. The bar is +3.0 on judge 1–4 with
   the CI excluding zero. **Veto:** adversarial below the fresh base's.
2. Reported: against m19 (69.87), the standing number.

**Predictions.**
- Rows where a fact shares a slot with its episode fall from ~72% to 0.
- **Multi-hop +2 to +5**, because more distinct memories reach the reader.
- Single-hop flat: its one gold turn was usually already there.
- **Overall +0.5 to +2**. This mechanism alone may well *miss* the +3.0
  bar. It is measured alone so that L2/L3 can bundle on top of a known
  effect (the bundle-validated-switches rule).

**Falsifier.** Multi-hop does not rise, or single-hop falls by more than 1.
Either would mean the fact was the better slot-holder and dropping its
episode, or the reverse, lost the precise statement.

---

## Result *(2026-09-24 20:13)* — **falsified: −4.81, every category down**

The fresh base was `runs/m63_locomo_base` (today's code, all switches off):
**70.52**, +0.65 over m19, the drift. The arm was `runs/m63_locomo_dedupe`.
Both ran on the 9B at 6 slots in one window, and the arm was judged seeded
from the base.

| paired over 1,986, vs the fresh base | base | M63 | Δ | 95% CI |
|---|---|---|---|---|
| **judge 1–4** | 70.52 | **65.71** | **−4.81** | [−6.36, −3.25] |
| multi-hop | 59.57 | 52.48 | −7.09 | [−11.35, −3.19] |
| temporal | 60.44 | 55.45 | −4.98 | [−8.72, −1.56] |
| single-hop | 82.52 | 78.36 | −4.16 | [−6.18, −2.14] |
| adversarial | 67.94 | 63.68 | −4.26 | [−7.17, −1.35] |
| declines on 1–4 | 116 | **169** | | |

**The falsifier fired**, and harder than it was written. Multi-hop, the
predicted winner, fell the most.

**What it means.** A fact and the episode it came from are **not**
redundant for this reader. The fact is a short, dated statement. The
episode carries who said it, the surrounding turns and the wording.
Removing either one made the 9B decline more often (+53 declines). The
retrieval-only sweep (M65) saw dedupe as a small coverage gain (+0.9
all-gold); the reader shows the opposite. Coverage counted by lineage is
not what the reader uses. *(Kept as a falsified mechanism; the switch
stays off.)*
