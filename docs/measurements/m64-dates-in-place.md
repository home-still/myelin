# M64 — the date where the phrase is, in words *(L2; pre-registered 2026-09-24, before any row)*

## Why

The bottleneck review of `runs/m19_locomo_full` (69.87, the shipped 9B)
found the temporal reading losses have a mechanical cause:
- **The resolution is out of reach.** Compose resolves relative dates
  (M19's `resolve_relative`) but appends them after the whole record. For
  "I got a puppy two weeks ago" the annotation
  `(two weeks ago = 2023-07-28..2023-08-03)` sat ~2,000 characters after
  the phrase, at the end of a 2,193-character episode. The reader answered
  with the session stamp, 2023-08-11. Of the temporal rows whose gold turn
  holds a relative phrase, 139 were right and **86 lost** (58 wrong, 28
  declined).
- **The ISO form is judged wrong.** 18 temporal rows the deterministic
  temporal scorer counts right were judged wrong. The reader had copied the
  ISO range (`2023-06-02..2023-06-08` for "the week before 9 June 2023").

## The mechanism

`ComposeConfig::inline_dates`:
- Each resolution goes right after its phrase, bracketed and in words, as
  LoCoMo's gold answers write dates: "two weeks ago [28 July – 3 August
  2023]".
- The tokenizer now carries source byte spans, so `resolve_relative`
  returns where each phrase stands. The spans are tested on every resolver
  case, and a unit test gives the exact rendering of the real Coco turn.
- The record is never rewritten, and nothing is re-ranked; the selection is
  identical.

This is TReMu's resolved timeline (Ge et al. 2025,
`10.18653/v1/2025.findings-acl.972`) and Chronos's resolved event dates
(arXiv 2603.16862), done at read time.

## The arm

- LoCoMo full 1,986 at the shipped settings (9B, `recall`, k = 6) plus
  `--inline-dates` → `runs/m64_locomo_inline`.
- It is paired against **M63's fresh base** (`runs/m63_locomo_base`, the
  same code with every switch off).
- Judged by the 9B, seeded from that base.
- It runs on big in the next window between M54 chunks.

**Comparisons, paired over 1,986:**
1. **To ship:** against the fresh base. The bar is +3.0 on judge 1–4 with
   the CI excluding zero. **Veto:** adversarial below the fresh base's.
2. Reported: against m19 (69.87).

**Predictions.**
- **Temporal +5 to +9** on its stratum (n = 321): roughly 30 of the 86
  relative-date losses, plus most of the 18 format rows.
- Other categories flat, since their items rarely carry relative phrases.
- **Overall +1 to +2.** It may miss the +3.0 bar alone, and it is measured
  alone so that bundles build on known effects.

**Falsifier.** Temporal does not rise, or the reader starts answering with
the bracketed range for questions that asked something else (single-hop
falls by more than 1).
