# Defect: no schema's intended field order ever reached the model *(found 2026-09-28)*

## What is wrong

Several mechanisms depend on the order a model writes its JSON fields in:
- M42 puts the answer before the decision to decline;
- M44 R1 puts the reasoning before the answer;
- M61/M71b name each thing before answering.

llama.cpp builds its grammar in the order the schema's `properties` arrive,
so that order is the order the model must write in.

myelin builds schemas with `serde_json::json!`. Without serde_json's
`preserve_order` feature, which no crate in this workspace enables
(`cargo tree -e features -i serde_json`), a JSON object is a `BTreeMap`. So
**every schema's properties go out alphabetically**, whatever order the
source writes them in.

**Measured on the wire (2026-09-28).** The same request was sent to a
llama.cpp server twice, with the properties in the two orders. The model's
output followed the order it was sent:
- properties `{zeta, alpha}` gave `{"zeta": …, "alpha": …}`;
- properties `{alpha, zeta}` gave `{"alpha": …, "zeta": …}`.

## Where the intended order was reversed

Each row is a schema whose source order differs from the alphabetical order
the model actually received:

| schema | intended | on the wire | where it runs |
|---|---|---|---|
| M44 R1 reader (`bench.rs:1660`) | reasoning → answer → evidence_absent | answer → evidence_absent → reasoning | R1 arm only. **Its −0.8 measured "answer first", not "reasoning first".** |
| grounded pass (`bench.rs:1259`, `:1264`) | named → supporting → answer; thing → memory | answer → named → supporting; memory → thing | M61, M71, M71b. The model answered before it named the things. The code check (`accept_grounded`) still held. |
| investigate reflect (`investigate.rs:572`) | sufficient → conflict → next_query → reason | conflict → next_query → reason → sufficient | **the shipped LongMemEval_S path** (`--mode investigate`) |
| investigate support (`investigate.rs:766`) | verdict → missing | missing → verdict | investigate |
| self_ask (`investigate.rs:930`) | ask → answer | answer → ask | M39 (−4.6, in "Do not re-run") |
| extract (`extract.rs:117`, `:144`) | text → kind → t_valid → entities | entities → kind → t_valid → text | the write path |
| events (`events.rs:105`) | subject → verb → object → when → aliases | aliases → object → subject → verb → when | M50, M73b |
| consolidate (`consolidate.rs:291`) | op → target → contradicts_target → reason | contradicts_target → op → reason → target | consolidation |
| trajectory agent (`trajectory_agent.rs:172`) | trajectory → first → last | first → last → trajectory | LME-V2 native agent |
| LME-V2 build notes (`build.rs:604`, `:615`) | procedure_note → hint_note; title → description → content | hint_note → procedure_note; content → description → title | LME-V2 build |

**Unaffected:** schemas whose intended order is already alphabetical, such
as M42's `{answer, evidence_absent}`, and one-field schemas like the
selector's `{keep}`.

The existing tests check the `required` array, whose order serde_json does
keep. They never checked the order that reaches the wire, which is why none
of this was caught.

## What this does and does not change

- **No shipped number changes.** Every number was measured with the order
  that actually went out, and the artifacts reproduce.
- **Some conclusions change:**
  - M44 R1's "a reasoning field before the answer loses −0.8" never tested
    reasoning before the answer.
  - M71b's "names every thing, then answers" answered first.
  - The shipped `reflect` call writes `sufficient` last, after its reason.
  These are marked in their records, not deleted.
- **Fixing it changes the shipped pipeline.** Turning on `preserve_order`
  changes `reflect`, `support` and extraction on the shipped path, so the
  fix is a measured item (BACKLOG "schema field order") with an
  `--evidence-only` control, not a silent flip.
- **New schemas do not wait for it.** M78 names its fields so that the
  alphabetical order is the intended order, and a test pins the order on the
  serialized request.

## Addendum: not every reversal hurt, so each order must be chosen *(2026-09-28, round-5 catalog)*

Tam et al. (2024, *Let Me Speak Freely?*, EMNLP Industry,
`10.18653/v1/2024.emnlp-industry.91`) found that "100% of GPT 3.5 Turbo
JSON-mode responses placed the 'answer' key before the 'reason' key, resulting
in zero-shot direct answering instead of zero-shot chain-of-thought
reasoning". CRANE (Banerjee et al. 2025, arXiv 2502.09061) found that
grammars loose enough to leave room for reasoning preserve it.

So the accident cuts both ways:
- **Where the reversal hurt:** M44 R1, whose reasoning went out after its
  answer, and the grounded pass, which answered before naming anything.
- **Where it helped:** on the wire, the shipped `reflect` writes its
  `reason` before `sufficient`, and `support` writes `missing` before
  `verdict`. That is the reasoning-before-decision order this research
  favours. The *source* orders put the decision first.

**The fix is therefore not "restore the source order".** Each schema's order
is chosen deliberately, reasoning before the decision, and then
`preserve_order` is turned on so the source order is the one path. Then it
is measured as the BACKLOG item says. See `docs/research/sota-catalog-2026-09-28.md` §d.
