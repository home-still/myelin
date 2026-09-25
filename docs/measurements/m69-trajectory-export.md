# M69 — myelin serves the files the winning LME-V2 controller reads *(built and checked 2026-09-25)*

## Why

On the 47-question LME-V2 pilot, with the same local Bonsai controller and
the same 9B reader, AgentRunbook-C's file view scored **82.98** (M54). Our
native bounded tools scored **36.17** (M62b).

The paired trace analysis (2026-09-25) found the reader side identical.
The whole gap is which states the controller names, and the file view
names them well:
- Codex greps an annotated summary and opens states with a helper script.
- The native tools failed their reads (454 of 455) and had a grep scope bug.

Cao et al. 2026 (arXiv 2603.20432) report the same shape: text in files,
worked with shell and code, beats bespoke retrieval tools.

The code-first path to the LME-V2 gate is therefore to make the file view
**myelin's**, rather than a harness export beside it.

## What was built

`myelin_core::pipeline::trajectory_export` renders each trajectory stored in
myelin's ledger (M62's `trajectory` and `trajectory_state` tables) as
`<id>/trajectory.json`, exactly as the harness writes it:
- **serialisation:** Python's `json.dumps(indent=2, ensure_ascii=True) + "\n"`,
  including key order, escapes, surrogate pairs and empty lists;
- **`actions`:** the states' non-empty actions;
- **`screenshot`:** `screenshots/<index:04>.png`.

`myelin-eval trajectories-export --tenant <t> --out <dir> [--check <harness dir>]`
writes a tenant's trajectories and compares them byte for byte against a
harness workspace.

## The check

| tenant | trajectories | byte-identical | differ | missing |
|---|---|---|---|---|
| `lme_v2_small/web` | 100 | **100** | 0 | 0 |
| `lme_v2_small/enterprise` | 100 | **100** | 0 | 0 |

The reference directories are the harness's own workspaces
(`runs/m54_cloud_smoke_prompts/…/trajectories`,
`runs/m54_full_ent_c00_prompts/…/trajectories`).

## What it establishes, and what it does not

- The harness renders its summaries (`TRAJECTORY_SUMMARY_CONCISE/FULL.md`)
  from these files at query time. So every text file the M54 controller can
  read is reproducible from myelin's store.
- **Screenshots are not.** myelin stores text, and the M54 controller is
  text-only by pre-registration: the shim replaces every image with a note.
- When the M54 full pair reports, its number is a measurement of controller
  and reader over files myelin can serve. The adoption PR can name it as
  myelin's agent-history mode.

**Next:**
1. The MCP `trajectories` tool gains `controller=codex`. It exports the
   tenant, runs the controller in that directory, and passes its output
   through `trajectory_tools::evidence`.
2. One pilot question end to end.
3. The native tools' own fixes (grep scope, annotated summary, match and
   state helpers), measured against the same files.
