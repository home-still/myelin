# Vendored benchmark code

> **Nothing under `longmemeval-v2/` may be edited.** Not a line, not a
> docstring, not a typo fix.

`PLAN.md` §3.3 states the rule and the reason: *"for LongMemEval-V2 we run the
authors' `evaluation/run_eval.py`, their `qa_eval_metrics.py`, and their
`leaderboard/` builders. Reimplementing a benchmark's metric is how people
accidentally publish incomparable numbers."* A patched harness produces a
number that cannot be compared to the leaderboard it is being submitted to,
and the whole point of G1 is that **we do not get to say "SOTA" unless the
harness says it** (`PLAN.md` §1).

If upstream is genuinely broken, the fix is an upstream issue plus a re-pin —
never a local patch.

## Pin

| | |
|---|---|
| upstream | `https://github.com/xiaowu0162/LongMemEval-V2` |
| commit | `2cc8c54` ("Add news section to README with latest updates") |
| license | Apache-2.0 (`longmemeval-v2/LICENSE`) |
| vendored | 2026-09-14, 53 files, 776 KB |
| requires | Python ≥ 3.11; `huggingface_hub numpy openai openai-agents pillow tqdm transformers` |
| torch | `requirements-torch.txt` pins `torch==2.6.0+cu124`; **no such wheel exists for macOS arm64** |

Vendored as a flat copy rather than a git submodule on purpose: the tree is
776 KB, and a submodule's pin survives only as long as upstream keeps the
object reachable. A submission has to be reproducible from this repository
alone.

## How myelin plugs in without forking

`memory_modules/memory.py` keeps a `MEMORY_TYPES` registry and populates it
from imports at the bottom of its own file (lines 225+). Adding `myelin`
there would be a fork, so we do not.

Instead:

1. `adapters/myelin.py` defines `MyelinMemory` and decorates it with the
   harness's own `@register_memory`.
2. `adapters/run_myelin.py` imports that module — which is what runs the
   decorator — writes a `memory_config.json` naming `"memory_type": "myelin"`,
   and then calls `evaluation.harness.main()` with the same argv that
   `evaluation/run_eval.py` builds.

That last step is exactly what `run_eval.py` itself does (it is a wrapper that
sets `sys.argv` and calls `harness_main()`), so our runner is a sibling of
theirs rather than a replacement for it. `run_eval.py` is unusable directly
only because its `--method` argument is a closed `choices=` set; every stage
after it takes `--memory-config-path` and is fully general.

## The one platform deviation

`requirements-torch.txt` pins `torch==2.6.0+cu124` / `torchvision==0.21.0+cu124`.
Those local-version wheels are CUDA-only and do not exist for macOS arm64, so
this workstation installs the same upstream versions without the `+cu124`
tag:

```
.venv/bin/pip install "torch==2.6.0" "torchvision==0.21.0"
```

This is an install-time substitution, not an edit: `requirements-torch.txt`
is untouched. torch is needed here only because
`evaluation/harness.py` tokenizes the memory context to enforce
`--memory-context-max-tokens`; that runs on CPU and the reader, judge and
embedder are all remote HTTP, so the CUDA build would buy nothing even if it
installed.

## Scoring stays theirs

`evaluation/qa_eval_metrics.py` scores 295 of the 451 questions
deterministically and routes the other 156 to an LLM judge.
`leaderboard/compute_lafs.py` computes the LAFS gain that G1 is defined
against. `myelin-eval` calls both and reimplements neither.
