# Serving myelin's models on `bmb` — the fallback host

`big`'s RTX 3090 is shared, and on 2026-09-22 it spent an evening with ~3 GB
free: ollama's `qwen3:8b` (9.7 GB, reloaded on demand seconds after being
evicted) and home-still's distill embedder (grown to 9.5 GB) are ungoverned
by `gpu-tenant` and stayed resident under our lease. `bmb` (Apple M4 Pro,
48 GB unified memory, Metal) is the fallback for those evenings.

## What is the same, and what is not

| | `big` | `bmb` |
|---|---|---|
| llama.cpp | `cuda-ece963` build, b10450 | Homebrew, **b10450 / ece963f41** — the same commit |
| reader | `Qwen3.5-9B-UD-Q4_K_XL.gguf` | the same file, sha256-verified copy |
| reranker | `bge-reranker-v2-m3-Q8_0.gguf` | the same file |
| embedder | bge-m3 through ollama (`:11434/v1`) | the same bge-m3 blob served by llama-server (`:5811/v1`, CLS pooling) |
| backend | CUDA | **Metal** |
| tenancy | `gpu-tenant` lease required | none; bmb's own llama-swap (`:9292`) and the scribe server share the memory |

**The backend is the confound.** Greedy decoding on Metal can differ from CUDA
on some rows, and every paired CI in this project rests on untouched rows
being byte-identical. So an arm measured here pairs against a base measured
here — `runs/bmb_base`, the shipped operating point re-run on bmb — never
against a CUDA base. The CUDA-vs-Metal delta between the two bases is a
reproducibility number worth recording once.

## Copy the files (once)

From the Mac, streaming through it (bmb has no key on big):

```bash
B=/Users/rebekahbrittain/llm/models
ssh bmb "mkdir -p $B/bge $B/qwen"
ssh big cat /home/ladvien/models/bge-reranker-v2-m3-GGUF/bge-reranker-v2-m3-Q8_0.gguf | ssh bmb "cat > $B/bge/bge-reranker-v2-m3-Q8_0.gguf"
ssh big cat /var/lib/ollama/blobs/sha256-daec91ffb5dd0c27411bd71f29932917c49cf529a641d0168496c3a501e3062c | ssh bmb "cat > $B/bge/bge-m3.gguf"
ssh big cat /home/ladvien/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf | ssh bmb "cat > $B/qwen/Qwen3.5-9B-UD-Q4_K_XL.gguf"
# verify: sha256sum on big against `shasum -a 256` on bmb, for each file
```

Quote nothing with `~` on the remote side: bmb's login shell is zsh and a
quoted `~` is a literal directory name (measured: three empty transfers).

## Run it

```bash
ssh bmb "MYELIN_READER_SLOTS=2 MYELIN_READER_CTX=32768 bash -s" < ops/bmb/serve-models.sh
# local ports 5810/5813 may already be tunnelled to big; use other local ports
ssh -fN -L 127.0.0.1:15810:127.0.0.1:5810 -L 127.0.0.1:15813:127.0.0.1:5813 \
        -L 127.0.0.1:15811:127.0.0.1:5811 bmb
MYELIN_LLM__URL=http://127.0.0.1:15810/v1 \
MYELIN_RERANK__URL=http://127.0.0.1:15813 \
MYELIN_EMBED__URL=http://127.0.0.1:15811/v1 \
  myelin-eval bench ... --out runs/bmb_base
ssh bmb bash -s < ops/bmb/stop-models.sh
```

Overrides are set on the remote side, as on big: `ssh` does not forward the
environment. `MYELIN_READER_THINK_BUDGET=1024` for M44 R2, as on big. There
is no vision projector on bmb; it was never on the text path.

Qdrant stays on `big` (`192.168.1.110:6334`); bmb only replaces the three
model servers. If `big` is down entirely, so is the store, and no arm can run
anywhere.

## Measured, 2026-09-22

First serve, 5-row probe of the shipped LongMemEval_S operating point
(`--mode investigate --k 6 --budget-tokens 4096 --max-steps 2
--select-sufficient --item-digest --digest-dates`), bmb also carrying a
17.6 GB llama-swap model and the 14.5 GB scribe server at the time:

| | `big` (3090, CUDA) | `bmb` (M4 Pro, Metal) |
|---|---|---|
| model load to `/health` 200, all three servers | ~60 s | ~60 s |
| per-row query time, p50 | 7.1 s (500-row base) | **46 s** (5 rows: 40, 46, 64, 52, 43) |
| answers byte-identical to the CUDA base | — | **4 of 5** (one `I don't know.` where CUDA answered) |
| composed evidence identical to the CUDA base | — | **2 of 5** |

Two conclusions, both as predicted above. **The backend is not neutral**:
the selector and the digest are model calls, and three of five rows composed
different evidence on Metal, so a Metal arm cannot pair against a CUDA base
— re-run the base here first. **It is 6–7× slower**: a 500-row run is ~6.5 h
here against ~1 h on the 3090, so bmb is the fallback for a lost evening, not
a second lane. The three servers are stopped after use (`stop-models.sh`);
bmb is a daily driver.
