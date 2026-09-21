# M35 — two more allocation nulls, and the diagnosis that redirects the project

**Verdict: the LME-V2 gap is an abstention gap, not a retrieval gap.** The
milestone set out to measure the two mechanisms M34's null pointed at. Both
are nulls or worse, and the diagnostic that explains why is the result worth
keeping.

| arm | population | Δ vs shipped | 95% CI | p |
|---|---|---|---|---|
| `typed_probes` | web, n = 240 | **−2.92** | [−8.33, +2.50] | 0.3778 |
| `premise_analysis` | web, n = 240 | **−8.75** | [−15.00, −2.92] | **0.0072** |
| `kind_quota` | — | built, tested, **unmeasured** | — | — |

Neither ships. `ComposeConfig::kind_quota` is implemented and its
pre-registered rule is untouched for whoever runs it.

## 1. The diagnosis

Our shipped configuration on LME-V2 tier-small, split by what the question
asks for:

| stratum | n | share | ours |
|---|---|---|---|
| answerable | 323 | 72% | **46.75%** |
| **abstention** | **128** | **28%** | **17.97%** |
| combined | 451 | | 38.58 |

**We answer when we should decline about 82% of the time.** If abstention
merely matched our own answerable rate, the combined number would be
**46.75 — worth +8.17 points, which is 41% of the entire gap to
AgentRunbook-R's 58.60.**

No allocation mechanism addresses that. It is not a question of which
evidence wins a slot; it is the reader producing an answer for a question
whose premise the store cannot support.

`docs/research/11-frontier-2026.md` §D.1 said so before any of this ran:

> AgentRunbook-R reduces retrieval+reading errors vs RAG but does **NOT**
> improve abstention (presents evidence that misleads reader into using it
> instead of rejecting). AgentRunbook-**C** also improves abstention because
> the memory module is instructed to explicitly flag wrong
> premises/contradictions.

That is the documented difference between their 58.6 and their 72.5, and it
is not a retrieval mechanism. M32 quoted this passage and then spent M34 and
most of M35 on allocation anyway.

## 2. The pattern in our own record

| milestone | mechanism | judged effect |
|---|---|---|
| M25–M31 | width, budget, MMR, pool rerank | ~0 |
| **M32** | **sufficiency selector — a model call that decides** | **+5.8** |
| M34 | two new knowledge pools, 33% of emitted evidence | +0.22 |
| M35 | typed probes | −2.92 |

Every mechanism that re-ranks or re-allocates existing candidates is a null.
The one that moved a number put a model decision in the loop. That is a
consistent signal across eleven milestones and it should be read as one.

## 3. `typed_probes`: −2.92, and the emitted set barely moves

M23 D1 built it; until M34 minted the pools it had nothing to aim at, so this
is its first measurement. Web, n = 240, against the M34 store and operating
point:

| | base | typed | Δ |
|---|---|---|---|
| combined | 44.58 | 41.67 | −2.92 |

Why it does nothing is visible in the emitted mix:

| kind | base | typed |
|---|---|---|
| episodic | 66.7% | 68.1% |
| procedural | 22.7% | 21.2% |
| semantic | 10.6% | 10.7% |

Tagging probes changes which candidates enter the pool; the pool is then
unioned across steps and re-composed by one fused ranking, which puts back
almost exactly the same mix. This is M21's per-probe null in a second
location, and for the same structural reason — `investigate` re-composes, so
probe-level interventions wash out.

### Stopped early, and why that is not narrowing the population

The pre-registered rule was ≥ +3.0 combined over the 451. With web at −2.92
and web being 53.2% of the set, enterprise would have to return **+9.73** for
the arm to clear it. The decision was determined, so the enterprise run was
cancelled 18 minutes in.

This is a stated early stop on an arithmetic argument, published with the
number that determines it — not a population quietly reduced to the half that
looked better. The reverse would have been reporting web alone had it been
*favourable*.

## 4. `premise_analysis`: the right target, the wrong instrument

M23 A3 replaces a bare insufficiency statement with an explicit premise
analysis. It is the mechanism closest to what AgentRunbook-C does, and it had
never been measured on anything.

| stratum | n | base | premise | Δ |
|---|---|---|---|---|
| answerable | 168 | 52.98 | 38.10 | **−14.88** |
| **abstention** | 72 | 25.00 | **30.56** | **+5.56** |
| combined | 240 | 44.58 | 35.83 | **−8.75** |

Paired: **−8.75, 95% CI [−15.00, −2.92], p = 0.0072.** A significant harm, and
the first arm in this project to be significantly *negative* rather than null.

**It does what the literature says it does.** Abstention improves by 5.56.
The mechanism is aimed correctly. It just charges 14.88 points on the other
72% of the set.

### The failure is selectivity, and it is measurable

How often the reader declines, by stratum:

| | answerable | abstention |
|---|---|---|
| base | 14/168 = **8.3%** | 21/72 = 29.2% |
| premise | 45/168 = **26.8%** | 27/72 = 37.5% |

Premise analysis multiplies declines by **3.2× on questions that have an
answer** and only **1.3× on questions that do not**. The gate is not merely
loud; it is *anti-selective* — it fires hardest exactly where it should stay
silent.

So the switch stays off, and the number that killed it goes in its doc
comment. What it establishes is more useful than a pass would have been: the
abstention lever is real and reachable, and the problem is discrimination,
not volume.

Note the ceiling this also exposes. Even in the base arm the reader declines
on only **29.2%** of abstention questions — it answers 70% of questions whose
premise the store cannot support. That is the headroom.

## 5. `kind_quota`: built, tested, unmeasured

`ComposeConfig::kind_quota` allocates `k`'s slots per record kind instead of
handing them to one fused ranking — AgentRunbook-R's published **top-6
events, top-3 notes**, raw states taking the rest. The case for it is M34's
measured allocation gap:

| | AgentRunbook-R | ours (M34) | Δ |
|---|---|---|---|
| raw states | 52.6% | 66.7% | −14.1 |
| **events** | **31.6%** | **10.6%** | **+21.0** |
| notes | 15.8% | 22.7% | −6.9 |

It is a **reordering, never a filter**: three buckets — reserved, raw,
overflow — each internally in the incoming rank order, so no tie-break is
invented, nothing is dropped that `None` would have kept, and the budget loop
keeps its `k_bound` / `dropped_for_tokens` semantics. An arm on it therefore
measures allocation rather than allocation plus loss.

Deliberately **not tunable from the wire**: the allocation under test is the
paper's, and a tunable one invites fitting it to the test set.

Five tests pin the contract — a quota guarantees slots the fused ranking
denied; it is also a ceiling with overflow last; it never drops a candidate;
overflow backfills when raw states run out; off is the unmodified order. One
of them caught a bad fixture of mine: eight identical `"raw"` texts collapsed
under `compose`'s dedup, so the test was measuring dedup instead of
allocation.

It is wired MCP → adapter → runner, which `untrusted_max` never was —
that switch is `bench`-only and could not have been measured on the LME-V2
path at all.

**Unmeasured, and its pre-registered rule stands**: ships on for `investigate`
at ≥ +3.0 combined over the 451 with a paired CI excluding zero. It was
deprioritised, not refuted, when §1 showed 41% of the gap sitting somewhere
else.

## 6. Reproduction

```sh
ssh big gpu-tenant claim coding
ssh big "MYELIN_READER_SLOTS=1 MYELIN_READER_CTX=65536 bash -s" \
    < ops/big/serve-models.sh
ssh -N -o ServerAliveInterval=15 -L 5810:127.0.0.1:5810 \
    -L 5813:127.0.0.1:5813 big &
export MYELIN_QDRANT__URL=http://192.168.1.110:6334
myelin-mcp --serve 127.0.0.1:7446 \
  --collection myelin_lme_v2_small --ledger data/lme_v2_small.ledger &

cd crates/myelin-eval
export OPENAI_API_KEY=local
common="--data-root ../../data/lmev2 --domain web --tier small --k 25 \
  --budget-tokens 10000 --mode investigate --max-steps 2 --select --undated \
  --ledger ../../data/lme_v2_small.ledger \
  --evaluator-base-url http://127.0.0.1:5810/v1 \
  --evaluator-model Qwen/Qwen3.5-9B --reader-model Qwen/Qwen3.5-9B"

PYTHONPATH=vendor/longmemeval-v2:adapters ../../.venv/bin/python \
  adapters/run_myelin.py $common --typed-probes \
  --output-dir ../../runs/m35_typed_web       # 1h01m
PYTHONPATH=vendor/longmemeval-v2:adapters ../../.venv/bin/python \
  adapters/run_myelin.py $common --premise \
  --output-dir ../../runs/m35_premise_web     # 35m56s
```

Both arms pair against `runs/m34_pools_web`, which is the shipped
configuration on the same store — the `store_fingerprint` check M34 added
makes that checkable rather than assumed.
