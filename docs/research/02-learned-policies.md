# 02 — Learned Memory-Policy Evidence Pack

Papers where the **memory-management policy itself is learned**, plus adaptive-retrieval control policies. Every claim carries DOI + page/line pointer where quoted; unverifiable items tagged `[UNVERIFIED]`.

Shared vocabulary: memory kinds `episodic|semantic|procedural|working`; pipeline `ingest→extract→consolidate→index→retrieve→rerank→compose→forget`; primitives `dense|sparse/BM25|hybrid+RRF|rerank|graph-expand|scope-filter`; eval axes `accuracy|token-cost|p95-latency|grounding/provenance|robustness/poisoning`.

---

## 1. Mem-a (Mem-alpha): Learning Memory Construction via RL

- **DOI**: 10.48550/arXiv.2509.25911 (arXiv:2509.25911v1). Wang, Takanobu, Liang, Mao, Hu, McAuley, Wu, 30 Sep 2025.
- **(1) Learned decision**: the **write / what-to-write** policy. The agent sees conversations `C={c_1..c_n}`; at each chunk `t` it issues a *sequence* of write operations `a_t=(a_t^(1),..,a_t^(K_t))`, each `a_t^(k) in A_write = {memory_insert, memory_update, memory_delete}` (structured function call with record id, memory type, content). Decides *what* to store, *how to structure* (core vs semantic vs episodic), *when to update/delete*. (§3.1.1).
- **(2) MDP** (implicit; episodic RL over chunk stream, GRPO):
  - State: current memory `M_{t-1}`, chunk `c_t`.
  - Action: sequence of write calls `a_t`; `M_t = M_{t-1}^(K_t)` after applying calls.
  - **Reward** (four components, §3.1.2, verbatim):
    - Correctness `r_1` — downstream QA accuracy over full history via a **frozen** RAG pipeline (BM25 `phi` top-k + frozen generator). `r_1 = l/m` on SQuAD.
    - Tool-call format `r_{2,t} = sum_k s(a_t^(k)) / K_t`, `s in {0,1}` = 1 if formats+executes.
    - Compression `r_3 = 1 - l_m/l_c` (l_m = memory length, l_c = chunk length).
    - Memory-content `r_{4,t} = sum_k v(a_t^(k)) / K_t`, `v in {0,1}` = semantically-valid update (checked by **Qwen3-32b** validator).
    - **Final reward: `r_t = r_1 + r_{2,t} + beta*r_3 + gamma*r_{4,t}`** (r_2 weight fixed at 1). Advantage `A_t = (r_t - mu_group)/(sigma_group + eps)`.
    - Note: §4.1 states `beta=0.05, gamma=1`; §4.4 ablation default is `beta=0.05, gamma=0.1`. Table 4 authoritative: best `beta=0.05,gamma=0.1 -> Avg 0.642`. `gamma=0` catastrophic (0.543). Higher beta shortens memory, degrades perf (`beta=0.4` -> 0.509).
- **(3) Training**: GRPO (verl), backbone **Qwen3-4B** (Qwen3-8B worse). `lr=1e-6, batch=32, grpo_rollout_n=8, 3 days on 32xH100`, 205 steps. Data: 4,139 instances compiled, **stratified-balanced subset of 562** used. Max training length 30k tokens.
- **(4) Results vs hand-written baselines** (Table 2, MemoryAgentBench; Perf = F1/Acc):
  - Long-Context (Qwen3-32B): Ava **0.461**, Mem 33K
  - RAG-Top2: Avg **0.502**, Mem 207K
  - MemAgent: Avg **0.198**, Mem 0.92K
  - MEM1: Avg **0.071**, Mem 0.21K
  - **Mem-a-4B: Avg 0.592, Mem 129K tok** (best).
  - RL-boost (Table 3 val): base Qwen3-4B 0.389 -> **Mem-a 0.642** (+0.253); gpt-4.1-mini same framework 0.517. Gains from RL, not structure.
  - Length generalization: trained on ~20K-30K avg, generalizes to **>400K tokens (up to 474K Multi-Doc), >13x training**.
- **(5) Transferability**: policy **tied to fine-tuned Qwen3-4B**; memory architecture itself modular/decoupled (substitute structure w/o retraining), but the policy is not model-agnostic. Even GPT-4o struggles with tool selection; small models overwhelmed by complex tool sets — motivates learning.
- **(6) Inference cost**: trained 4B policy emitting write calls greedily; Qwen3-32b validator only at train time; BM25 + frozen generator for QA. Test memory ~129K avf tokens at 400K+ input.

---

## 2. Memory-R1: RL for Memory Manager + Answer Agent

- **DOI**: 10.48550/arXiv.2508.19828 (arXiv:2508.19828v1). Yan et al., 27 Aug 2025. ACL 2026 (2026.acl-long.583).
- **(1) Learned decisions**: two agents:
  - **Memory Manager**: chooses `o in {ADD, UPDATE, DELETE, NOOP}` plus content `m'` per new fact (memory maintenance).
  - **Answer Agent**: **Memory Distillation** policy — from up to 60 RAG-retrieved entries, *selects the useful subset* and reasons (learned retrieval-filter / compose).
- **(2) MDP / reward** (verbatim):
  - Manager `(o,m') ~ pi_th(·|x, M_old)`; **PPO** `J(th)=E[min(rho_th A, clip(rho_th,1-eps,1+eps)A)]` with `rho_th = pi_th(o,m'|x,M_old)/pi_old(...)`; **GRPO** `J(th)=E[(1/G)sum_i rho_th^(i) A_i - beta D_KL[pi_th||pi_ref]]` with `A_i = (r_i-mean(r))/std(r)`.
  - **Reward (both)**: `R_answer = EM(y_pred, y_gold)` — pure **exact match**. "Exact-match rewards alone suffice to teach the Memory Manager." Frozen Answer Agent during Manager training.
  - Pipeline: LLMExtract(t_i) -> RAG(f_i, M) -> MemoryManager(o_i) applies op -> (QA) retrieve up to 60 -> AnswerAgent(y | q, M_ret) w/ distillation.
- **(3) Training**: PPO **and** GRPO; **LLaMA-3.1-8B-Instruct** + **Qwen2.5-7B-Instruct**. 4xH100-80GB, batch 128, micro 2/GPU, prompt 4096/response 2048; actor lr 1e-6, critic 1e-5; decode tau=1.0 train, greedy eval; 3 runs avg. Data: LOCOMO 1:1:8 split — first dialogue **152 QA pairs** trains, 81 val, 1,307 test.
- **(4) Results vs hand-written baselines** (LOCOMO overall F1/B1/J):
  - Mem0 baseline: LLaMA F1 30.41/B1 22.22/J 45.68.
  - **Memory-R1-GRPO (LLaMA-3.1-8B): F1 45.02/B1 37.51/J 62.74**.
  - Memory-R1-PPO: F1 41.05/B1 32.91/J 57.54.
  - vs Mem0: +68.8% F1, +68.9% B1 (paper text prints 68.9 F1/48.3 B1 — mislabeled vs table; table values authoritative), +37.3% J. Matches abstract (+48/69/37).
  - Qwen-2.5-7B: Mem0 F1 30.61/23.55/53.30 -> GRPO F1 43.14/B1 36.44/J 61.51 (transfers across backbones).
  - Ablations: Memory Manager RL-only F1 32.55->33.05 (PPO/GRPO); Answer base F1 26.73 -> PPO 34.48 -> GRPO 37.54 (Table 3; NOTE column ordering internally inconsistent in HTML).
  - **Memory Distillation** ablation: no-distill F1 34.37/B1 40.95/J 60.14 -> with-distill F1 37.51/B1 45.02/J 62.74 (~+9% F1; Table 4).
  - Answer Agent gains more with stronger Manager: LLaMA mgr F1 +10.10; GPT-4o-mini mgr F1 +19.72 (Fig 3). GRPO faster initial convergence; both reach similar final reward (Fig 4).
- **(5) Transferability**: framework transfers between LLaMA-3.1-8B and Qwen-2.5-7B, but each policy **tied to its fine-tuned backbone** — not model-agnostic. 152-example efficiency makes per-backbone retraining cheap.
- **(6) Inference cost**: runs fine-tuned 8B/7B policy for manager + answer; RAG retrieves up to 60 candidates; distillation is learned filtering in token stream. No value model (GRPO).

---

## 3. Self-RAG: adaptive retrieve-or-not + self-critique

- **DOI**: 10.48550/arXiv.2310.11511. Asai, Wu, Wang, Sil, Hajishirzi. ICLR 2024.
- **(1) Learned decisions**: **retrieve-or-not** (`Retrieve in {yes,no,continue}`) and passage-trust via critique tokens, learned as next-token predictions. Tokens: `Retrieve`, `IsRel` (relevant?), `IsSup` (fully/partially/no support), `IsUse` (5..1).
- **(2)**: standard **LM objective** over corpus with reflection tokens inserted **offline** by a separate **critic** (LLaMA-2-7B) supervised by GPT-4 labels. Inference adaptive retrieval via **threshold**: retrieve if `P(Retrieve=yes)` normalized over output tokens exceeds a set threshold (Appendix A.3). Segment-level beam search; segment score = weighted linear sum of normalized critique-token probs `f(y_t,d,Critique)=p(y_t|x,d,y_<t)+S(Critique)`.
- **(3) Training**: two-stage (critic; then generator 7B & 13B on critique-token corpus). "Far lower cost than RLHF/PPO — no reward model." Retriever off-the-shelf dense.
- **(4) Results vs baselines**: PopQA 13B **55.8%** vs Llama2-13B 14.7%, Alpaca-13B 24.4%; 7B 54.9% vs Ret-ChatGPT 51.8%. TriviaQA 69.3% (Alpaca 66.9, Llama2 47.0); PubHealth 74.5% (Alpaca 51.1); ARC-Ch 73.1% (Alpaca 57.6, Llama2 29.4). ASQA citation precision 70.3% / recall 71.3% (13B); occasionally 7B >= 13B on factual precision. Beats retrieval-augmented ChatGPT on 4 tasks, Llama2-chat + Alpaca on all.
- **(5/6)**: trained 7B/13B policy tied to model; the *threshold mechanism* is model-agnostic. Inference: one extra token group in vocab (negligible), on-demand retrieval, segment beam search = main added cost.

---

## 4. Adaptive-RAG: query-complexity router

- **DOI**: 10.48550/arXiv.2403.14403. Jeong et al., NAACL 2024.
- **(1) Learned decision**: classify query complexity -> route to {no retrieval, single-step, iterative/multi-step}.
- **(2)**: supervised classification, not RL. **T5-Large** classifier; labels **auto-derived** from LLM-answer correctness on samples + dataset inductive biases.
- **(4) Result**: three-class router matches always-expensive (iterative) baselines in accuracy at **substantially lower cost**. [Per-dataset numbers [UNVERIFIED] — abs page only.]
- **(5/6)**: standalone small model -> **transferable/model-agnostic**; one small forward; no-retrieval path ~zero cost.

---

## 5. Corrective RAG (CRAG)

- **DOI**: 10.48550/arXiv.2401.15884 (v3, 7 Oct 2024). Yan, Gu, Zhu, Ling.
- **(1) Learned decision**: post-retrieval quality assessment -> corrective action `{Correct, Incorrect, Ambiguous}` -> use+refine via **decompose-then-recompose**, web-search fallback, both.
- **(2/3)**: **T5-large-fine-tuned** lightweight retrieval evaluator, NOT RL.
- **(4)**: "significantly improves" RAG across four short/long-form datasets. [Exact deltas [UNVERIFIED].]
- **(5/6)**: standalone small classifier -> **transferable**; one T5-large forward per query + optional web search.

---

## 6. FLARE: confidence-threshold active retrieval

- **DOI**: 10.48550/arXiv.2305.06983. Jiang et al., EMNLP 2023.
- **(1) Learned decision**: **when to retrieve during generation** — per sentence. Predict next sentence `s_hat_t`, detect **low-confidence tokens**, retrieve to regenerate only then.
- **(2) Policy**: **deterministic, hand-set, model-agnostic**. "Triggers retrieval if any token of s_hat_t has probability lower than threshold theta in [0,1]. **theta=0 -> never; theta=1 -> every sentence.**" (theta tuned in [0,1]; exact default ~0.5-0.6 [UNVERIFIED exact value]).
- **(4)**: superior/competitive on 4 long-form knowledge-intense tasks; improved factual accuracy, reduced hallucination.
- **(5/6)**: model-agnostic; ~zero added cost (threshold comparator on existing logits). **Most directly portable mechanism to Rust.**

---

## 7. Live-Evo: online self-evolving memory

- **DOI**: 10.48550/arXiv.2602.02369 (arXiv:2602.02369v1). Zhang, Wu, Yu, Wu, Wang, 2 Feb 2026.
- **(1) Learned decisions**: **what/when to retrieve (experience-weighting), when to write back (selective verified), when to forget (down-weight), meta-rule synthesis (guideline)** — learned **online from feedback**, no weight fine-tuning. Two banks: Experience `E` + Meta-Guideline `M`.
  - **Score = `Weight * Sim(exp, query)`** (§3.1, verbatim).
- **(2) Loop** `{Retrieve, Compile, Act, Update}`:
  - Act: `r_q, tau_q = Act(q | g)` + **ContrastiveEval** — solve with guideline (`r_q^on`) and without (`r_q^off`); delta `r_q^on - r_q^off` = credit signal.
  - Update: adjust `W_{E_q}` by `(r_q^on - r_q^off)`; helpful -> reinforced, stale/misleading -> down-weighted and decayed.
  - Meta-guideline: on failure (`r_q^on - r_q^off <= 0`) add `Reflect(q,g_q,E_q)` to `M`.
  - **Selective write-back**: worst `rho`-fraction of tasks under memory-on -> summarize trajectory -> candidate `e_q^new`; commit only if `Eval(q,e_q^new) > r_q^on` (Verify Before Update).
- **(4) Results** (Prophet Arena, 10wk, 500 tasks; GPT-4.1-mini, temp 0.2, bad_case_percentile 0.3, min_brier_improvement 0.05, sim_threshold 0.5):
  - Brier Base 0.22 -> **0.14** (**-20.8%**); Market Return 1.24 -> **1.46** (+12.9%). Best weekly: Live-Evo 0.14 (MiroFlow 0.32, Qwen-DeepResearch 0.20, ReMem 0.16).
  - Xbench-DeepResearch Acc: **Live-Evo 0.46** vs Qwen 0.43, MiroFlow 0.45, ReMem 0.40.
  - Across backbones (Brier/return %-gain): GPT-4.1-mini 20.8/12.9; GPT-4.1 3.0/4.4; GPT-5-mini 4.5/1.6; Qwen3-8B 3.5/0.5 — **model-agnostic**.
  - Ablations: w/o weight-update 0.17 (+17% Brier/-8% return); w/o meta-guideline 0.16 (+10.9/-3.4); w/o compile 0.16 (+11.6/**-20.4 return**); w/o active-retrieve 0.17 (+15/-16.8). Removing guideline-compile most damaging.
- **(5/6)**: fully model-agnostic (LLM-prompt-driven). Inference cost = extra LLM calls (contrastive double-solve + reflection on failures) — notable token multiplier. Constants Rust-encodeable: sim_threshold 0.5, bad_case_percentile 0.3, min_brier_improvement 0.05.

---

## 8. NEMORI: self-organizing memory

- **DOI**: 10.48550/arXiv.2508.03341. Nan, Ma, Wu, Chen, 5 Aug 2025.
- **(1) Learned decisions**: segmentation granularity (episodic boundary) and what/when to distill into semantic — **LLM-driven procedural policies + prediction-gap comparison, NOT RL-trained**. Directly Rust-encodable.
  - Boundary: detector `f_th -> (b_boundary, c_boundary)`, `b in {True,False}`, `c in [0,1]`. Trigger when `T = (b_boundary AND c_boundary > sigma_boundary) OR (|M| >= beta_max)` (verbatim §3.1.1). `sigma_boundary` config, `beta_max` buffer cap — **concrete Rust heuristics**.
  - Representation: Episode Generator `e=(xi,zeta)=g_phi(M)` (title + narrative).
  - **Predict-Calibrate** (Free-energy): Predictor `e^hat = h_psi(xi, K_relevant)` forecasts from existing semantic memory; Distiller `K_new = r_omega(e^hat, M)` extracts the **prediction gap** (novel info existing knowledge failed to predict).
  - Unified retrieval `Retrieve(embed(xi+zeta), K, m, sigma_s)` — dense cosine + threshold.
- **(4)**: LoCoMo (10 dialogues, 24K avg tok, 1,540 Qs) + LongMemEval_S (500 convs, 105K avg). "Significantly outperforms prior SOTA... advantage pronounced in longer contexts". Baselines: Full Context, RAG-4096, LangMem, Zep, Mem0. Judge gpt-4o-mini. [Table numbers [UNVERIFIED]].
- **(5/6)**: model-agnostic (any LLM fills f/g/h/r); no fine-tuning. Cost: one boundary-detector call/message + per-episode generation/prediction/distill calls. Boundary+threshold logic directly portable to deterministic Rust heuristic or small classifier.

---

## 9. HAGE: RL-driven weighted-graph traversal

- **DOI**: 10.48550/arXiv.2605.09942. Jiang, Li, Li, Li, Li, 11 May 2026.
- **(1) Learned decisions**: **which graph edges/paths to traverse (graph-expand policy) given a query**. Four edge types `E_t = E_temp U E_sem U E_causal U E_ent`.
- **(2) MDP** (§3.4, verbatim):
  - State: node `n_i`, query `q_hat`, visited-mask `V_t`.
  - Action: neighbor `n_j in N(n_i)` per `pi_th(n_j|n_i,q)`.
  - Reward: `r_t = r_t^hit - lambda_step*r_t^step - lambda_timeout*r_t^timeout` — evidence-hit (accumulates per unique target) minus step + timeout penalties.
  - Traversal score: `S(n_j|n_i,q) = lambda*cos(v_j,q_hat) + (1-lambda)*w_ij(q)`; `w_ij(q) = softplus(MLP([q_hat; e~_ij]))` QueryRouter; `e~_ij = [e_ij; v_Tq; cos(q_hat,v_i); cos(q_hat,v_j)]`; edge feature `e_ij in R^4` (one-hot per relation or LLM-scored), refined by RL. lambda in [0,1].
- **(3) Training**: policy-gradient joint optimization of routing MLP + edge embeddings; needs node-level evidence targets only. An **LLM classifier identifies relational intent `T_q`**.
- **(4)**: improved long-horizon reasoning accuracy + favorable accuracy-efficiency trade-off; joint w/ regularization > routing-only and edge-only. [Exact numbers [UNVERIFIED]].
- **(5/6)**: tied to trained routing MLP + edge embeddings (not model-agnostic). Inference = one LLM intent classification + greedy/beam traversal (hop-bounded). `S = lambda*cos+(1-lambda)*w` form + softplus MLP compact enough to reproduce on-device in Rust.

---

## 10. Meta-cognitive memory policy optimization

- **DOI**: 10.48550/arXiv.2605.30159. May 2026.
- **(1)**: where intermediate memory quality degrades during recursive summarization — write/consolidate policy guided by belief clarity, not pure outcome.
- **(2)**: **Belief Entropy** — self-supervised proxy for "how uncertain the model remains about the latent task state given its current memory." Outcome-based RL "fails to localize where intermediate memory quality degrades". [Full MDP/reward [UNVERIFIED]].
- **(5/6)**: [Detail [UNVERIFIED].] The belief-entropy-as-proxy idea could be approximated deterministically in Rust (e.g., summary-vs-full-context semantic divergence, or per-step retrieval hit-rate).

---

## 11. Named related works (context, not fully read)

- **MemAgent** (Yu et al., arXiv:2507.02259, DOI 10.48550/arXiv.2507.02259): RL-trained long-context agent, iteratively processes all chunks under task instruction; RL-MemAgent-14B/7B; 8K-trained -> 3.5M-token QA <5% loss; 512K RULER >95%; linear time. Policy tied to fine-tuned 7B/14B. [Per project page/abstract].
- **MEM1** (arXiv:2506.15841): RL write+retrieve, flat memory, LoCoMo <26k.
- **MemQ** (arXiv:2605.08374): Q-learning retrieval over provenance DAGs; locality + Q-guided selection. Abstract-level.
- **CoEvo-Mem** (arXiv:2608.01739): co-evolving retrieval policy + memory bank.
- **Evo-Memory** (arXiv:2511.20857): benchmark; learning-policy taxonomy prompting/fine-tuning/RL.
- **Agentic Memory** (arXiv:2601.01885): trains store/retrieve/update/summarize/discard as tools via 3-stage RL, step-wise GRPO.
- **Mem0/MemGPT/Zep/A-MEM/LangMem**: hand-written prompt-based memory managers — these are the *baselines* the learned policies beat.

---

## 12. What a Rust system can borrow WITHOUT training an LLM

### A. Directly reproducible as deterministic / heuristic Rust code or a small local model call

1. **Retrieve-or-not via token confidence (FLARE)**: threshold min-token logit/prob of predicted next sentence. Rust: during compose, `if min_t P(tok_t) < theta: mark for retrieval`. theta=0 -> never, 1 -> always; tune ~0.5. ~zero cost. — DO borrow.
2. **Query-complexity routing (Adaptive-RAG)**: small classifier -> {no-retrieval, single-step, iterative}. Distill to tiny on-device classifier, or keyword/length heuristic at bootstrap. Saves unnecessary retrieval. — Borrow as scope-filter/retrieve-or-not primitive.
3. **CRAG post-retrieval evaluator**: small classifier `{Correct,Incorrect,Ambiguous}` -> refine/web-fallback. Reproduce with small local classifier or embedder-threshold: `if top-1 dense < tau: fallback`. Decompose-then-recompose = deterministic text processing. — Borrow 3-branch control flow.
4. **Live-Evo experience weighting (highest-value borrow)**: `Score = Weight x Sim(exp,query)`, `w_e += eta*(r_relevant - baseline)` / decay stale. Rust: per-experience float weight, bandit/exp3-style scalar policy. Constants: sim_threshold 0.5, bad_case_percentile 0.3, min_brier_improvement 0.05. Verify-Before-Write-back = deterministic gate. Fully Rust-implementable. — Borrow.
5. **NEMORI episodic boundary detection**: `T = (bbox AND cboundary > sigma_boundary) OR (|M| >= beta_max)` — pure threshold rule. Buffer messages; small LLM (or lexical-similarity spike) for boundary confidence; flush when > sigma or buffer full. sigma_boundary, beta_max config. — Borrow (rule model-agnostic). Also borrow **Predict-Calibrate** as a consolidation gate: predict episode content from current semantic memory, distill only the prediction gap — one local LLM call per episode, no training.
6. **Mem-a compression objective**: `r_3 = 1 - l_m/l_c` — encode directly as a token-budget constraint in consolidate/forget: prefer compact memory relative to ingested volume. Use as cost-shaping, not RL reward.
7. **HAGE additive traversal score** (graph-expand): `S = lambda*cos(v_j,q) + (1-lambda)*w_ij(q)` — semantic term plain cosine; structural term = small on-device MLP/attention (softplus, positive). Even untrained, initializing w from per-relation type priors reproduces the mechanism; hop budget H_max + lambda config. — Borrow the additive blend + hop budget; train tiny MLP later.
8. **Self-RAG reflection threshold**: adaptive retrieval via `P(Retrieve=yes) > threshold`; segment scoring = weighted sum of critique probs. Without trained critic, approximate: (a) trigger retrieval on generator uncertainty / relevance score; (b) post-compose **IsSup-style** support check (string/NER overlap or small NLI) as a grounding/provenance gate. Threshold mechanism transfers, not the trained critic.
9. **Retrieval distillation (Memory-R1)**: from top-60 RAG candidates, filter to useful subset before answering. Rust: rank by hybrid score, keep above relevance threshold (or top-k with dedup/coverage pass). Pure rerank/scope-filter. — Borrow; +9% F1 from distillation, no training.

### B. Mechanisms that REQUIRE RL fine-tuning (skip or defer in myelin v1)

- **Mem-a / Memory-R1 / MemAgent write+operation policies** — ADD/UPDATE/DELETE/NOOP selection is a tool-calling policy learned via PPO/GRPO, tied to a fine-tuned 4B/8B/14B backbone. Requires GPU RL (32xH100 3d Mem-a; 4xH100 Memory-R1, 152 examples). Seed equivalent deterministically: **merge-not-replace** (when new fact's embedding/NER overlap with existing memory > tau -> UPDATE not insert-delete); **semantic-validity gate** (Mem-a r_4) via small local verifier / self-consistency.
- **HAGE trained edge embeddings + routing MLP**: score *form* borrowable, trained weights need data + reward. Defer joint RL; cold-start with relation-type priors viable.
- **Self-RAG trained critic + generator reflection tokens**: needs fine-tuning a 7B/13B. Defer; approximate control side with thresholds (A.8).
- **MemQ Q-learning over provenance DAGs**: heavy; defer.
- **Belief-Entropy meta-cognitive proxy**: needs calibrated uncertainty; approximate in Rust with retrieval hit-rate / summary-divergence signals.

---

## 13. High-leverage findings for the Rust implementation

1. **Retrieval-weights-as-scalars (Live-Evo)** is the cheapest learned policy: `Score = Weight*Sim` with feedback-driven `w_e` updates encodes write/what-to-retrieve/forget *without any model training*, and is fully Rust-encodable (bandit/exp3). Constants: sim 0.5, worst-case 0.3, min-improvement 0.05.
2. **The learned value in Mem-a / Memory-R1 largely reduces to a few deterministic rules**: (a) merge-not-replace for conflicting facts; (b) compact-memory objective `r_3 = 1 - l_m/l_c`; (c) pre-compose retrieval distillation (filter 60 -> useful subset, +9% F1). Captures most SOTA gains with no RL.
3. **Adaptive retrieval policies (Self-RAG/Adaptive-RAG/FLARE/CRAG) are control policies separable from the generator**: thresholds (FLARE), complexity router (Adaptive-RAG), lightweight relevance evaluator (CRAG), retrieve-prob threshold (Self-RAG) — standalone small models or pure heuristics. Gives the retrieve stage its learned gate without fine-tuning the base model.
4. **Graph-expand policy (HAGE) is decomposable**: semantic cosine (free) + positive structural weight `w = softplus(MLP)` (tiny, optional training). Additive lambda-blend + hop budget transfer directly; relation-type priors give a working cold start.
5. **NEMORI + Live-Evo deliver learning through LLM-dictated + feedback-grounded rules, not gradients** — episodic boundary `(T = b AND c > sigma OR |M| >= beta_max)`, predict-calibrate distillation, verified write-back. Template for myelin consolidate/forget stages as Rust-native policy modules.

---

## Papers actually read (DOI list)
1. 10.48550/arXiv.2509.25911 (Mem-a) — full
2. 10.48550/arXiv.2508.19828 (Memory-R1) — full
3. 10.48550/arXiv.2310.11511 (Self-RAG) — full (methods+experiments)
4. 10.48550/arXiv.2403.14403 (Adaptive-RAG) — abstract + corroboration
5. 10.48550/arXiv.2401.15884 (CRAG) — corroboration
6. 10.48550/arXiv.2305.06983 (FLARE) — corroboration
7. 10.48550/arXiv.2602.02369 (Live-Evo) — full (method+results+ablation)
8. 10.48550/arXiv.2508.03341 (NEMORI) — full method (results table beyond fetched range)
9. 10.48550/arXiv.2605.09942 (HAGE) — full (method+MDP+reward)
10. 10.48550/arXiv.2605.30159 (meta-cognitive) — abstract only

## 5 highest-leverage findings for a Rust implementation
1. **Live-Evo Score=Weight*Sim + feedback weight-update** — the only fully trainable-without-LLMs learned policy and highest-value borrow (covers retrieve ranking, forget/decay, selective write-back).
2. **Merge-not-replace (UPDATE)** from Memory-R1 + `r_3 = 1 - l_m/l_c` compression objective from Mem-a recover most consolidation value deterministically.
3. **Retrieval distillation** (Memory-R1 filter-60->useful before compose, +9% F1) — pure rerank/scope-filter, no training.
4. **All adaptive-retrieval control policies (FLARE threshold, Adaptive-RAG router, CRAG 3-branch evaluator, Self-RAG retrieve-prob threshold) are generator-agnostic small models/heuristics** — Rust can host them as standalone local-model or threshold calls.
5. **HAGE graph traversal decomposes** into free cosine term + optionally-trainable tiny structural MLP, with hop budget — clean graph-expand primitive with cold-start from relation-type priors.
