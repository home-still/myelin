# 05 — Memory as an Attack Surface & Memory Governance

*Evidence pack for the `myelin` design doc. Every factual claim carries a DOI or doc_id; quoted numbers carry page/line or section pointers where available. `[UNVERIFIED]` marks numbers I could not confirm in a primary source.*

**Scope**: memory poisoning/injection (MINJA and analogues), indirect prompt injection into retrieved memory, RAG knowledge-base poisoning (PoisonedRAG and family), agent-memory backdoors, access control for shared memory (Collaborative Memory), privacy/PII in memory, machine unlearning / right-to-be-forgotten for memory stores, and memory-governance frameworks (SSGM, MemOS, lifecycle policy).

---

## 1. Threat Model — attacker capability and goal

### 1.1 MINJA (query-only memory injection) — `10.48550/arXiv.2503.03704`

**Attacker capability** (MINJA Sec 3, threat model): the attacker is a *regular user* who (1) cannot directly manipulate any part of the agent beyond what query-output interaction exposes — including the agent's responses and the memory bank — and (2) cannot modify or interfere with victim users' queries. The one systemic assumption is that the agent runs on a **shared memory bank** adopted across all user queries (common in deployed frameworks: ChatGPT "Improve the model for everyone", Waymo, Alexa, RAP, EHRAgent). This is the weakest-privilege attacker in this literature: prior work (ICL backdoored demonstrations; AgentPoison) assumes the attacker can *directly inject records into* the memory bank; MINJA removes that.

**Attacker goal**: for a prescribed *victim term* `v` (e.g., patient ID `027-22704`), cause the agent's reasoning `R_{q_v}` for any victim query `q_v` to instead be the reasoning `R_{q_t}` associated with a *target query* `q_t` in which `v` is replaced by a *target term* `t` (e.g., patient ID `015-91239`). Harm: an EHR agent returns the wrong patient's record/prescription (misdiagnosis, adverse drug outcomes); an autonomous-driving agent executes "stop" at extreme settings from poisoned demonstration records (Sec 1, fatal accident).

Record-kind mapping (shared vocab): the poisoned records are `episodic` (past (query, reasoning) interaction traces) injected into `working`/few-shot demonstration context. Stage mapping: attack targets `ingest → consolidate` (persist malicious records) and exploits `retrieve → compose`.

**Attack surface**: the user-input channel and any memory-API write path (`10.48550_arxiv.2601.05504`, Sec 3 threat model): "the primary attack surface is the user input channel … Another possible surface is through memory APIs that write to persistent memory."

### 1.2 RAG knowledge-base poisoning (PoisonedRAG) — `10.48550/arxiv.2402.07867` (arXiv, external; not in local corpus)

**Attacker capability** (PoisonedRAG abstract): can inject *a few malicious texts into the knowledge database* of a RAG system — the retrieval corpus/chunk store, **not** the LLM, no model access. This is the direct-write variant MINJA deliberately avoids and the strongest model of "memory-store poisoning," since RAG chunks are exactly `semantic` memory content.

**Attacker goal**: for an attacker-chosen *target question*, induce the LLM to emit an attacker-chosen *target answer*. Formulated as an optimization problem solved under black-box and white-box background knowledge.

**Measured success** (see Sec 3).

### 1.3 Indirect prompt injection into retrieved content (generalized to memory)

- `10.48550/arXiv.2507.20526` (AI-agent security competition, coordinated by UK/US AI Security Institutes): indirect prompt injections — hidden malicious instructions embedded in *untrusted data* (e.g., crafted log entries; analogously a retrieved memory chunk) — were the dominant high-success vector. Red-team ASR table: indirect injections achieved **27.1% overall policy-violation ASR vs 5.7% direct**, and **36.8% for "Prohibited Action"** (highest single cell). All tested models suffered repeated successful attacks (100% at behavior level).
- `10.48550_arxiv.2603.11768` (SSGM, Sec 1): evolving, self-refining memory *internalizes* indirect-injected instructions as valid knowledge, producing **cumulative and persistent** error rather than static-RAG per-retrieval isolation. Three compounding failure points: memory poisoning at ingestion → semantic drift at consolidation → conflict/hallucination at retrieval (Fig. 1).

### 1.4 Agent-memory backdoors (trigger in input → adversarial target in output)

**AgentPoison** — cited as [2] in MINJA and the EHR-defense paper; NeurIPS 2024 (`Chen et al., Adv. NeurIPS 37:130185–130213, 2024`). Poisoned records carry a *trigger* in the agent input and an *adversarial target* in the output; the trigger is **optimized** (unlike MINJA's victim-term trigger). Requires direct injection into the memory bank or knowledge base (stronger privilege than MINJA).
**BadRAG** — cited [14] in the EHR-defense paper (`Jiaqi Xue et al., 2024`): identifies RAG vulnerabilities; present in the local corpus only as a citation. `[UNVERIFIED: BadRAG's quantitative results not found — searched "PoisonedRAG BadRAG retrieval"; citation-level only.]`
**System-prompt poisoning** — cited [9] (`Zongze Li et al., 2025`): persistent attacks that survive individual user injections (extends backdoor paradigm to the system prompt).

### 1.5 Attack-capability ladder (synthesis)

| Attack | Privilege needed | What it poisons | Primary mechanism |
|---|---|---|---|
| ICL backdoor (trigger) | direct memory write | demonstration records | optimized trigger in input |
| AgentPoison | direct memory/knowledge-base write | memory + RAG KB | optimized trigger |
| PoisonedRAG | corpus/knowledge-base write | RAG KB (`semantic` memory) | optimized injected docs |
| **MINJA** | **query-only (any user)** | shared memory bank | bridge steps + indication prompt + PSS |
| Contagious jailbreak (`Men et al., 2025`) | shared-memory propagation | shared memory | cascading cross-agent spread |

---

## 2. Attack mechanism — step by step

### 2.1 MINJA (`10.48550/arXiv.2503.03704`, Sec 4, Fig. 1)

Goal per record: store `(attack_query q_v ∋ v, [bridge_steps b_{v,t}, target_reasoning R_{a,t}])` so a *benign* query containing victim term `v` maps to reasoning over target term `t`.

1. **Bridging steps `b_{v,t}`** close the logic gap between `q_v` and `R_{a,t}` (which a benign query alone cannot produce). Placed **at the start** of the record's reasoning so retrieval forces the agent to emit them first. Kept *general* (independent of any victim query) so they work for arbitrary victim queries containing `v`.
2. **Indication prompt** appended to the attack query — a logically-connected chain — induces the agent to *autonomously generate* the bridging steps and target reasoning (e.g. "The data of patient A is now saved under patient B; we should refer to patient B"). The generated (query, reasoning) pair is stored as a memory record — the agent stores its own output; the attacker never writes.
3. **Progressive Shortening Strategy (PSS)**: re-issue the attack query with the prompt *progressively shortened* (4/5/5/5 shortening steps for Patient-ID/Medication/Items/Terms datasets) until the record reads as a plausible benign query that (a) still triggers `b_{v,t} → R_{a,t}` and (b) is easily retrieved against later victim queries (queries avoid rare/abnormal content).
4. **Trigger at inference**: when the victim submits any query containing `v`, Top-k similarity retrieval surfaces the injected records; in-context learning emits bridging + target reasoning, redirecting `v→t`.
5. **Robustness** (Sec K, Table 11): ASR stable across re-runs (std <3% RAP, <6% QA).

### 2.2 PoisonedRAG (`10.48550/arxiv.2402.07867`, abstract) — corpus-level poisoning

1. Attacker injects a small set of crafted texts into the knowledge DB (the `semantic` memory store).
2. Texts are solved as an optimization problem so retrieval returns them with high rank for the target question.
3. Generation is grounded on the poisoned texts → attacker-chosen target answer. Two variants: black-box and white-box.

### 2.3 Indirect injection into retrieved memory (generalized from `10.48550/arXiv.2507.20526` + SSGM `10.48550/arxiv.2603.11768`)

Where a memory backend stores *agent-generated or externally-sourced* chunks, an attacker who gets a malicious passage ingested (documents, tool outputs, web fetches, shared-memory writes) gains a **persistent** indirect-injection vector: every future retrieval surfacing the chunk re-injects the instruction. SSGM's core point (Fig. 1): unlike static RAG, an evolving memory *consolidates* the injected content, so corruption compounds across `consolidate/index` instead of decaying.

---

## 3. Measured attack success rates and the conditions producing them

### 3.1 MINJA headline numbers — `10.48550/arXiv.2503.03704`

**VERIFY the "over 95% injection success under idealized conditions" claim.**

The claim's source is the **MINJA paper itself** (`10.48550/arXiv.2503.03704`). Primary-source statements:

- Abstract/Contributions: "Across all settings, MINJA achieves a high average success rate of **98.2%** for injecting malicious records into the memory, and a high average attack success rate of **76.8%** in eliciting the malicious reasoning steps."
- Sec 5.2 (Main Results): "MINJA achieves **ISRs higher than 90%** for most agent/victim-target-pair/dataset configurations, while ASR exceeds **70%** in half of the cases and surpasses **90%** for GPT-4-based EHR on eICU and GPT-4o-based RAP on Webshop."
- Sec J (Table 10): DeepSeek-R1 hits **ISR 100% / ASR 90-100%** on QA Agent, matching GPT-4. Llama-2-7B is an outlier (task accuracy ~17-19% < random 25%) — flagged as *insufficient task utility*, making attack analysis meaningless; attack success is contingent on the host model being capable enough to follow reasoning.
- **Idealized conditions** (what the "over 95%" figure refers to), Sec 5.2: EHRAgent starts with only **4 benign records as demonstrations** (RAP/MMLU start *empty*); victim-target pairs reserve 50 benign queries (EHR/RAP) or 30 (QA); attack queries **randomly shuffled with benign queries**. ISR/ASR definitions (Sec 5.1): ISR = fraction of attack queries producing targeted reasoning; ASR = fraction of separate victim-test queries (10 MMLU / 30 others) whose responses contain the targeted reasoning.
- Robustness ablation (Sec L, Table 12): **removing PSS drops ISR 93.3% → 80% (fewer steps) → 75.6% (no PSS)**; ASR 87.8 → 87.8 → 82.2. PSS is a necessary component of the >90% regime.

**The EHR-defense paper (`10.48550_arxiv.2601.05504`) independently restates the claim** as its motivation: "the work done in *A Practical Memory Injection Attack against LLM Agents* by [4] claims the attack achieves **over 95% Injection Success Rate (ISR) across all LLM-based agents and datasets, and over 70% Attack Success Rate (ASR) on most datasets**," characterizing [4]'s evaluation as "largely idealized or controlled settings with minimal initial memory, limited attack queries, and fixed retrieval parameters."

**Condition qualifier (important for design)**: the >95% ISR is *specific to idealized/empty-initial-memory settings* — it degrades sharply when the memory bank already holds legitimate memories (Sec 3.2). Treat "98.2% ISR / 76.8% ASR" as the upper bound under empty-or-near-empty shared memory with PSS, not a production-baseline guarantee.

### 3.2 Realistic-condition results — `10.48550_arxiv.2601.05504` (EHR agents, MIMIC-III; GPT-4o-mini, Llama-3.1-8B-Instruct, Gemini-2.0-Flash)

Setup (Sec 5.2): Levenshtein-distance Top-N relevance retrieval; baseline = empty initial memory, up to **3 relevant memories** appended, 2 indication prompts + PSS.

| Setting | Model | ISR | ASR |
|---|---|---|---|
| Baseline (empty init. memory) | GPT-4o-mini | 100% | 62% |
| Baseline (empty init. memory) | Llama-3.1-8B-Instruct | 100% | 52.94% |
| **6 pre-existing legit. memories** (2 relevant to victim ID) | GPT-4o-mini | 26.67% | **6.67%** |
| **6 pre-existing legit. memories** | Llama-3.1-8B-Instruct | 99.95% | **0%** |

- Realistic case confirms **Hypothesis 1**: pre-existing correct memory collapses ISR/ASR. For Llama, ISR stays ~100% (poison *is stored*) but ASR → 0% because few-shot correct examples anchor behavior (persistent memory as defense).
- **Number of indication prompts (2→4)** has *no significant effect* — **disproves Hypothesis 2** (Levenshtein retrieval surfaces benign victim-ID queries lacking the target ID).
- **Number of relevant memories retrieved (3/5/10)**:

| Retr. memories | GPT-4o-mini ASR / ISR | Llama-3.1 ASR / ISR |
|---|---|---|
| 3 | 6% / 26% | 0% / 99% |
| 5 | 20% / 50% | 13.33% / 93.94% |
| 10 | **38% / 100%** | 27.27% / 97.87% |

Confirmed Hypothesis 3: higher retrieval count raises ASR. **Author bottom line**: in pre-populated realistic memory, query-only attack best-case success is ~38% (GPT-4o-mini) / ~28% (Llama); real healthcare banks with millions of records make the attack impractically expensive (spam gets flagged; raising k costs context/tokens).

### 3.3 PoisonedRAG — `10.48550/arxiv.2402.07867`

**>90% ASR by injecting 5 docs into a multi-million-text DB** ("achieve a **90% attack success rate when injecting five malicious texts for each target question** into a knowledge database with millions of texts"). Independently confirmed by survey `10.3390/info17010054` ("just five carefully crafted documents can manipulate AI responses **90%** of the time through RAG poisoning"). Corpus-level analogue of the memory-write attack; assumes the stronger write-to-corpus privilege.

### 3.4 Indirect prompt injection — `10.48550/arXiv.2507.20526`

Red-team table (policy-violation ASR): indirect **27.1%** vs direct **5.7%**; indirect "Prohibited Action" **36.8%**; indirect "Confidentiality Breaches" **29.8%**; every model had repeated (100% at-behavior-level) successes. Context: a large public competition with expert/human adjudication — a real-world proxy, not a synthetic lab setting.

---

## 4. Proposed defenses and measured effectiveness/cost

### 4.1 MINJA paper's suggested directions (`10.48550/arXiv.2503.03704`, Impact Statement)

No measured numbers; qualitative: **memory isolation, strong user authentication, secure memory management (access control), advanced prompt filtering.** MINJA Sec I (retrieval-filtering robustness) notes embedding-level sanitization and prompt-level defenses are both limited: semantic overlap makes benign-vs-malicious records hard to distinguish; LLM filtering either fails to generalize (targeted prompts) or has high false positives (general prompts).

### 4.2 EHR-defense paper (`10.48550_arxiv.2601.05504`) — two empirically-tested defenses

**D1 — Input/Output Moderation** (Sec 6.1): two-stage gate around model execution.
- Input: fast static heuristics + keyword matching ("Knowledge:", "refer", "ignore previous", suspicious control chars); on high risk → SKIP (no model call). Optional semantic LLM classifier for subtler cases.
- Output: inspect generated code for dangerous ops (exec/eval, subprocess, network/file I/O); optional LLM relevance check; **composite trust score** `t∈[0,1]` from multiple orthogonal signals (static patterns, dynamic answer verification, code-safety heuristics, optional sandbox re-execution). Memory append only if `t` > conservative threshold.
- **Measured** (Sec 8.1, GPT-4o-mini): **all 23 candidate memories rejected** (0 accepted), mean trust 0.28 (sd 0.10, max 0.4); of 101 poison queries, 2 blocked at memory layer (1.98%), **0 leaked**. **Failure mode**: too conservative → rejects *benign* entries → effectively **empty memory** (utility loss).

**D2 — Memory Sanitization with trust-aware retrieval** (Sec 6.2):
- *Append-time*: trust score from guard output assessment + answer verification against labels + optional sandbox re-execution + static pattern filters for poisoning templates ("Refer X to Y", "use Y instead"). Low-trust not stored.
- *Retrieval-time*: **temporal decay** on base trust; entries below threshold excluded from few-shot selection; pattern filters block templates; rank by (textual-similarity, trust).
- **Measured** (Sec 8.2, Gemini-2.0-Flash): 151 candidates → 82 accepted at **trust 1.0**, 69 rejected (mean 0.35); **54 of 82 accepted were malicious** ("Knowledge:" instructions), 28 benign. Retrieval filtering at threshold 0.5 filtered **none** (all held 1.0). **Critical failure**: trust defense became a "confidence filter, not a security filter" — sophisticated prompts ("justification" clauses) socially engineer high confidence, so well-phrased attacks pass. Recommend **decoupling trust from safety** via external "verification-of-effect" checks against immutable ground truth (e.g., validating patient-ID redirects before persisting).

**Design lesson (both defense papers):** naive trust/threshold filtering has a security/utility tradeoff — too strict → empty memory (no poisoning, no value); lenient → model overconfidence admits confident malice.

### 4.3 RAG-robustness defenses (external; `[UNVERIFIED in local corpus]`)

- **RobustRAG** (`10.48550/arxiv.2405.15556`, external): *isolate-then-aggregate* — partition retrieved passages into disjoint groups, generate per-group responses, securely aggregate (keyword- or decoding-based) → **certifiable** lower bounds on response quality even against an adaptive attacker injecting a bounded number of malicious passages. Directly applicable to a `retrieve` step treating each chunk as an independent evidence group.
- **RAGForensics** (`10.1145/3696410.3714756`, external): iterative *traceback* — retrieve a subset, LLM-guided detection of poisoned texts in the DB → identifies *which* stored texts caused an attack (audit/remediation).
- **Phantom** (`10.48550/arxiv.2405.20485`, external): single injected doc, two-stage optimization achieving trigger-targeted retrieval + adversarial generation objectives; transfers to GPT-3.5/4. Motivates *write quarantine + provenance* at ingest.

### 4.4 Governance frameworks

- **SSGM** (`10.48550/arxiv.2603.11768`, Sec 6.1-6.2): four principles, formal equations (Sec 5.2 below): P1 pre-consolidation **Write Validation Gate** (NLI contradiction check `ΔM ∧ M_core ⊨ ⊥ → reject`); P2 **Read Filtering Gate** — cryptographic provenance `σ(μ)` + cognitive decay (`w(Δτ)=exp(−(Δτ/η)^α)`, Weibull), prune below freshness threshold; P3 **access-scoped retrieval** (ABAC identity predicates injected into the query layer, citing `10.48550/arXiv.2505.18279`); P4 **reversible reconciliation** — mutable Active Graph + append-only immutable Episodic Ledger, async replay corrects drift. **Theorem 1 (bounded drift)**: naive drift grows `O(T·ε_step)`; with reconciliation every `N` steps, bounded `O(N·ε_step)` regardless of horizon. **Stated trade-offs**: latency–safety (governance adds "System 2" verification latency → *asynchronous governance*), stability–plasticity (over-strict rejection ossifies knowledge / blocks legitimate updates like address changes), graph scalability.
- **MemOS** (`10.48550/arXiv.2507.03724`, external): OS-like memory abstraction with a dedicated **Memory Governance** module — ACLs, TTL, conditional activation per unit; five-state lifecycle `Generated → Activated → Merged → Archived → Expired`; versioned MemVault store; redaction + watermarking before cross-institution sharing. Maps directly to `myelin`'s `forget`/lifecycle stage.
- **MOOM** (`10.48550/arxiv.2509.11860`) — forgetting-mechanism ablation (Sec D): optimal `α=0.1` (retrieval-induced forgetting weight), `β=0.9` (time-induced), `γ=1`, `k=9` recalled memories. Retrieval-induced forgetting outweighs time-induced; a *forgetting policy* is a tunable model, not a fixed schedule.

### 4.5 Privacy in memory

- `10.48550/arxiv.2406.01171` & `10.18653/v1/2024.findings-emnlp.969` (persona surveys, Sec 5.5 Privacy): membership-inference on personalized models can **leak personal information**; persona assignment aids jailbreaking and introduces personalization bias. Privacy mitigations cited: `ProPILE` (Kim et al.) detects privacy leaks.
- `10.48550/arxiv.2112.04359` (Weidinger et al., Sec 2.2.4): LMs as "storage devices" leak training data; **privacy leaks** (model memorizes) vs **inference of private traits** differ in root cause → distinct mitigations (differential privacy for leaks; capability restriction for inference).
- **SSGM** (`10.48550/arxiv.2603.11768`, Sec 1): **topology-induced knowledge leakage** — fully-connected memory networks let sensitive contexts solidify into long-term storage and leak across personas/tenants — motivates access-scoped retrieval (P3). "Unveiling privacy risks in LLM agent memory" (Wang et al., ACL 2025) is a key primary source on agent-memory privacy.
- GDPR framing for memory stores: `10.1002/cpe.6426` (Sec 189-197): storage limitation, data minimization, right to be forgotten, authentication, and the impossibility of removing data once baked into a trained model. Relevant to `myelin`'s PII-in-memory + retention design.

### 4.6 Machine unlearning / right-to-be-forgotten applied to memory stores

Local-corpus findings are **thin on unlearning applied to memory stores** — a genuine gap to flag. What exists:
- SSGM conclusions (`10.48550/arxiv.2603.11768`, Sec 7): calls for "machine unlearning protocols to surgically remove toxic memories" as open research — i.e., unlearning-for-memory is *not yet a solved, benchmarked problem*.
- `10.48550/arxiv.2404.18231` (Sec 7.3): unlearning strategies (Neel et al. 2021; Pawelczyk et al. 2023) proposed for *out-of-scope knowledge* in role-playing agents — but described as "underexplored."

`[UNVERIFIED/absent]` Searched "machine unlearning", "right to be forgotten", "data removal" — top hits were generic ML-testing, representation-engineering memorization control, and GDPR chatbot prose; **no paper in the corpus quantifies unlearning applied to an agent memory store**. For `myelin`, treat deletion/unlearning as an **implementable but unevidenced-in-this-corpus** control grounded on (a) GDPR storage-limitation/right-to-be-forgotten `10.1002/cpe.6426`; (b) SSGM/ledger-based rollback `10.48550/arxiv.2603.11768` (P4 dual-store gives a natural deletion+rollback substrate); (c) PoisonedRAG's lesson that removing the *source text* is the only defense at the store level (Sec 3.3). Note the SSGM dual-store (active graph + immutable ledger) is exactly the primitive a "forget" stage needs: delete from the mutable graph, keep the append-only ledger for audit/replay.

---

## 5. Access-control formalism — Collaborative Memory (`10.48550/arXiv.2505.18279`) — implementable detail

Purpose: the paper's central formal contribution and the most directly-importable access-control model for a shared-memory backend.

### 5.1 Dynamic bipartite access graphs (Eq. 1-2)

Let `U`, `A`, `R` = sets of users, LLM agents, resources (tools/APIs/data sources). At timestep `t`:

```
G_UA(t) ⊆ U × A        edge (u_i, a_j) = user u_i may invoke agent a_j at t
G_AR(t) ⊆ A × R        edge (a_j, r_k) = agent a_j may access resource r_k
A(u,t) := { a | (u,a) ∈ G_UA(t) }        agents user u can invoke at t
R(a,t) := { r | (a,r) ∈ G_AR(t) }        resources agent a can access at t
```

Graphs **evolve over time** (grant/revoke edges: on-boarding, role change, policy change). Implementation note: `G_UA` and `G_AR` are the two ACL relations a memory backend must persist per-request (not a single static role table) to support revocation.

### 5.2 Memory tiers + immutable provenance (Sec 3.2)

Every fragment `m ∈ ℳ` carries **immutable provenance**: creation time `𝒯(m)`, contributing user `𝒰(m)`, contributing agents `𝒜(m)`, resources accessed `ℛ(m)`. Partition `ℳ = ℳ^private ∪ ℳ^shared`:
- `ℳ^private(u,t)` = everything from user `u`'s past interactions up to `t`.
- `ℳ^shared(a,t)` = fragments agent `a` generated for any user up to `t`.

**Effective access set** for agent `a` serving user `u` at `t` (Eq. 3):

```
ℳ(u,a,t) := { m ∈ ℳ | 𝒜(m) ⊆ 𝒜(u,t)  ∧  ℛ(m) ⊆ ℛ(a,t) }
```

This single predicate is the **permission check to implement**: a fragment is readable only if the *set of its contributing agents* is a subset of what user `u` may invoke, AND the *set of resources it accessed* is a subset of what agent `a` may access. Note the asymmetry: user-level check on agents, agent-level check on resources. Cross-user and cross-agent sharing flow through these two conjunctions.

### 5.3 Read policy → filtered/view-derived memory (Sec 3.3)

User submits query `q`; eligible agents `𝒜(u,t,q) ⊆ 𝒜(u,t)`. Agent `a` responds using **only the filtered view** built by the read policy `π^{read}_{u,a,t}` over the effective set (Eq. 4):

```
y_{u,a,t} = a(q, π^{read}_{u,a,t}(ℳ(u,a,t)), ℛ(a,t))
```

Read policy can: limit the number of fragments retrieved, filter by keywords, and (per Sec B.1 retrieval) return `top-k_user` fragments from the user tier + `top-k_cross` from the shared tier that **satisfy the provenance constraint**, by cosine similarity of subquery embedding vs fragment keys. Views are *projections*, not copies — "project existing memory fragments into filtered transformed views."

### 5.4 Write policies → retention & sharing (Sec 3.4, Eq. 5-6)

Two write policies after the agent produces `y_{u,a,t}`:

```
π^{write/private}_{u,a,t} : y_{u,a,t}, ℳ^private(u,t) ↦ ℳ^private(u,t+1)
π^{write/shared}_{u,a,t}  : y_{u,a,t}, ℳ^shared(u,t)  ↦ ℳ^shared(u,t+1)
```

They may **anonymize, redact, or block content**; an implementation "simply embed the output and insert the resulting vector into a storage table," or apply richer structures. Separating the two lets a user customize confidentiality per tier.

### 5.5 Policy scoping & granularity (Sec 3.4)

`π^{*}_{u,a,t} ∈ { π^{*}_global (all), π^{*}_u (per user), π^{*}_a (per agent), π^{*}_t (over time) }` — read AND write policies independently tunable at each scope. **Provable adherence**: enables "provable adherence to asymmetric, time-varying policies and full auditability of memory operations."

### 5.6 Measured instantiation (dynamic scenario, Sec E + Table 19)

- Access graph evolved via Bernoulli granting (`p=0.2`, 5 edges/step, `t_0..t_4`) then revoking (`t_5..t_8`); agent→resource held 1:1; embedding `text-embedding-3-large`; coordinator/aggregator/agents `gpt-4o`; retrieval `k_user=10`, `k_cross=10`, similarity threshold 0.1.
- Results (Table 19): accuracy climbs with granted edges (0.27@t0 → 0.61@t4) and falls on revocation (0.37@t8); cross-memory 11.27→13.10 (grant) → 6.75 (revoke). **Enforcement evidence**: Fig. 11 shows agents/resources used *only* inside granted access — the access graph provably gates both invocation and resource use, and revoking edges immediately reduces reachable memory (a deletion-by-revocation primitive).
- Private/shared reuse examples (Table 18): private reused same-user (Query #18→#40); shared reused cross-user (user_5 → user_2) **only** because both had access to the chemistry agent — demonstrating the provenance/`ℳ(u,a,t)` gate in action.

### 5.7 Relation to existing standards (Sec 2)

Inherits ABAC policy modularity (NIST SP800-162) but adds explicit *time-varying bipartite graphs* between users, agents, resources — the key extension over static RBAC/ABAC for a memory backend.

---

## 6. What a memory-security eval looks like — attack corpora and metrics

### 6.1 Metrics (MINJA `10.48550/arXiv.2503.03704` Sec 5.1 — primary definitions)

- **Inject Success Rate (ISR)**: ratio of attack queries whose responses contain the *targeted/reinforcing reasoning steps* (i.e., the malicious record was actually generated and stored).
- **Attack Success Rate (ASR)**: proportion of separate victim-test queries (10 MMLU / 30 others) whose responses *contain the target reasoning steps*, regardless of prior injection success — isolates downstream poisoning.
- **Utility Drop (UD)**: task-performance delta on benign queries between poisoned and unpoisoned memory. MINJA observed UD mostly 0 to −20%.
- **Stepwise ISR** (Sec M): ISR per progressive-shortening iteration.
- **ASR stability** (Sec K): std across re-runs (<3% RAP, <6% QA).

### 6.2 Defense-eval metrics (`10.48550_arxiv.2601.05504` Sec 6-7)

- **Trust-score distribution**: candidate count, accepted vs rejected, mean/var/range, binned high (≥0.8) / medium (0.5-0.8) / low (<0.5).
- **Blocking rate / leakage rate** at memory-sanitization layer: # poison queries blocked vs leaked into long-term memory.
- **Retrieval-time filter rate**: of accepted entries, fraction filtered vs retrieved at trust threshold; compare avg trust.
- **False positives / false negatives** relative to the guard decision boundary.
- **Contamination audit**: of accepted entries, how many are confirmed malicious ("Knowledge:" instructions) — the GPT-4o-mini/Gemini split (0 vs 54 confirmed-poisoned-accepted) is the canonical way to expose confidence-vs-security failure.

### 6.3 Concrete attack corpora / datasets

| Corpus/domain | Source | Used for |
|---|---|---|
| **MIMIC-III** clinical data; EHR agent (`EHRAgent`) | `10.48550/arXiv.2503.03704`, `10.48550_arxiv.2601.05504` | patient-ID-redirection poisoning; 5 victim/target pairs (Sec 10.2); 50 indication prompts (Sec 10.1); 101-query attack set (`attack_queries_extensive.json`) |
| **eICU** (EHR, GPT-4) | `10.48550/arXiv.2503.03704` | ASR >90% regime |
| **Webshop** (RAP agent, GPT-4/4o) | same | e-commerce item-redirection (victims: curtain, shampoo, toothbrush) |
| **MMLU** (QA agent) | same | multi-choice answer-redirection (victims: evidence, food, patient) |
| Agent-attack competition (real world) | `10.48550/arXiv.2507.20526` | indirect vs direct injection ASR per policy-violation category |
| Multi-million-text RAG KB (5 poisoned docs) | `10.48550/arxiv.2402.07867` | knowledge-base poisoning ASR (90%) |
| Agent Security Benchmark (ASB) | Zhang et al., ICLR 2025 | formalized attack/defense benchmarking across threat models |

### 6.4 Eval-design recommendations surfaced by the sources

- **Vary the initial-memory state** (empty vs populated) — this single knob moved ASR by an order of magnitude (62%→6.7% on GPT-4o-mini); the #1 realism factor (Sec 3.2).
- **Vary retrieval count `k`** (3/5/10) — directly changes ASR (6%→38%).
- **Run both a capable and a weak host model**; discard underpowered models whose low task accuracy makes attack numbers meaningless (Llama-2-7B, MINJA Sec J).
- **Co-locate benign and adversarial queries** to measure false-positive utility loss (SSGM H1/H2; defense paper's explicit future-work item).
- **Measure a "leakage plateau"** by injecting adversarial content into neighbor nodes of a shared/topology memory (SSGM H2, citing Liu et al. topology-leakage work) — cross-tenant/cross-persona leakage, not just same-tenant injection.
- **Track drift over horizon** with LLM-as-a-Judge + BERTScore fidelity on ground truth (LongMemEval-style) to verify governance bounds drift (SSGM H1/Theorem 1).

---

## 7. Implementable controls — checklist for a Rust memory backend

Each control tied to the citation that motivates it. Map to `myelin` stages: `ingest → extract → consolidate → index → retrieve → rerank → compose → forget`.

### C1. Immutable provenance per record (REQUIRED)
Every record stores: contributing user, contributing agent, accessed resources, creation timestamp, and an immutable source pointer (source doc/chunk ID + line range). Enables retrospective permission checks and audit.
**Motivation**: `10.48550/arXiv.2505.18279` Sec 3.2 (provenance → permission check, "full auditability"); `10.48550/arxiv.2603.11768` P2 (provenance `σ(μ)` required at read).

### C2. Capability scoping via bipartite ACL (REQUIRED)
Persist `G_UA(t)` and `G_AR(t)`; enforce the `ℳ(u,a,t)` predicate (subset-of-agents ⊆ user's invocable set AND subset-of-resources ⊆ agent's accessible set) before any fragment is surfaced. Support revocation by edge removal (immediately shrinks reachable memory). Per-user, per-agent, per-time scoped read policies; project views, never copy.
**Motivation**: `10.48550/arXiv.2505.18279` Sec 3.1/3.3 (Eq. 3), Sec 5.6 enforcement; `10.48550/arxiv.2603.11768` P3 (ABAC predicates at query layer).

### C3. Two-tier memory: private vs shared (REQUIRED for multi-user)
Store `ℳ^private(u,t)` and `ℳ^shared(a,t)` with distinct write policies; a user keeps fragments confidential and shares selectively. Shared fragments carry the sharing permission (which users/agents) at write time.
**Motivation**: `10.48550/arXiv.2505.18279` Sec 3.2/3.4.

### C4. Trust tiers + write quarantine (REQUIRED)
Assign every candidate write a multi-signal trust score `t∈[0,1]`: static-pattern filter (poisoning templates: "refer X to Y", "ignore previous", "Knowledge:"), code-safety static analysis, optional sandbox re-execution / answer verification. Route low-trust or unverified writes to a **quarantine buffer** (staged, not committed); promote to long-term memory only above a calibrated threshold. Log every append/reject decision with reasons.
**Motivation**: `10.48550_arxiv.2601.05504` Sec 6.1/6.2 (composite trust scoring, append gate); `10.48550/arxiv.2405.20485` Phantom single-doc write threat.

### C5. Decouple trust from safety — external verification of effect (REQUIRED where tampering is costly)
Do not let a single confidence score gate writes; add an **external "verification-of-effect" check** against immutable ground truth (e.g., validate a patient-ID redirect against the canonical registry before persisting). Directly fixes the documented failure where Gemini accepted 54 poisoned entries at trust=1.0 (confidence filter ≠ security filter).
**Motivation**: `10.48550_arxiv.2601.05504` Sec 8.2/Conclusions (explicit future-work recommendation).

### C6. Retrieval-time sanitization: temporal decay + provenance freshness (REQUIRED)
At `retrieve`: (a) apply cognitive/temporal decay so stale memories are down-weighted or pruned; (b) reject any candidate lacking valid provenance `σ(μ)`; (c) apply known poisoning-pattern filters; (d) rank by (similarity, trust) jointly. Bounds stale/malicious activation in `compose`.
**Motivation**: `10.48550_arxiv.2601.05504` Sec 6.2 (trust-aware retrieval); `10.48550/arxiv.2603.11768` P2 (Weibull decay + provenance); `10.48550/arxiv.2509.11860` forgetting weights (retrieval-induced `α=0.1` > time-induced `β=0.9`).

### C7. Access-scoped + freshness-gated retrieval predicate (REQUIRED)
Retrieved context must satisfy `C_t = { μ ∈ Top-K(q,M) | ACL(μ,u_id) ∧ w(Δτ_μ) ≥ θ_fresh }` — semantic Top-K proposes, governance filters by ACL and freshness before composing. Prevents cross-tenant injection surfacing in other users' context.
**Motivation**: `10.48550/arxiv.2603.11768` Sec 6.2 Eq. (5) (SSGM constrained retrieval), citing `10.48550/arXiv.2505.18279` for ACL.

### C8. Write-governance / gated consolidation (REQUIRED)
Route candidate consolidation deltas `Agent(C_t)` through a write gate: reject updates contradicting protected core facts (`ΔM ∧ M_core ⊨ ⊥`), preventing hallucination/injection cascades from entering the semantic graph. Decouple agent reasoning from memory mutation (middleware between the two).
**Motivation**: `10.48550/arxiv.2603.11768` P1 + Sec 6.2 Eq. (6) — the load-bearing governance mechanism of SSGM.

### C9. Dual-store: mutable active + append-only immutable ledger (REQUIRED)
Pair the mutable memory graph with an **append-only episodic ledger** of raw interactions (source of truth). Enables (a) reversible reconciliation/rollback when drift or contamination is detected; (b) an audit trail for every mutation; (c) a right-to-be-forgotten substrate (delete from mutable view; keep ledger for compliant audit, or purge ledger too per policy).
**Motivation**: `10.48550/arxiv.2603.11768` P4 + Theorem 1 (drift bounded `O(N·ε_step)` via periodic reconciliation); `10.48550/arxiv.2404.13501` (ledger-style retention); GDPR.

### C10. Audit log of every memory operation (REQUIRED)
Append-only log: per candidate write — question, trust score, per-check booleans, accept/reject decision, textual reason; per read — requesting user/agent id, fragments projected into the view, timestamps. Enables adversarial-forensics (traceback to the responsible stored text) and retrospective permission checks.
**Motivation**: `10.48550_arxiv.2601.05504` Sec 6.1 (audit log); `10.48550/arXiv.2505.18279` (full auditability); `10.1145/3696410.3714756` RAGForensics traceback.

### C11. Deletion / unlearning guarantees (REQUIRED — implementable, not yet benchmarked)
Provide a `forget`/`delete` path: (a) remove from the mutable graph and the embedding index; (b) purge the source chunk and its embeddings; (c) update/break any consolidated summaries referencing the removed item; (d) honor a retention/TTL policy (storage limitation). Because no corpus paper quantifies memory-unlearning effectiveness, mark this control as **implemented-for-compliance, unevidenced-in-curated-corpus**, grounded on GDPR (`10.1002/cpe.6426`), SSGM's unlearning agenda (`10.48550/arxiv.2603.11768` Sec 7), and PoisonedRAG's store-level source-removal lesson (`10.48550/arxiv.2402.07867`). The dual-store makes it tractable (mutable delete + ledger retention, or full purge).
**Motivation**: `10.1002/cpe.6426` (GDPR storage-limitation/right-to-be-forgotten); `10.48550/arxiv.2603.11768` Sec 7; `10.48550/arxiv.2402.07867`.

### C12. Isolation / authentication against query-only poisoning (REQUIRED)
MINJA's premise is that **any user** can poison a shared memory bank via queries. Counter with: memory **isolation** (per-tenant/user memory; shared only where explicitly permitted — C3), strong user authentication, and rate limiting on write/append paths (spam detection is a natural defense, `10.48550_arxiv.2601.05504` Sec 5.3). Where isolation is impossible (shared-value memory), treat the shared tier as untrusted input and heavily sanitize (C4+C5+C6), and keep the shared tier small relative to legitimate content (the single biggest real-world ASR reducer).
**Motivation**: `10.48550/arXiv.2503.03704` Impact Statement (memory isolation, strong auth, secure management, prompt filtering); `10.48550_arxiv.2601.05504` Sec 5.3 (pre-existing memory collapses ASR; spam flagging).

### C13. PII detection + redaction at ingest & write (REQUIRED)
Detect and redact/anonymize PII at `ingest` and at `consolidate` before persistence (emails, phones, patient IDs, personas). Prevent membership-inference leakage from consolidated memory and persona-aided jailbreak. For the shared tier, apply entity redaction + watermarking before cross-tenant surfacing.
**Motivation**: `10.48550/arxiv.2404.18231` Sec 7.4 (anonymization, ProPILE, persona-aided jailbreak); `10.48550/arxiv.2406.01171` Sec 5.5 (membership-inference leakage); `10.48550/arxiv.2112.04359` Sec 2.2.4 (leak vs inference distinction); `10.48550/arXiv.2507.03724` MemOS (redaction + watermarking before sharing).

### C14. Isolated-group retrieval / certified aggregation where feasible (OPTIONAL)
For high-stakes `compose`, partition retrieved chunks into disjoint groups, generate per-group answers, and aggregate (keyword/decoding) — gives certifiable robustness against a bounded number of injected chunks (RobustRAG isolate-then-aggregate). Cost: more LLM calls (latency). OPTIONAL for `myelin` v1; strong candidate once the threat model demands adaptive-attacker guarantees.
**Motivation**: `10.48550/arxiv.2405.15556` RobustRAG (certifiable bound).

---

## 8. Top 5 highest-leverage findings for a Rust implementation

1. **The realistic-condition number that matters**: pre-existing legitimate memory collapses MINJA ASR from 62%→6.7% (GPT-4o-mini) and 53%→0% (Llama). A shared memory backend kept *large and populated with legitimate per-user content* is itself the strongest defense — but only when paired with small retrieval `k` (ASR rose 6%→38% as `k` went 3→10) and write-rate limits. Design `retrieve k` + quarantine defaults around this.
2. **Trust-scoring memory sanitization fails exactly when it matters**: both defense experiments (`10.48550_arxiv.2601.05504`) show two failure modes — over-conservative → empty memory (utility collapse), or model-overconfident → 54/82 accepted entries are poisoned at trust=1.0. **Decouple trust from safety** via external verification-of-effect; never gate security on a single self-reported confidence score.
3. **The provenance + bipartite-ACL formalism (`10.48550/arXiv.2505.18279`) is directly implementable** and is the correct access-control substrate: per-record immutable provenance + `ℳ(u,a,t)` predicate + private/shared tiers + per-scope read/write policies. It composes with SSGM's constrained-retrieval predicate.
4. **Governance is a write-gate + read-gate + ledger, not a static store**: SSGM's decoupling (write-governance operator Eq. 6, constrained-retrieval Eq. 5, dual-store reconciliation with a provable `O(N·ε_step)` drift bound) gives a concrete, provable architecture. The append-only immutable ledger is simultaneously the audit log (C10), the rollback primitive (C9), and the unlearning/right-to-be-forgotten substrate (C11) — one storage design serves all three.
5. **Evaluation must vary initial-memory state, `k`, and co-located benign traffic**: the single biggest fidelity lever is initial-memory population; the standard "ISR≈98.2%" figure only holds for empty/near-empty shared memory with PSS. Any `myelin` security eval should report ISR/ASR/UD *and* trust-distribution/leakage metrics under both idealized and populated-memory conditions, on a capable host model, with benign+adversarial queries interleaved.

---

## Appendix — papers read (DOIs / doc_ids) and search coverage

**Read in full / key sections (primary):**
- `10.48550/arXiv.2503.03704` — MINJA (memory injection attack) — full text.
- `10.48550_arxiv.2601.05504` — "Memory Poisoning Attack and Defense on Memory Based LLM-Agents" — full text (attack realism + two defenses).
- `10.48550/arXiv.2505.18279` — Collaborative Memory (access control) — full text incl. Sec 3 formalism, Sec E dynamic scenario, Table 19.
- `10.48550_arxiv.2603.11768` — SSGM governance framework — full text incl. Sec 6 formalism, Theorem 1.

**Read excerpts / citation-verified:**
- `10.48550/arXiv.2507.20526` — agent-competition indirect-vs-direct ASR table.
- `10.48550/arxiv.2402.07867` — PoisonedRAG (external search; abstract/claims).
- `10.48550/arxiv.2405.15556` — RobustRAG (external; abstract).
- `10.48550/arxiv.2405.20485` — Phantom (external; abstract).
- `10.1145/3696410.3714756` — RAGForensics (external; abstract).
- `10.48550/arxiv.2509.11860` — MOOM forgetting ablation.
- `10.48550/arXiv.2507.03724` — MemOS lifecycle/governance.
- `10.48550/arxiv.2404.18231`, `10.48550/arxiv.2406.01171` / `10.18653/v1/2024.findings-emnlp.969` — persona/privacy.
- `10.48550/arxiv.2112.04359` — privacy leaks/inference.
- `10.1002/cpe.6426` — GDPR/right-to-be-forgotten for chatbots.
- `10.48550/arxiv.2404.13501` — agent-memory survey (operations).
- `10.48550/arxiv.2305.10250` — MemoryBank Ebbinghaus forgetting model.
- `10.48550/arxiv.2304.13343` — Self-Controlled Memory framework.

**Searched, absent or citation-only in local corpus:** BadRAG quantitative results (only cited); AgentPoison full text (only cited as NeurIPS 2024); machine-unlearning applied to memory stores (no quantitative primary source found — flagged as a gap).
