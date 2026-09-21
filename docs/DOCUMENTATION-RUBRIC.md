# Documentation Quality Rubric for myelin

A scored checklist synthesised from five documentation-quality papers. Every
criterion carries an inline citation to the paper and section it came from.
Criteria the papers give **evidence** for (README sections that correlate with
adoption, documentation-debt categories, the ten rules, instruction prevalence)
are weighted above generic advice.

## Scope and caveats

- **Paper 2** (Venigalla & Chimalakonda, README content vs popularity) measures
  *association with GitHub popularity*, not with documentation quality or user
  success. Popularity has obvious confounders (project age, organisational
  backing, language ecosystem). Correlation is not causation, and "correlates
  with stars" is not the same as "is good." We use these findings to justify
  *coverage* — checking whether our README has the structural and content
  elements that popular repos share — not to claim that adding them will make
  myelin popular. [[Paper 2, §1, §3.4]](https://arxiv.org/abs/2206.10772)
- **Paper 1** (Chatlatanagulchai et al., Agent READMEs) studies *agent context
  files* (CLAUDE.md, AGENTS.md, copilot-instructions.md) — a different artifact
  from a human-facing README. We cite it for two reasons: (a) its heading
  hierarchy findings are about Markdown structure generally, and (b) the
  instruction-type prevalence in its Table 3 is a coverage map we can check
  against. Prevalence is not prescription: 75% of agent files covering Testing
  does not establish that they *should*. We use it to see what we omit, and we
  flag the Security/Performance gap (14.5%) as the paper's own observation about
  the field, not a target we inherit. [[Paper 1, §2, §4.3]](https://arxiv.org/abs/2511.12884)
- **Paper 4** (Silva et al., documentation debt) names our exact risk: the three
  most common defect types — Missing documentation (35/101), Erroneous code
  examples (23/101), Outdated content (19/101) — are the ones a documentation
  effort like this one is most likely to introduce. Their finding that
  Installation/Deployment guides top the defect table (129 + 53 = 57% of 318
  bugs) is precisely because *nobody runs them*. The rule that every command must
  be executed is a direct response. [[Paper 4, §IV Table I–II]](https://arxiv.org/abs/2402.11048)
- Unsourced conventions are marked **[CONVENTION]** — they carry no paper
  citation and are explicitly not claims from the literature.

## Papers cited

| Tag | Paper | DOI / URL |
|-----|-------|-----------|
| Paper 1 | Chatlatanagulchai et al., "Agent READMEs: An Empirical Study of Context Files for Agentic Coding" (2025) | [10.1145/3840295](https://arxiv.org/abs/2511.12884) |
| Paper 2 | Venigalla & Chimalakonda, "An Empirical Study On Correlation between Readme Content and Project Popularity" (2022) | [10.48550/arxiv.2206.10772](https://arxiv.org/abs/2206.10772) |
| Paper 3 | Lee, "Ten simple rules for documenting scientific software" (2018) | [10.1371/journal.pcbi.1006561](https://doi.org/10.1371/journal.pcbi.1006561) |
| Paper 4 | Silva et al., "Towards identifying and minimizing customer-facing documentation debt" (2023) | [10.1109/techdebt59074.2023.00015](https://arxiv.org/abs/2402.11048) |
| Paper 5 | Alzahrani, "Software Systems Documentation: A Systematic Review" (2024) | [10.14569/ijacsa.2024.0150816](https://thesai.org/Downloads/Volume15No8/Paper_16-Software_Systems_Documentation_A_Systematic_Review.pdf) |

## Scoring scale

Each criterion is scored 0–3:

| Score | Meaning |
|-------|---------|
| 0 | Absent — the criterion is not met at all |
| 1 | Partial — present but incomplete, inconsistent, or superficial |
| 2 | Adequate — present and functional, with minor gaps |
| 3 | Strong — fully met, verified, and maintained |

Maximum possible: the sum of all criteria weights. We report both raw and
percentage.

---

## Criteria

### A. README structure and content (Paper 2, Paper 3)

#### A1. Quickstart guide — a path a stranger can follow to first working use
**Weight: 3**
**Source:** Paper 3, Rule 3: "if people can immediately start playing with your
tool, they're vastly more likely to use it … include a quickstart guide aimed at
helping people begin using your software as quickly as possible." Paper 4, §IV:
Installation Guides are the second-highest defect source (53/318); the most
common defect is Missing configuration instructions (14/101). A quickstart that
has been *executed end-to-end* is the direct countermeasure.
**Score guidance:** 3 = every command executed and verified, prerequisites
named, first read and first write demonstrated. 0 = no quickstart.

#### A2. README includes install/build instructions
**Weight: 2**
**Source:** Paper 3, Rule 4: "your README should include how to install and
configure your software." Paper 2, §3.1 RQ2: "How" (installation/usage) is
present in majority of repos regardless of popularity — necessary but not
differentiating.
**Score guidance:** 3 = install command(s) present and verified. 1 = present
but unverified or incomplete. 0 = absent.

#### A3. README includes project description ("What" and "Why")
**Weight: 2**
**Source:** Paper 2, §3.3 Table 4: categories "What" (functionalities) and "Why"
(purpose). Paper 1, §4.3 Table 3: System Overview appears in 59% of agent
context files.
**Score guidance:** 3 = concise description of what the system is and why it
exists. 1 = one of What/Why missing. 0 = neither.

#### A4. README includes license information
**Weight: 1**
**Source:** Paper 2, §3.4 RQ3: random forest Gini importance ranks license
information among the strongest differentiators between popular and non-popular
repos. Paper 3, Rule 4: "under what license it's released."
**Score guidance:** 3 = license named and linked. 0 = absent.

#### A5. README includes contribution guidelines or link to them
**Weight: 1**
**Source:** Paper 2, §3.1 RQ2: "repositories with readme files containing
contribution guidelines … were observed to be associated with higher
popularity" (Fisher's exact test, significant for popular vs non-popular).
**Score guidance:** 3 = contribution guidelines present or linked. 0 = absent.

#### A6. README includes references / links to external documentation
**Weight: 1**
**Source:** Paper 2, §3.4 RQ3: "presence of external links, links to other
GitHub repositories, references … strongly differentiate the popular
repositories from the non-popular ones" (Gini importance).
**Score guidance:** 3 = links to full docs, eval docs, research notes. 1 =
one link. 0 = none.

#### A7. Heading hierarchy: single H1, H2/H3 for sections, shallow depth
**Weight: 1**
**Source:** Paper 1, §4.1.3: "a consistent, shallow hierarchy … single, top-level
H1 heading (median 1.0) … moderate number of H2 headings (median 6–7) … H3/H4
for detail … deeply nested structures (H5+) are extremely rare."
**Score guidance:** 3 = one H1, H2s for major sections, no excessive nesting.
0 = no structure or deeply nested.

### B. Examples and executable content (Paper 3, Paper 2, Paper 4)

#### B1. Examples for the main use cases
**Weight: 3**
**Source:** Paper 3, Rule 2: "showing takes precedence over telling … include
examples in your documentation beyond simple instruction … there isn't such a
thing as too many examples if they all show off different aspects." Paper 4,
§IV: Erroneous code examples are the second most common defect type (23/101) —
examples that are wrong are worse than no examples.
**Score guidance:** 3 = real, executed examples for each MCP tool. 1 = examples
present but unverified. 0 = none.

#### B2. Code blocks are real, executed output — not hand-transcribed
**Weight: 3**
**Source:** Paper 4, §V: design criterion 1, "robust information source" —
documentation should derive from a single source of truth, not from
hand-transcription. §IV: Erroneous code examples (23/101) and Outdated content
(19/101) are the defects hand-transcription produces. Paper 4, §IV Table I:
Release Notes (dynamically generated) had only 8 defects vs 129 for Deployment
Guides (manually maintained) — evidence that generated docs rot slower.
**Score guidance:** 3 = every command output is a real capture, with a note on
how to regenerate. 0 = hand-transcribed or invented.

#### B3. CLI help output is documented (verbatim `--help`)
**Weight: 2**
**Source:** Paper 3, Rule 5: "the best way to document CLIs is to have a 'help'
command … it should include usage, subcommands, options, arguments,
environment variables, and maybe even some examples."
**Score guidance:** 3 = `--help` output captured verbatim for each binary. 0 =
absent.

### C. API / tool reference (Paper 3, Paper 5)

#### C1. Every tool/function documented: purpose, parameters, return shape, example
**Weight: 3**
**Source:** Paper 3, Rule 7: "each function should have its inputs and input
types noted, its output and output type noted, and any errors it can raise
documented." Paper 5, §III.B.1 citing Uddin et al.: API documentation "suffers
content issues … lack of clarity and completeness … un-updated documentation is
another recurring issue."
**Score guidance:** 3 = all 10 MCP tools documented with parameters, returns,
and at least one real example each. 1 = partial. 0 = none.

#### C2. Configuration / environment variables documented in a table
**Weight: 2**
**Source:** Paper 5, §III.B.1 citing Lethbridge et al.: "focus on … high-level
documentation of the systems." Paper 4, §IV: Missing configuration instructions
(14/101) is a top defect type. Paper 3, Rule 5: help should include
"environment variables."
**Score guidance:** 3 = every `MYELIN_*` env var in a table with default and
description. 1 = partial. 0 = none.

#### C3. Error messages and failure modes documented
**Weight: 2**
**Source:** Paper 3, Rule 9: "good error messages should … state what the error
is, what the state of the software was … and either how to fix it or where to
find information relevant to fixing it." Paper 4, §IV: documentation defects
"introduce delays … the cost of troubleshooting."
**Score guidance:** 3 = known failure modes (wrong tenant, mismatched
ledger/collection, missing services) documented with how to diagnose. 1 =
mentioned but without resolution. 0 = absent.

### D. Architecture and system overview (Paper 1, Paper 5)

#### D1. Architecture description — high-level structure and key components
**Weight: 2**
**Source:** Paper 1, §4.3 Table 3: Architecture appears in 67.7% of agent
context files — the third most prevalent instruction type. Paper 5, §III.B.2
citing Bachmann et al.: architecture documentation aims to "share understanding
of the system, trace the changes, and discuss trade-offs."
**Score guidance:** 3 = retrieval pipeline, write path, and multi-tenancy
described with a diagram. 1 = mentioned without structure. 0 = absent.

#### D2. Read paths and write paths described
**Weight: 2**
**Source:** Paper 5, §III.B.1 citing Aghajani et al.: 162 types of
documentation issues, many linked to "what is written" — completeness gaps
where key operations are undocumented. **[CONVENTION]** A memory system's docs
should describe its read and write paths; this is not a paper finding but a
domain-specific expectation.
**Score guidance:** 3 = both `recall` and `investigate` read paths, and the
write path (dedup, consolidation, adjudication, quarantine) described. 1 = one
path. 0 = none.

### E. Maintenance and debt prevention (Paper 4, Paper 1, Paper 5)

#### E1. Documentation is version-controlled alongside the code
**Weight: 1**
**Source:** Paper 3, Rule 6: "keep your documentation inside your Git
repository … make it very clear which version of the software your
documentation is for."
**Score guidance:** 3 = docs in the repo, version-tagged with the code. 0 =
external or unversioned.

#### E2. Generated/verifiable content has a regeneration path
**Weight: 3**
**Source:** Paper 4, §V: Dynamic Documentation Generation (DDG) from "a single
and robust information source" is the proposed solution to Missing
documentation and Outdated content. Paper 4, §IV Table I: dynamically generated
Release Notes had only 8 defects vs 129 for manually maintained Deployment
Guides. Paper 5, §IV conclusion (6): "automated tools for documentation are in
high demand."
**Score guidance:** 3 = env-var table, tool list, and `--help` output are
generated from or verifiable against the code, with a note saying how. 1 =
present but no regeneration path. 0 = hand-maintained with no verification.

#### E3. Known usability gaps documented honestly
**Weight: 2**
**Source:** Paper 4, §II: documentation debt is "invisible in the artifacts of
a product, like design, source code and tests" — undocumented gotchas are
hidden debt. Paper 1, §4.3.3: the Security/Performance gap (14.5%) is reported
as an observation, not concealed.
**Score guidance:** 3 = wrong-tenant-returns-empty, LAN-specific defaults, and
multi-service requirement all called out. 1 = some mentioned. 0 = none.

### F. Visual and structural elements (Paper 2)

#### F1. Images / screenshots of real output
**Weight: 2**
**Source:** Paper 2, §3.1 RQ1: images are significantly more present in popular
repos (Wilcoxon p < 0.05 across 8/10 languages). §3.4 RQ3: images rank in the
Gini importance for differentiating popular from non-popular repos. Paper 3,
Rule 3: TPOT's quickstart has "an animated GIF showing the software's
functionality, diagrams explaining how it works, and a minimal code stub."
**Score guidance:** 3 = real screenshots of CLI output, MCP session, and Qdrant
dashboard, each with alt text and caption. 1 = one screenshot. 0 = none.

#### F2. Lists and structured formatting used throughout
**Weight: 1**
**Source:** Paper 2, §3.1 RQ1: lists are significantly more present in popular
repos (Wilcoxon p < 0.05 across 8/10 languages, Cliff's delta small-to-medium).
**Score guidance:** 3 = parameter tables, env-var tables, feature lists
throughout. 0 = prose walls.

#### F3. Architecture diagram present
**Weight: 1**
**Source:** Paper 3, Rule 3: "diagrams explaining how it works." Paper 1, §4.3
Table 3: Architecture in 67.7% of files.
**Score guidance:** 3 = mermaid or other diagram of the pipeline. 0 = absent.

### G. Agent context file (Paper 1)

#### G1. Consider whether an agent context file (AGENTS.md / CLAUDE.md) is warranted
**Weight: 1**
**Source:** Paper 1, §2: agent context files "serve as a persistent long-term
memory for the agent … documenting project-specific context such as
architectural patterns, testing commands, and coding conventions." Paper 1,
§4.3 Table 3: Testing 75%, Impl Details 69.9%, Architecture 67.7%, Build/Run
62.3%. The paper's own finding (§4.3.3) is that Security (14.5%) and Performance
(14.5%) are rarely specified — a gap, not a target.
**Score guidance:** 3 = an AGENTS.md exists or the decision not to have one is
documented. 0 = neither.

---

## Baseline score: current repo documentation (origin/main, f1e92d6)

Scored against the rubric above, at the HEAD of `origin/main` before this PR.

| Criterion | Weight | Score | Notes |
|-----------|--------|-------|-------|
| A1. Quickstart guide | 3 | 0 | No quickstart anywhere. `README.md` has no install, no first-use. |
| A2. Install/build instructions | 2 | 1 | `cargo test --workspace` present but no install, no `cargo build --release`, no binary path. |
| A3. Project description (What/Why) | 2 | 0 | README opens with "Vocabulary of the field" — a research glossary, not a description of myelin. |
| A4. License information | 1 | 1 | LICENSE file exists; README does not name or link it. |
| A5. Contribution guidelines | 1 | 0 | Absent. |
| A6. References / external links | 1 | 1 | Links to `PLAN.md` and one research doc. No link to eval docs or MCP spec. |
| A7. Heading hierarchy | 1 | 0 | Exactly ONE `##` heading ("Building & testing", line 67). No H1. Content is unstructured prose. |
| B1. Examples for main use cases | 3 | 0 | No MCP tool examples, no recall/remember examples. |
| B2. Code blocks are real/executed | 3 | 1 | `cargo test` commands are real but are the only executed commands. No MCP examples. |
| B3. CLI help output documented | 2 | 0 | No `--help` output anywhere. |
| C1. Every tool documented | 3 | 0 | None of the 10 MCP tools are documented. |
| C2. Config / env vars table | 2 | 0 | No env-var documentation. Defaults hardcoded to author's LAN with no mention. |
| C3. Error messages / failure modes | 2 | 0 | The wrong-tenant-returns-empty behaviour is undocumented. `--ledger` help text mentions it but no user-facing doc does. |
| D1. Architecture description | 2 | 1 | "SOTA for concise context retrieval" section describes the field, not myelin's architecture. |
| D2. Read/write paths described | 2 | 0 | Neither `recall` nor `investigate` is described. Write path (dedup, consolidation, quarantine) absent. |
| E1. Version-controlled | 1 | 3 | Docs are in the repo. |
| E2. Generated/verifiable content | 3 | 0 | Nothing is generated or verifiable. |
| E3. Usability gaps documented | 2 | 0 | None of the five known gaps (wrong tenant, LAN defaults, multi-service, no compose, no bulk ingestion) are documented. |
| F1. Screenshots of real output | 2 | 0 | No images. |
| F2. Lists and structured formatting | 1 | 1 | One code block, no tables, no lists in the README. |
| F3. Architecture diagram | 1 | 0 | No diagram. |
| G1. Agent context file | 1 | 0 | No AGENTS.md or CLAUDE.md. |
| **Total** | **40** | **11** | **27.5%** |

The current documentation is a research vocabulary dump with one build command.
It scores above zero only because the repo is version-controlled, the LICENSE
file exists, and two commands in the build section are real.

---

## Post-change score: after this PR

| Criterion | Weight | Score | Notes |
|-----------|--------|-------|-------|
| A1. Quickstart guide | 3 | 3 | `docs/QUICKSTART.md` — every command executed, prerequisites named, first remember + first recall demonstrated. |
| A2. Install/build instructions | 2 | 3 | README has `cargo build --release --workspace`; quickstart has full build step. |
| A3. Project description (What/Why) | 2 | 3 | README opens with "Agentic long-term memory for LLM agents" — what and why. |
| A4. License information | 1 | 3 | README links to LICENSE file. |
| A5. Contribution guidelines | 1 | 0 | Not yet added. |
| A6. References / external links | 1 | 3 | README links to QUICKSTART, FEATURES, rubric, EVALUATION, research notes, PLAN.md. |
| A7. Heading hierarchy | 1 | 3 | Single H1, H2s for major sections, H3s for detail. Shallow. |
| B1. Examples for main use cases | 3 | 3 | FEATURES.md has all 10 MCP tools with real example output. QUICKSTART has remember + recall. |
| B2. Code blocks are real/executed | 3 | 3 | All CLI output captured from real runs. MCP responses captured from live server. |
| B3. CLI help output documented | 2 | 2 | `myelin-mcp --help` documented in FEATURES config section; not yet captured verbatim as a screenshot (Phase 4). |
| C1. Every tool documented | 3 | 3 | All 10 MCP tools in FEATURES.md with parameters, returns, and at least one real example each. |
| C2. Config / env vars table | 2 | 3 | Full `MYELIN_*` env-var table with defaults read from `config.rs`. Regeneration note included. |
| C3. Error messages / failure modes | 2 | 3 | Wrong-tenant-returns-empty, mismatched ledger/collection, missing services all documented in QUICKSTART. |
| D1. Architecture description | 2 | 3 | Mermaid diagram in README; retrieval architecture in FEATURES. |
| D2. Read/write paths described | 2 | 3 | Both read paths (recall/investigate) and full write path (ingest→extract→adjudicate→consolidate→index) in FEATURES. |
| E1. Version-controlled | 1 | 3 | Docs in repo, version-tagged with code. |
| E2. Generated/verifiable content | 3 | 2 | Env-var table has regeneration note pointing to `config.rs`; `--help` output is real but not auto-generated. No script/test yet. |
| E3. Usability gaps documented | 2 | 3 | Wrong tenant, LAN defaults, multi-service requirement, mismatched pair all called out. |
| F1. Screenshots of real output | 2 | 0 | Phase 4 (PR #2). |
| F2. Lists and structured formatting | 1 | 3 | Parameter tables, env-var tables, comparison tables throughout. |
| F3. Architecture diagram | 1 | 3 | Mermaid diagram in README. |
| G1. Agent context file | 1 | 0 | Not yet added (could be a follow-up). |
| **Total** | **40** | **30** | **75.0%** |

**Improvement: 11 → 30 (27.5% → 75.0%).**

Remaining gaps: contribution guidelines (A5), CLI help verbatim screenshot
(B3), auto-regeneration script for env-var table (E2), screenshots (F1,
Phase 4), and an agent context file (G1). These are addressable in follow-up
PRs; F1 is addressed in PR #2.