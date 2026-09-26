//! `myelin-eval` — the evaluation harness (`PLAN.md` §3.3).
//!
//! The subcommand surface exists now so later milestones add bodies rather than
//! argument surfaces.  Only `fetch` is implemented (M3); the remaining six
//! arms stay as `not implemented (milestone M5+)` placeholders.

use std::path::Path;

use anyhow::Context;
use clap::{Parser, Subcommand};

use myelin_core::config::MyelinConfig;
use myelin_core::model::query::Mode;
use myelin_eval::build::build_locomo;
use myelin_eval::datasets::{self, locomo, longmemeval};

/// myelin-eval — the agentic-memory evaluation harness
#[derive(Parser)]
#[command(name = "myelin-eval")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

/// The corpora `PLAN.md` §9.1 names. Each pins its own default collection
/// and ledger so that a `--corpus` switch cannot quietly append one corpus's
/// records to another's memory.
/// Corpora `bench` can score, each carrying the paths `build` wrote.
///
/// Defaults live on the enum rather than on the flags so `--corpus
/// longmemeval-s` alone is correct; a default collection of `myelin_locomo`
/// silently benching the wrong store is the failure this prevents.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum BenchCorpus {
    Locomo,
    #[value(name = "longmemeval-s")]
    LongmemevalS,
}

struct BenchDefaults {
    dataset: &'static str,
    collection: &'static str,
    ledger: &'static str,
    slug: &'static str,
    /// Which scorer this corpus reports by default.
    ///
    /// Per corpus and not one global flag, because the evidence that settled
    /// it is per corpus: `docs/measurements/m14-temporal-scorer.md` validated
    /// the date-aware scorer against a judge on **LoCoMo category 2** and
    /// found it agreeing 96.7% against token F1's 84.9% (+11.8 points, 95% CI
    /// [+7.7, +15.8]). LongMemEval_S has no such docket, and its grammar
    /// coverage is 26 of 470 answerable golds — 11 of 127 even in its own
    /// temporal-reasoning stratum — so there is nothing there to flip on.
    scorer: myelin_eval::bench::Scorer,
}

impl BenchCorpus {
    fn defaults(self) -> BenchDefaults {
        match self {
            Self::Locomo => BenchDefaults {
                dataset: "data/locomo10.json",
                collection: myelin_eval::bench::LOCOMO_COLLECTION,
                ledger: "data/locomo.ledger",
                slug: "locomo",
                scorer: myelin_eval::bench::Scorer::Temporal,
            },
            Self::LongmemevalS => BenchDefaults {
                dataset: "data/longmemeval_s.json",
                collection: myelin_eval::bench::LONGMEMEVAL_S_COLLECTION,
                ledger: "data/longmemeval_s.ledger",
                slug: "lme_s",
                scorer: myelin_eval::bench::Scorer::TokenF1,
            },
        }
    }
}

fn mode_slug(mode: Mode) -> &'static str {
    match mode {
        Mode::Recall => "recall",
        Mode::Investigate => "investigate",
    }
}

/// Where a `bench` run lands when `--out` is not given.
///
/// Every switch gets its own suffix, in a fixed order, so a directory name is
/// a function of the switch set and no arm can clobber another — least of all
/// the M9 baselines in `runs/locomo_recall` and `runs/lme_s_recall` that every
/// paired-CI comparison is against.
///
/// The scorer suffix is keyed on the scorer's *identity*, not on whether it is
/// the default: M14 flipped LoCoMo's default to `temporal`, and that must not
/// silently redirect a `--scorer token-f1` run onto the M9 baseline path.
fn bench_out_dir(
    slug: &str,
    mode: Mode,
    switches: &myelin_eval::bench::BenchSwitches,
    scorer: myelin_eval::bench::Scorer,
) -> String {
    let cats = if switches.categories.is_empty() {
        String::new()
    } else {
        let codes: Vec<String> = switches.categories.iter().map(u8::to_string).collect();
        format!("_cat{}", codes.join(""))
    };
    // λ as an integer percentage: a directory named `_mmr0.7` would carry a
    // `.` into every path, and `_mmr70` sorts.
    let mmr = switches
        .mmr
        .map(|lambda| format!("_mmr{}", (lambda * 100.0).round() as u32))
        .unwrap_or_default();
    format!(
        "runs/{slug}_{}{}{}{}{mmr}{}{}{}{}{cats}{}",
        mode_slug(mode),
        if switches.graph { "_graph" } else { "" },
        if switches.chronological {
            "_chrono"
        } else {
            ""
        },
        if switches.question_date { "_qdate" } else { "" },
        if switches.select_sufficient {
            "_sel"
        } else {
            ""
        },
        if switches.profile_ledger.is_some() {
            "_prof"
        } else {
            ""
        },
        if switches.events_ledger.is_some() {
            "_events"
        } else {
            ""
        },
        if switches.profile_clause {
            "_pclause"
        } else {
            ""
        },
        match scorer {
            myelin_eval::bench::Scorer::TokenF1 => "",
            myelin_eval::bench::Scorer::Temporal => "_temporal",
            myelin_eval::bench::Scorer::Judge => "_judge",
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use myelin_eval::bench::{BenchSwitches, Scorer};

    #[test]
    fn only_token_f1_with_every_switch_off_reaches_the_m9_baseline_path() {
        // `runs/locomo_recall` and `runs/lme_s_recall` are the committed M9
        // artifacts every paired CI is measured against. Exactly one switch
        // set may name them, and it is the one that produced them.
        let off = BenchSwitches::default();
        assert_eq!(
            bench_out_dir("locomo", Mode::Recall, &off, Scorer::TokenF1),
            "runs/locomo_recall"
        );
        assert_eq!(
            bench_out_dir("lme_s", Mode::Recall, &off, Scorer::TokenF1),
            "runs/lme_s_recall"
        );
        // The post-M14 LoCoMo default must not be one of them.
        assert_eq!(
            bench_out_dir("locomo", Mode::Recall, &off, Scorer::Temporal),
            "runs/locomo_recall_temporal"
        );
    }

    #[test]
    fn switch_suffixes_keep_their_fixed_order() {
        assert_eq!(
            bench_out_dir(
                "locomo",
                Mode::Recall,
                &BenchSwitches {
                    graph: true,
                    chronological: true,
                    question_date: true,
                    ..Default::default()
                },
                Scorer::Temporal
            ),
            "runs/locomo_recall_graph_chrono_qdate_temporal"
        );
        assert_eq!(
            bench_out_dir(
                "locomo",
                Mode::Investigate,
                &BenchSwitches {
                    chronological: true,
                    ..Default::default()
                },
                Scorer::TokenF1
            ),
            "runs/locomo_investigate_chrono"
        );
        // M21's two arms take their places in the same fixed order, between
        // `_qdate` and `_prof`.
        assert_eq!(
            bench_out_dir(
                "lme_s",
                Mode::Recall,
                &BenchSwitches {
                    question_date: true,
                    mmr: Some(0.7),
                    select_sufficient: true,
                    profile_ledger: Some("data/p.ledger".into()),
                    ..Default::default()
                },
                Scorer::Judge
            ),
            "runs/lme_s_recall_qdate_mmr70_sel_prof_judge"
        );
    }

    /// M20's two arms are independent, so all four combinations must name
    /// four different directories — an arm that silently wrote over the
    /// baseline's rows would make the paired difference unmeasurable.
    #[test]
    fn the_two_profile_arms_never_share_a_directory() {
        let arm = |profile: bool, profile_clause: bool| {
            bench_out_dir(
                "lme_s",
                Mode::Recall,
                &BenchSwitches {
                    profile_ledger: profile.then(|| "data/p.ledger".to_string()),
                    profile_clause,
                    categories: vec![3],
                    ..Default::default()
                },
                Scorer::Judge,
            )
        };
        assert_eq!(arm(false, false), "runs/lme_s_recall_cat3_judge");
        assert_eq!(arm(true, false), "runs/lme_s_recall_prof_cat3_judge");
        assert_eq!(arm(false, true), "runs/lme_s_recall_pclause_cat3_judge");
        assert_eq!(arm(true, true), "runs/lme_s_recall_prof_pclause_cat3_judge");
    }

    /// M21's arms are measured against a base run on the same store, so the
    /// three directories must be three directories — and λ has to survive
    /// into the name, or two sweep points would overwrite each other.
    #[test]
    fn the_two_selection_arms_never_share_a_directory() {
        let arm = |mmr: Option<f32>, select_sufficient: bool| {
            bench_out_dir(
                "lme_s",
                Mode::Recall,
                &BenchSwitches {
                    mmr,
                    select_sufficient,
                    categories: vec![4],
                    ..Default::default()
                },
                Scorer::Judge,
            )
        };
        assert_eq!(arm(None, false), "runs/lme_s_recall_cat4_judge");
        assert_eq!(arm(Some(0.7), false), "runs/lme_s_recall_mmr70_cat4_judge");
        assert_eq!(arm(Some(0.3), false), "runs/lme_s_recall_mmr30_cat4_judge");
        assert_eq!(arm(None, true), "runs/lme_s_recall_sel_cat4_judge");
    }

    /// A stratum arm never names the full-set path it is measured against.
    #[test]
    fn a_stratum_arm_cannot_clobber_the_full_set_path() {
        let arm = |categories: &[u8]| {
            bench_out_dir(
                "locomo",
                Mode::Recall,
                &BenchSwitches {
                    categories: categories.to_vec(),
                    ..Default::default()
                },
                Scorer::Temporal,
            )
        };
        assert_eq!(arm(&[2]), "runs/locomo_recall_cat2_temporal");
        assert_eq!(arm(&[2, 5]), "runs/locomo_recall_cat25_temporal");
        // No stratum: the 1,540-question path the baseline owns.
        assert_eq!(arm(&[]), "runs/locomo_recall_temporal");
    }

    #[test]
    fn the_scorer_default_is_per_corpus() {
        // M14's judge docket was LoCoMo category 2 and nothing else; the
        // grammar reaches 26 of LongMemEval_S's 470 answerable golds, so the
        // flip is LoCoMo-only.
        assert_eq!(BenchCorpus::Locomo.defaults().scorer, Scorer::Temporal);
        assert_eq!(BenchCorpus::LongmemevalS.defaults().scorer, Scorer::TokenF1);
    }
}

/// Which grid `ablate --width` sweeps.
///
/// An enum rather than a set of mutually exclusive booleans: `--select`
/// and `--budget` could both be passed and one silently won, which is the
/// same class of quiet-wrong-configuration defect M23's drift guard and
/// M27's degraded guard exist for.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum GridName {
    /// `prefetch_limit` × `rerank_depth` at the shipped budget (M25/M26).
    Width,
    /// The two width extremes, with and without the selector (M27).
    Select,
    /// The shipped width swept over `max_tokens` (M28/M29).
    Budget,
    /// The shipped width crossed over budget and selector — does the
    /// selector add anything the tight budget is not already doing? (M31).
    Interaction,
    /// LoCoMo's shipped retrieval, one cell, for sweeping `k` (L3).
    Shipped,
}

impl GridName {
    fn cells(self) -> &'static [(u64, usize, bool, usize)] {
        match self {
            GridName::Width => &myelin_eval::ablate::WIDTH_GRID,
            GridName::Select => &myelin_eval::ablate::SELECT_GRID,
            GridName::Budget => &myelin_eval::ablate::BUDGET_GRID,
            GridName::Interaction => &myelin_eval::ablate::INTERACTION_GRID,
            GridName::Shipped => &myelin_eval::ablate::SHIPPED_GRID,
        }
    }

    /// Run-directory slug, so two grids on one corpus do not overwrite
    /// each other's artifact.
    fn slug(self) -> &'static str {
        match self {
            GridName::Width => "width",
            GridName::Select => "select",
            GridName::Budget => "budget",
            GridName::Interaction => "interaction",
            GridName::Shipped => "shipped",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum)]
enum Corpus {
    Locomo,
    LmeV2Small,
    LmeV2Medium,
    #[value(name = "longmemeval-s")]
    LongmemevalS,
}

impl Corpus {
    fn slug(self) -> &'static str {
        match self {
            Corpus::Locomo => "locomo",
            Corpus::LmeV2Small => "lme_v2_small",
            Corpus::LmeV2Medium => "lme_v2_medium",
            Corpus::LongmemevalS => "longmemeval_s",
        }
    }

    fn collection(self) -> String {
        format!("myelin_{}", self.slug())
    }

    fn ledger(self) -> String {
        format!("data/{}.ledger", self.slug())
    }
}

// One `Command` is parsed per process and never moved in a hot path, so the
// size difference between `Bench` and the small variants costs nothing.
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Command {
    /// Download and checksum-pin the benchmark datasets
    Fetch,
    /// Build a memory from a dataset into a backend
    Build {
        /// Which corpus. `locomo` extracts facts; `lme-v2-small` and
        /// `lme-v2-medium` ingest episodically — see
        /// `WritePath::extract_facts` for the ~250 GPU-hour measurement
        /// behind that split.
        #[arg(long, value_enum, default_value_t = Corpus::Locomo)]
        corpus: Corpus,
        /// Qdrant collection to build into. Must be myelin_*-prefixed: the
        /// nine production collections on `big` are off limits.
        #[arg(long)]
        collection: Option<String>,
        /// SQLite ledger path.
        #[arg(long)]
        ledger: Option<String>,
        /// Ingest only the first N units (per domain, for LME-V2), for a
        /// throughput probe before committing to a long GPU window.
        #[arg(long)]
        limit: Option<usize>,
        /// LongMemEval_S only: ingest only these `question_type` strata,
        /// e.g. `single-session-preference`.
        ///
        /// Sound rather than a shortcut — that corpus writes one tenant per
        /// `question_id` and a question may only be answered from its own,
        /// so a stratum build is byte-identical to those tenants inside the
        /// full one. Applied before `--limit`.
        #[arg(long, value_delimiter = ',')]
        question_types: Option<Vec<String>>,
        /// How many model calls the write path keeps in flight.
        ///
        /// Must not exceed the reader's slot count (`llama-server -np N`, set
        /// by `MYELIN_READER_SLOTS` in `ops/big/serve-models.sh`): beyond that
        /// requests queue and the extra concurrency only adds latency. The
        /// default matches the shipped `WritePath::concurrency`.
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
        /// Directory holding the LME-V2 release files.
        #[arg(long, default_value = "/tmp/lmev2")]
        lmev2_dir: String,
        /// Repair any ledger/vector drift found at the end of the build
        /// instead of failing.
        #[arg(long)]
        repair: bool,
        /// M23 D1: mint the typed pools (events/notes) for LME-V2 into the
        /// same collection instead of the episodic pass. Off by default so
        /// the shipped store stays reproducible from the same command.
        #[arg(long)]
        pools: bool,
        /// M62: store every haystack trajectory state by state in an LME-V2
        /// ledger built before the trajectory tables existed. No model, no
        /// embedder, no Qdrant: it copies the release into the ledger,
        /// anchored on the goal records the episodic pass wrote. A fresh
        /// episodic build already does this for every trajectory it ingests.
        #[arg(long, conflicts_with = "pools")]
        trajectories: bool,
        /// Accept a LongMemEval build in which some sessions carried no
        /// parseable date.
        ///
        /// Those episodes are stamped with the *build date* instead of the
        /// conversation's, which is the M19 incident: 162,181 records all
        /// dated 2026-09-15, six milestones of temporal numbers invalidated,
        /// and a 57-minute re-ingest to repair. The build refuses by default
        /// so the next one cannot be discovered by reading a report.
        #[arg(long)]
        allow_undated: bool,
    },
    /// M50: extract Chronos-style event tuples (`10.48550/arXiv.2603.16862`
    /// §3.1) from every distinct session of a conversational corpus into a
    /// JSONL cache. Reader only — no store is touched — so it can run on a
    /// second host (`ops/bmb`) while `big`'s reader is held for a
    /// measurement, and `--shard i/n` splits one corpus across hosts.
    EventsExtract {
        #[arg(long, value_enum)]
        corpus: Corpus,
        /// The cache to append to; sessions it already holds are skipped.
        /// Defaults to data/events/<slug>.jsonl.
        #[arg(long)]
        out: Option<String>,
        /// Only the first N units (conversations, or LongMemEval questions).
        #[arg(long)]
        limit: Option<usize>,
        /// `i/n`: extract only the i-th of n disjoint shares of the distinct
        /// sessions. Every host partitions identically.
        #[arg(long, default_value = "0/1")]
        shard: String,
        /// LongMemEval_S: only these questions' haystacks (a `--questions`
        /// file, as `bench` takes).
        #[arg(long)]
        questions: Option<String>,
        /// Extraction calls in flight. At most the reader's slot count.
        #[arg(long, default_value_t = 2)]
        concurrency: usize,
    },
    /// M50: write cached events into a store as `Semantic` records derived
    /// from their session's episodic records. Embedder and store only — no
    /// reader. Build into a COPY of the corpus's store (`reindex
    /// --collection …` over a copied ledger), so the shipped store and every
    /// base run measured on it stay what they were.
    EventsBuild {
        #[arg(long, value_enum)]
        corpus: Corpus,
        /// One or more extraction caches (shards), comma-separated.
        #[arg(long, value_delimiter = ',', required = true)]
        cache: Vec<String>,
        /// LongMemEval_S: only these questions' haystacks.
        #[arg(long)]
        questions: Option<String>,
        #[arg(long)]
        collection: String,
        #[arg(long)]
        ledger: String,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 4)]
        concurrency: usize,
    },
    /// Populate the phrase↔record incidence graph over an already-built
    /// ledger. Pure SQLite: no Qdrant, no GPU, no model.
    Phrases {
        #[arg(long, value_enum, default_value_t = Corpus::Locomo)]
        corpus: Corpus,
        /// Defaults to data/<slug>.ledger.
        #[arg(long)]
        ledger: Option<String>,
        /// Records per transaction.
        #[arg(long, default_value_t = 5000)]
        batch: usize,
        /// Stop after N records, for a smoke run.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Rebuild a Qdrant collection from the ledger. Embed-only: no reader,
    /// no extraction, and every record id is preserved, so runs measured
    /// before the rebuild stay comparable after it. This is the way back from
    /// a deleted or stale collection — `build` is not, because its resume
    /// guard lives in the ledger and would skip every unit.
    Reindex {
        #[arg(long, value_enum, default_value_t = Corpus::Locomo)]
        corpus: Corpus,
        /// Defaults to data/<slug>.ledger.
        #[arg(long)]
        ledger: Option<String>,
        /// Defaults to the corpus's own collection.
        #[arg(long)]
        collection: Option<String>,
        /// Stop after N records, for a throughput probe.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// M69: write a tenant's stored trajectories as the files AgentRunbook-C's
    /// controller reads (`<out>/<id>/trajectory.json`), byte for byte as the
    /// harness writes them. With `--check`, compare every file against a
    /// harness workspace's own and fail on any difference.
    TrajectoriesExport {
        #[arg(long, default_value = "data/lme_v2_small.ledger")]
        ledger: String,
        #[arg(long, default_value = "lme_v2_small")]
        namespace: String,
        /// e.g. `lme_v2_small/web`.
        #[arg(long)]
        tenant: String,
        #[arg(long)]
        out: String,
        /// A harness `memory_workspace/shared/trajectories` directory.
        #[arg(long)]
        check: Option<String>,
    },
    /// Apply M42's decline-recovery pass to a finished run, writing a new
    /// one. The mechanism is a pure function of the first response, so rows
    /// that answered come back byte-identical and the non-firing control is
    /// exact by construction rather than empirical.
    CommitArm {
        /// The base run to pair against.
        #[arg(long)]
        run: String,
        /// LongMemEval_S dataset, for each question's `<today>`.
        #[arg(long, default_value = "data/longmemeval_s.json")]
        dataset: String,
        #[arg(long)]
        out: String,
        /// M45: draw this many seeded samples per declining row instead of
        /// M42's one greedy pass, cluster them by meaning, and commit only
        /// the majority. Omit for M42's arm exactly.
        #[arg(long, requires = "agree")]
        samples: Option<usize>,
        /// First seed; sample i uses seed + i. Recorded on the artifact.
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Share of samples the majority cluster must hold to commit.
        /// **Calibrated on a split that is not the reported population**,
        /// never tuned; the samples are recorded so it can be re-applied.
        #[arg(long, requires = "samples")]
        agree: Option<f64>,
        /// M61: the grounded second pass — cite the memories that state the
        /// answer about the entity the question names, then answer from those.
        #[arg(long, conflicts_with = "samples")]
        grounded: bool,
    },
    /// Score LoCoMo end-to-end: retrieve, read, and grade the answer with
    /// a deterministic scorer (no LLM judge). See `bench.rs`.
    Bench {
        /// Which corpus to score.
        #[arg(long, value_enum, default_value_t = BenchCorpus::Locomo)]
        corpus: BenchCorpus,
        /// Defaults follow --corpus when left unset.
        #[arg(long)]
        dataset: Option<String>,
        #[arg(long)]
        collection: Option<String>,
        #[arg(long)]
        ledger: Option<String>,
        #[arg(long, default_value_t = 6)]
        k: usize,
        /// Total token ceiling for the composed evidence set. 4096 unless
        /// widened; every run before M23 used exactly 4096.
        #[arg(long)]
        budget_tokens: Option<usize>,
        /// Per-channel candidate depth before fusion. Overrides
        /// `RetrieveConfig::default()`'s 50 (M23 width arms).
        #[arg(long)]
        prefetch_limit: Option<u64>,
        /// How many fused candidates the reranker sees. Must stay ≤
        /// prefetch_limit: the reranked pool cannot be wider than its
        /// prefetch. Overrides `RetrieveConfig::default()`'s 25.
        #[arg(long)]
        rerank_depth: Option<usize>,
        /// `recall` is the fast path; `investigate` runs the agentic loop.
        #[arg(long, default_value = "recall")]
        mode: String,
        /// `investigate` only. Default matches `InvestigateConfig`.
        #[arg(long, default_value_t = 2)]
        max_steps: usize,
        /// Stop after N questions. Omit for all 1,986.
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long)]
        out: Option<String>,
        /// Keep the rows an earlier attempt wrote to `--out` and score only
        /// the questions it did not reach. The inherited count is recorded
        /// on the run as `resumed_rows`.
        #[arg(long)]
        resume: bool,
        /// Fuse the PPR channel over the phrase↔record graph (`--corpus`'s
        /// ledger must have been through `myelin-eval phrases`).
        #[arg(long)]
        graph: bool,
        /// Emit evidence oldest-first instead of `bookend`'s relevance
        /// interleave.
        #[arg(long)]
        chronological: bool,
        /// State each `[timeline]` entry's distance from the question's day
        /// — `28 days ago; 4 weeks` — so a duration question is a lookup
        /// rather than a subtraction the reader does itself (M46). Zero
        /// model calls; inert on questions that get no timeline.
        #[arg(long)]
        timeline_ago: bool,
        /// Give the reader a `<today>` reference date. LongMemEval_S always
        /// carries one; this adds LoCoMo's last session date.
        #[arg(long)]
        question_date: bool,
        /// Tell the reader to answer recommendation questions from the
        /// user's stated preferences (M20 arm B).
        #[arg(long)]
        profile_clause: bool,
        /// Select the composed evidence for joint coverage of the question
        /// instead of by independent rank: maximal marginal relevance at
        /// this lambda, 1.0 pure relevance and 0.0 pure diversity (M21 arm A).
        #[arg(long)]
        mmr: Option<f32>,
        /// Ask the model which of the reranked candidates jointly answer the
        /// question and put those first (M21 arm B). A ceiling probe: it
        /// costs a model call per query, which `PLAN.md` §7.1 forbids in
        /// `recall`, so it can never become a `recall` default.
        #[arg(long)]
        select_sufficient: bool,
        /// Rerank the whole accumulated pool against the original question
        /// once, after the last step, before compose (M23 A2). The pool's
        /// stored scores are cross-encoder logits from *different* probe
        /// queries and are not mutually comparable; this is the one pass
        /// that is question-conditioned. `--mode investigate` only.
        #[arg(long)]
        rerank_pool: bool,
        /// Replace the bare insufficiency statement with an explicit
        /// analysis of the question's premise (M23 A3). Implies the
        /// insufficiency gate, because the analysis is what the gate emits.
        #[arg(long)]
        premise: bool,
        /// Verify what the question assumes against the composed memories
        /// and append a `[premise]` line only when a memory contradicts it
        /// (M47). Silence appends nothing — the one rule that separates it
        /// from `--premise`, which M35 measured at −8.75. One model call.
        #[arg(long)]
        premise_check: bool,
        /// Let the reflect gate aim its next probe at a record kind:
        /// raw | event | note (M23 D2). Inert against a store with no typed
        /// pools — build one with `build --pools`.
        #[arg(long)]
        typed_probes: bool,
        /// Decompose the question into follow-ups, answer each from the
        /// composed evidence, and append them as one additive `[notes]`
        /// item (M39, self-ask). One model call per query;
        /// investigate-only. Aimed at the measured compositionality gap:
        /// with every gold session retrieved, accuracy is 79.3% when one
        /// fact answers the question and 54.3% when two must be combined.
        #[arg(long)]
        self_ask: bool,
        /// State what EVERY composed memory contributes to the question and
        /// append the contributions as one additive `[notes]` item (M40).
        /// `--self-ask` with the entry count fixed by the schema instead of
        /// chosen by the model: M39 measured that on two-fact questions the
        /// model produced fewer than two follow-ups 70% of the time while
        /// holding 7 or 8 memories. One model call per query.
        ///
        /// Ships **on** for `investigate` since M43 (+5.80 judged with
        /// `--digest-dates`). The bench CLI still defaults every switch off
        /// so an arm names exactly what it runs: pass
        /// `--item-digest --digest-dates` for the shipped configuration.
        #[arg(long)]
        item_digest: bool,
        /// Prefix each digest line with the `(YYYY-MM-DD)` of the memory it
        /// came from (M41). Inert without `--item-digest`; costs no extra
        /// model call, since the stamp is already on the composed item.
        /// M40 measured omitting it at -4.2 on `knowledge-update`; M43
        /// measured its marginal at +3.40 [+1.2, +5.8] and shipped it on.
        #[arg(long)]
        digest_dates: bool,
        /// When the reader declines, ask once more with the two decisions
        /// split into separate schema fields, so it must write a candidate
        /// answer before it may assert the evidence is absent (M42).
        ///
        /// Measured motivation: the reader declines on 63 questions that are
        /// not abstention problems and scores zero on all of them; on 48 the
        /// evidence contained every gold session. Costs one extra call on the
        /// ~20% of rows that decline. Abstention stays reachable — the model
        /// keeps its own `evidence_absent` escape hatch.
        #[arg(long)]
        commit_answer: bool,
        /// Let the digest mark a memory as not bearing on the question and
        /// drop its line (M43). Inert without `--item-digest`; no extra call.
        ///
        /// M40's arm emitted 265 prose negations across 2,355 digest lines —
        /// 11.3%, on 26% of noted rows — despite being told to write the
        /// literal `nothing`. Rows whose note carries one gained −0.9; rows
        /// with none gained +4.1.
        #[arg(long)]
        digest_relevance: bool,
        /// Type each digest entry `answers` / `context` / `irrelevant` and
        /// drop only the last (M48). Chain-of-Note's three types; M43's
        /// boolean collapsed the first two and lost 245 rows their note.
        /// An alternative to `--digest-relevance`. Inert without
        /// `--item-digest`; no extra call.
        #[arg(long, conflicts_with = "digest_relevance")]
        digest_role: bool,
        /// Let the reader reason before answering (M44 R1): a
        /// `{reasoning, answer, evidence_absent}` schema in that field order,
        /// with the completion ceiling raised from 160 to 480.
        ///
        /// Until this exists every myelin number was produced by a reader
        /// told "Do not explain." with thinking off — a configuration no
        /// published comparator uses.
        #[arg(long)]
        reader_reasoning: bool,
        /// Let the reader think natively (M44 R2): `enable_thinking: true`
        /// under the server's `--reasoning-budget` (1,024 tokens, verified
        /// by a probe before the run), sampled at the Qwen3 report's
        /// thinking-mode setting. An alternative to `--reader-reasoning`,
        /// and it needs `--reader-seed`.
        #[arg(long, conflicts_with = "reader_reasoning")]
        reader_thinking: bool,
        /// The sampling seed for `--reader-thinking`, recorded on the run.
        /// Two seeds make one arm: the CI must carry sampling noise.
        #[arg(long, requires = "reader_thinking")]
        reader_seed: Option<u64>,
        /// Declare that the reader server was started with
        /// `MYELIN_READER_THINK_MESSAGE` (M44 R2b): a capped trace is closed
        /// with Qwen's "I have to give the answer now" instead of a bare
        /// tag. Recorded on the artifact; the harness cannot verify it.
        #[arg(long, requires = "reader_thinking")]
        reader_think_message: bool,
        /// Tell the reader to answer a question built on an assumption the
        /// memories do not support with "I don't know." first and the
        /// correction after (M57).
        #[arg(long)]
        reader_premise_clause: bool,
        /// Tell the reader to give its most likely answer whenever the
        /// memories bear on the question, and to decline only when nothing
        /// in them does (M59).
        #[arg(long)]
        reader_best_guess: bool,
        /// L1: a fact and the episode it was abstracted from count as one
        /// memory in compose; the higher-ranked copy keeps the slot.
        #[arg(long)]
        dedupe_lineage: bool,
        /// M64: relative-date resolutions right after their phrase, in words
        /// (`ComposeConfig::inline_dates`), instead of appended in ISO form.
        #[arg(long)]
        inline_dates: bool,
        /// M66: episodes emitted as turn windows of this radius, ranked turn
        /// by turn with the cross-encoder (`RetrieveConfig::turn_windows`).
        /// `recall` mode only; refuses `--select-sufficient`.
        #[arg(long)]
        turn_windows: Option<usize>,
        /// M72: a counting or summing question is retrieved with this `k`
        /// (and `--aggregation-budget-tokens`) instead of the run's.
        #[arg(long, requires = "aggregation_budget_tokens")]
        aggregation_k: Option<usize>,
        #[arg(long, requires = "aggregation_k")]
        aggregation_budget_tokens: Option<usize>,
        /// M73b: a ledger of events (`events-build`). When the question names
        /// a past day, the events dated inside it are ranked by the
        /// cross-encoder and the top 3 appended as an `[events]` block.
        #[arg(long)]
        events_ledger: Option<String>,
        /// M20b: a ledger of profile records. On an advice request, the
        /// user's dispositions are ranked by the cross-encoder and the top 8
        /// appended as a `[profile]` block.
        #[arg(long)]
        profile_ledger: Option<String>,
        /// Cap how many `Untrusted` records the composed set may contain
        /// (M23 B1). A ceiling, not an exclusion: the quota never drops
        /// untrusted evidence to zero and never drops a trusted record.
        /// Measured here for its utility cost; `attack --live` measures the
        /// same switch for ASR.
        #[arg(long)]
        untrusted_max: Option<usize>,
        /// Split the question into at most N sub-queries and retrieve for
        /// each, fusing them into the same RRF call as the original (M24).
        /// One model call per query, so it can never ship on for `recall`
        /// (§7.1); `investigate` decomposes every probe.
        #[arg(long)]
        decompose: Option<usize>,
        /// Score only these category codes, for a stratum arm. LoCoMo: 1
        /// multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.
        /// LongMemEval_S: 1 ss-user, 2 ss-assistant, 3 ss-preference,
        /// 4 multi-session, 5 temporal-reasoning, 6 knowledge-update.
        #[arg(long, value_delimiter = ',')]
        categories: Option<Vec<u8>>,
        /// Score only the question ids listed in this file, one per line
        /// (`#` comments allowed). A pre-registered population, e.g. M50's
        /// stratified pilot. Every id must exist in the corpus.
        #[arg(long)]
        questions: Option<String>,
        /// Which scorer `score` reports. Both columns are always written on
        /// every row, so a run stays readable under either. Defaults follow
        /// `--corpus`.
        #[arg(long, value_enum)]
        scorer: Option<myelin_eval::bench::Scorer>,
    },
    /// Run the MINJA-style poisoning attack suite (EVALUATION.md §7)
    Attack {
        /// Also run E1/E2, which need a live store and a GPU: they build
        /// two scratch collections through the real write path.
        #[arg(long)]
        live: bool,
        /// Where scratch ledgers go. Deleted when the run finishes.
        #[arg(long, default_value = "data")]
        ledger_dir: String,
        /// Where to serialise the `--live` sweep: writes
        /// `<dir>/attack_live.json`, which is the artifact `standing` reads
        /// for G3. Without it a 54-minute run leaves nothing on disk but
        /// stdout.
        #[arg(long)]
        out: Option<String>,
    },
    /// Run the injection adjudicator over LoCoMo's real episodes and report
    /// the false-positive rate (M15). Reader only — no store.
    AdjudicateProbe {
        #[arg(long, default_value = "data/locomo10.json")]
        dataset: String,
        /// Stop after N episodes, for a cost probe before the full pass.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Run the M4 retrieval ablation against an already-built memory
    Ablate {
        #[arg(long, default_value = "data/locomo10.json")]
        dataset: String,
        #[arg(long, default_value = "myelin_locomo")]
        collection: String,
        #[arg(long, default_value = "data/locomo.ledger")]
        ledger: String,
        /// Dev-split size in *conversations*. Splitting by question would
        /// leak: questions from one conversation share one memory.
        #[arg(long, default_value_t = 5)]
        units: usize,
        /// Evidence-set size. `EVALUATION.md` §8 row 4 sweeps this.
        #[arg(long, default_value_t = 6)]
        k: usize,
        /// Cap questions per arm, for a quick smoke of the harness itself.
        #[arg(long)]
        limit: Option<usize>,
        /// Score the held-out conversations (everything after --units)
        /// instead of the dev split. Use once, to confirm a decision the
        /// dev table already made.
        #[arg(long)]
        holdout: bool,
        /// Instead of the channel ablation, sweep `investigate`'s step
        /// budget and report the marginal value of each extra step (M7).
        #[arg(long, value_delimiter = ',')]
        steps: Option<Vec<usize>>,
        /// Instead of the channel ablation, sweep `prefetch_limit` ×
        /// `rerank_depth` and report emitted recall beside the reranked
        /// pool's recall — the ceiling truncation is measured against
        /// (M25). Needs an embedder, a reranker and Qdrant; never a reader.
        #[arg(long)]
        width: bool,
        /// Which corpus the width sweep scores. `locomo` resolves record
        /// ids to `dia_id` turns through I4 lineage; `longmemeval-s`
        /// matches the `has_answer` turn's text prefix. Ignored by the
        /// channel ablation and the step curve, which are LoCoMo-only.
        #[arg(long, default_value = "locomo")]
        corpus: String,
        /// Which grid `--width` sweeps.
        #[arg(long, value_enum, default_value_t = GridName::Width)]
        grid: GridName,
        /// L1's lineage-aware dedupe in compose, for the width sweep.
        #[arg(long, requires = "width")]
        dedupe_lineage: bool,
        /// M66's turn windows, of this radius, for the width sweep.
        #[arg(long, requires = "width")]
        turn_windows: Option<usize>,
    },
    /// Re-score a finished bench run under a different scorer. Pure CPU:
    /// `response_raw` and `answer_gold` are on disk, so no reader call and no
    /// GPU are needed to apply a scorer change to every historical run.
    Rescore {
        /// A run directory written by `bench`.
        #[arg(long)]
        run: String,
        #[arg(long, value_enum, default_value_t = myelin_eval::bench::Scorer::Temporal)]
        scorer: myelin_eval::bench::Scorer,
        /// Defaults to runs/rescored/<source-basename>_<scorer>.
        #[arg(long)]
        out: Option<String>,
    },
    /// Grade a finished run's answers with the local reader (M14 scorer
    /// validation). Needs the reader only — no embedder, reranker, or store.
    Judge {
        #[arg(long)]
        run: String,
        /// Restrict to one category. LoCoMo 2 is the temporal stratum.
        #[arg(long)]
        category: Option<u8>,
        #[arg(long)]
        limit: Option<usize>,
        /// Reuse verdicts from another run for rows whose answer is
        /// byte-identical to that run's.
        ///
        /// A verdict is a function of the answer, so re-grading an unchanged
        /// answer can only introduce disagreement. M42 measured it: 469 of 500
        /// responses were identical to the base and re-judging flipped two,
        /// moving the control that should be exactly zero to −0.43 against a
        /// +1.80 effect. Seeding from the base makes the control exact.
        #[arg(long)]
        seed: Option<String>,
    },
    /// Audit a vendored LongMemEval-V2 run: for every answerable question the
    /// harness scored wrong, was the answer in the evidence? (M16). Reader only.
    EvidenceAudit {
        /// A run directory written by `adapters/run_myelin.py`.
        #[arg(long)]
        run: String,
        /// Judge at most the first N answerable rows, for a cost probe.
        /// Verdicts cache per row, so an unlimited re-run judges only what
        /// is missing and an interrupted pass resumes.
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Did the reader ever see the gold evidence? Joins a `bench` run's
    /// composed evidence against the corpus's own per-turn annotation
    /// (LongMemEval_S `has_answer`, LoCoMo `qa[].evidence`) and reports
    /// gold-unit recall per category. Fully offline: no store, no reader (M21).
    Coverage {
        /// A run directory written by `myelin-eval bench` or `rescore`.
        #[arg(long)]
        run: String,
        /// Defaults to `data/longmemeval_s.json` or `data/locomo10.json`
        /// according to the run's own `corpus` field.
        #[arg(long)]
        dataset: Option<String>,
    },
    /// Join docs/sota/registry.json against the run artifacts and report where
    /// we stand, with a comparability verdict per row
    Standing {
        #[arg(long, default_value = "docs/sota/registry.json")]
        registry: String,
        #[arg(long, default_value = "runs")]
        runs: String,
        #[arg(long, default_value = "runs/standing")]
        out: String,
        /// Exit non-zero when a `gate: true` row is unsupported or not beaten
        #[arg(long)]
        gate: bool,
        #[arg(long, default_value = ".venv/bin/python")]
        python: String,
    },
    /// Compare today's artifacts against our own pinned floor and fail on a
    /// regression. `standing` compares us to the literature; this compares
    /// us to ourselves, which is the check that was missing when the shipped
    /// defaults lost 3.3 points between M16 and M22 without anyone noticing.
    Ratchet {
        #[arg(long, default_value = "runs")]
        runs: String,
        #[arg(long, default_value = "docs/sota/progression.json")]
        baseline: String,
        /// Raise the floor to today's quotable values first. Only ever
        /// moves a pin in the improving direction.
        #[arg(long)]
        update: bool,
        /// Also fail when a pinned metric's best artifact is an arm, is
        /// incomplete, or does not record its operating point.
        #[arg(long)]
        strict: bool,
        /// Points of slack before a drop counts, in the metric's own units.
        /// Zero by default: a floor with give is not a floor.
        #[arg(long, default_value_t = 0.0)]
        tolerance: f64,
        #[arg(long, default_value = ".venv/bin/python")]
        python: String,
    },
    /// Package a leaderboard submission
    Package,
}

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Command::Fetch => "fetch",
            Command::Build { .. } => "build",
            Command::EventsExtract { .. } => "events-extract",
            Command::EventsBuild { .. } => "events-build",
            Command::Phrases { .. } => "phrases",
            Command::Reindex { .. } => "reindex",
            Command::TrajectoriesExport { .. } => "trajectories-export",
            Command::CommitArm { .. } => "commit-arm",
            Command::Bench { .. } => "bench",
            Command::Attack { .. } => "attack",
            Command::AdjudicateProbe { .. } => "adjudicate-probe",
            Command::Ablate { .. } => "ablate",
            Command::Rescore { .. } => "rescore",
            Command::Judge { .. } => "judge",
            Command::EvidenceAudit { .. } => "evidence-audit",
            Command::Coverage { .. } => "coverage",
            Command::Standing { .. } => "standing",
            Command::Ratchet { .. } => "ratchet",
            Command::Package => "package",
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Fetch => fetch().await,
        Command::Build {
            corpus,
            ref collection,
            ref ledger,
            limit,
            ref question_types,
            concurrency,
            ref lmev2_dir,
            pools,
            trajectories,
            repair,
            allow_undated,
        } => {
            build_cmd(
                corpus,
                collection.as_deref(),
                ledger.as_deref(),
                limit,
                question_types.as_deref(),
                concurrency,
                lmev2_dir,
                pools,
                trajectories,
                repair,
                allow_undated,
            )
            .await
        }
        Command::EventsExtract {
            corpus,
            ref out,
            limit,
            ref shard,
            ref questions,
            concurrency,
        } => events_extract_cmd(corpus, out.as_deref(), limit, shard, questions.as_deref(), concurrency).await,
        Command::EventsBuild {
            corpus,
            ref cache,
            ref questions,
            ref collection,
            ref ledger,
            limit,
            concurrency,
        } => events_build_cmd(corpus, cache, questions.as_deref(), collection, ledger, limit, concurrency).await,
        Command::Phrases {
            corpus,
            ref ledger,
            batch,
            limit,
        } => {
            let path = ledger.clone().unwrap_or_else(|| corpus.ledger());
            eprintln!(
                "backfilling incidence for namespace {} from {path}",
                corpus.slug()
            );
            let stats = myelin_eval::phrases::backfill_phrases(
                Path::new(&path),
                corpus.slug(),
                batch,
                limit,
            )
            .await?;
            println!(
                "phrases {}: {} records, {} edges, {} distinct phrases",
                corpus.slug(),
                stats.records,
                stats.edges,
                stats.distinct_phrases
            );
            Ok(())
        }
        Command::TrajectoriesExport {
            ref ledger,
            ref namespace,
            ref tenant,
            ref out,
            ref check,
        } => trajectories_export(ledger, namespace, tenant, out, check.as_deref()).await,
        Command::Reindex {
            corpus,
            ref ledger,
            ref collection,
            limit,
        } => {
            let cfg = MyelinConfig::load().context("load myelin config")?;
            let path = ledger.clone().unwrap_or_else(|| corpus.ledger());
            let collection = collection.clone().unwrap_or_else(|| corpus.collection());
            let report =
                myelin_eval::reindex::reindex(&cfg, Path::new(&path), &collection, limit).await?;
            println!(
                "reindex {collection}: {} of {} live records indexed",
                report.indexed, report.expected
            );
            // A rebuild that quietly indexed nothing must not exit 0: that is
            // the exact shape of the `build`-against-a-surviving-ledger trap.
            anyhow::ensure!(
                limit.is_some() || report.is_complete(),
                "incomplete rebuild: {} of {} live records reached the index",
                report.indexed,
                report.expected
            );
            Ok(())
        }
        Command::CommitArm {
            ref run,
            ref dataset,
            ref out,
            samples,
            seed,
            agree,
            grounded,
        } => {
            let cfg = MyelinConfig::load().context("load myelin config")?;
            let pass = match (samples, agree, grounded) {
                (Some(n), Some(a), false) => myelin_eval::commit_arm::Pass::Consensus(
                    myelin_eval::bench::Consensus::new(n, seed, a)?,
                ),
                (None, None, true) => myelin_eval::commit_arm::Pass::Grounded,
                (None, None, false) => myelin_eval::commit_arm::Pass::Greedy,
                _ => anyhow::bail!("--samples/--agree and --grounded are different second passes; pass one"),
            };
            let report = myelin_eval::commit_arm::run(
                &cfg,
                Path::new(run),
                Path::new(dataset),
                Path::new(out),
                pass,
            )
            .await?;
            println!(
                "commit-arm {out}: {} rows, {} declined, {} committed, {} untouched",
                report.rows, report.fired, report.committed, report.untouched
            );
            anyhow::ensure!(
                report.is_consistent(),
                "arm is inconsistent: {} untouched + {} committed != {} rows",
                report.untouched,
                report.committed,
                report.rows
            );
            Ok(())
        }
        Command::Attack {
            live,
            ref ledger_dir,
            ref out,
        } => {
            let (e3, e5) = myelin_eval::attack::run_offline()?;
            myelin_eval::attack::print_gate_report(&e3, &e5);
            anyhow::ensure!(
                e3.catch_rate() >= 0.90,
                "E3 gate: catch rate {:.1}% is below the 90% floor",
                e3.catch_rate() * 100.0
            );
            anyhow::ensure!(
                e5.admitted == 0,
                "E5 gate: {} poisoned records admitted at first-party trust",
                e5.admitted
            );
            anyhow::ensure!(
                live || out.is_none(),
                "--out serialises the --live sweep; the offline E3/E5 gates have \
                 no per-condition artifact. Pass --live."
            );
            if live {
                let run = myelin_eval::attack_live::run(Path::new(ledger_dir), &[3, 6, 10]).await?;
                myelin_eval::attack_live::print(&run);
                if let Some(dir) = out {
                    let dir = Path::new(dir);
                    std::fs::create_dir_all(dir)
                        .with_context(|| format!("create {}", dir.display()))?;
                    let path = dir.join("attack_live.json");
                    std::fs::write(&path, serde_json::to_string_pretty(&run)?)
                        .with_context(|| format!("write {}", path.display()))?;
                    println!("\nwrote {}", path.display());
                }
            } else {
                println!("\nE1/E2 skipped (pass --live; they need a GPU and a live store).");
            }
            Ok(())
        }
        Command::AdjudicateProbe { ref dataset, limit } => {
            adjudicate_probe_cmd(dataset, limit).await
        }
        Command::Ablate {
            ref dataset,
            ref collection,
            ref ledger,
            units,
            k,
            limit,
            ref steps,
            width,
            ref corpus,
            grid,
            holdout,
            dedupe_lineage,
            turn_windows,
        } => {
            ablate_cmd(
                dataset,
                collection,
                ledger,
                units,
                k,
                limit,
                steps.as_deref(),
                width,
                corpus,
                grid,
                holdout,
                dedupe_lineage,
                turn_windows,
            )
            .await
        }
        Command::Bench {
            corpus,
            ref dataset,
            ref collection,
            ref ledger,
            k,
            budget_tokens,
            prefetch_limit,
            rerank_depth,
            ref mode,
            max_steps,
            limit,
            resume,
            ref out,
            graph,
            chronological,
            question_date,
            timeline_ago,
            profile_clause,
            mmr,
            select_sufficient,
            rerank_pool,
            premise,
            premise_check,
            typed_probes,
            self_ask,
            item_digest,
            digest_dates,
            commit_answer,
            digest_relevance,
            digest_role,
            reader_reasoning,
            reader_thinking,
            reader_seed,
            reader_think_message,
            reader_premise_clause,
            reader_best_guess,
            ref events_ledger,
            ref profile_ledger,
            dedupe_lineage,
            inline_dates,
            turn_windows,
            aggregation_k,
            aggregation_budget_tokens,
            untrusted_max,
            decompose,
            ref categories,
            ref questions,
            scorer,
        } => {
            let question_ids = match questions {
                Some(path) => read_question_ids(path)?,
                None => Vec::new(),
            };
            bench_cmd(
                corpus,
                dataset.as_deref(),
                collection.as_deref(),
                ledger.as_deref(),
                k,
                budget_tokens,
                prefetch_limit,
                rerank_depth,
                mode,
                max_steps,
                limit,
                out.as_deref(),
                resume,
                myelin_eval::bench::BenchSwitches {
                    graph,
                    chronological,
                    question_date,
                    timeline_ago,
                    profile_clause,
                    mmr,
                    select_sufficient,
                    rerank_pool,
                    premise,
                    premise_check,
                    typed_probes,
                    self_ask,
                    item_digest,
                    digest_dates,
                    commit_answer,
                    digest_relevance,
                    digest_role,
                    reader_reasoning,
                    reader_thinking,
                    reader_seed,
                    reader_think_message,
                    reader_premise_clause,
                    reader_best_guess,
                    events_ledger: events_ledger.clone(),
                    profile_ledger: profile_ledger.clone(),
                    dedupe_lineage,
                    inline_dates,
                    turn_windows,
                    aggregation_k,
                    aggregation_budget_tokens,
                    commit_grounded: false,
                    untrusted_max,
                    decompose,
                    categories: categories.clone().unwrap_or_default(),
                    question_ids,
                },
                scorer,
            )
            .await
        }
        Command::Rescore {
            ref run,
            scorer,
            ref out,
        } => rescore_cmd(run, scorer, out.as_deref()),
        Command::Judge {
            ref seed,
            ref run,
            category,
            limit,
        } => judge_cmd(run, category, limit, seed.as_deref()).await,
        Command::EvidenceAudit { ref run, limit } => evidence_audit_cmd(run, limit).await,
        Command::Coverage {
            ref run,
            ref dataset,
        } => {
            let report =
                myelin_eval::coverage::run(Path::new(run), dataset.as_deref().map(Path::new))?;
            myelin_eval::coverage::print_table(&report);
            Ok(())
        }
        Command::Standing {
            ref registry,
            ref runs,
            ref out,
            gate,
            ref python,
        } => myelin_eval::standing::run(
            Path::new(registry),
            Path::new(runs),
            Path::new(out),
            gate,
            python,
        )
        .map(|_| ()),
        Command::Ratchet {
            ref runs,
            ref baseline,
            update,
            strict,
            tolerance,
            ref python,
        } => myelin_eval::ratchet::run(
            Path::new(runs),
            Path::new(baseline),
            python,
            update,
            strict,
            tolerance,
        )
        .map(|_| ()),
        rest => {
            println!("{}: not implemented (milestone M5+)", rest.name());
            Ok(())
        }
    }
}

/// Fetch all pinned datasets into `data/` and print a verification table.
///
/// Each file is checksummed against its pinned SHA-256; a mismatch is a hard
/// error rather than a silent downgrade (`PLAN.md` §1.1 — reproducibility
/// requires a pinned corpus).
async fn fetch() -> anyhow::Result<()> {
    let data_dir = Path::new("data");

    let path = datasets::fetch_pinned(&datasets::LOCOMO, data_dir).await?;
    let digest = datasets::sha256_file(&path)?;

    // LongMemEval_S is 278 MB and only needed for G2's second number, but it
    // is fetched here rather than on demand so one command produces the whole
    // pinned corpus set and a checksum mismatch surfaces before a GPU window
    // is spent on it.
    let lme_s = datasets::fetch_pinned(&datasets::LONGMEMEVAL_S, data_dir).await?;
    let lme_s_items = longmemeval::load(&lme_s)?;
    println!(
        "longmemeval_s   {:>6} questions  sha256 {}",
        lme_s_items.len(),
        datasets::sha256_file(&lme_s)?
    );

    let conversations = locomo::load(&path)?;
    let counts = locomo::qa_counts(&conversations);

    println!("file:          {}", datasets::LOCOMO.name);
    println!("url:           {}", datasets::LOCOMO.url);
    println!("bytes:         {}", datasets::LOCOMO.bytes);
    println!("sha256:        {} (verified)", digest);
    println!("conversations: {}", conversations.len());

    println!("qa counts:");
    for (cat, n) in &counts.by_category {
        println!("  category {}: {}", cat, n);
    }
    println!("  total:        {}", counts.total);
    println!(
        "  comparable:   {}  (category 5 dropped)",
        counts.comparable_1540
    );
    println!("  adversarial:  {}  (category 5)", counts.adversarial);

    Ok(())
}
/// M3: drive LoCoMo through the write path and report records/unit, tokens
/// and wall time.
/// A question-id list: one id per line, blank lines and `#` comments
/// skipped. An empty list or a repeated id is refused — both are typos that
/// would silently change the population.
fn read_question_ids(path: &str) -> anyhow::Result<Vec<String>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let ids: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    anyhow::ensure!(!ids.is_empty(), "{path} lists no question ids");
    let distinct: std::collections::HashSet<&String> = ids.iter().collect();
    anyhow::ensure!(distinct.len() == ids.len(), "{path} repeats a question id");
    Ok(ids)
}

/// The session slots of a conversational corpus, for the M50 passes.
fn event_slots(
    corpus: Corpus,
    limit: Option<usize>,
    questions: Option<&str>,
) -> anyhow::Result<Vec<myelin_eval::events::SessionSlot>> {
    let ids = match questions {
        Some(path) => read_question_ids(path)?,
        None => Vec::new(),
    };
    match corpus {
        Corpus::Locomo => {
            anyhow::ensure!(ids.is_empty(), "--questions selects LongMemEval_S haystacks; LoCoMo has ten conversations, use --limit");
            myelin_eval::events::locomo_slots(Path::new("data/locomo10.json"), limit)
        }
        Corpus::LongmemevalS => {
            myelin_eval::events::longmemeval_slots(Path::new("data/longmemeval_s.json"), limit, &ids)
        }
        other => anyhow::bail!(
            "events are a conversational-corpus pass (LoCoMo, LongMemEval_S); {} is UI \
             trajectories, whose event pool is `build --pools`",
            other.slug()
        ),
    }
}

async fn events_extract_cmd(
    corpus: Corpus,
    out: Option<&str>,
    limit: Option<usize>,
    shard: &str,
    questions: Option<&str>,
    concurrency: usize,
) -> anyhow::Result<()> {
    let shard = myelin_eval::events::parse_shard(shard)?;
    let out = out.map_or_else(|| format!("data/events/{}.jsonl", corpus.slug()), str::to_string);
    let slots = event_slots(corpus, limit, questions)?;
    let r = myelin_eval::events::extract(&slots, Path::new(&out), shard, concurrency).await?;
    eprintln!(
        "events-extract {}: {} extracted ({} events), {} cached, {} failed, {:.0}s -> {out}",
        corpus.slug(),
        r.extracted,
        r.events,
        r.cached,
        r.failed,
        r.wall_secs
    );
    anyhow::ensure!(
        r.failed == 0,
        "{} sessions failed extraction and are not in the cache; re-run the same command to \
         retry only those",
        r.failed
    );
    Ok(())
}

async fn events_build_cmd(
    corpus: Corpus,
    cache: &[String],
    questions: Option<&str>,
    collection: &str,
    ledger: &str,
    limit: Option<usize>,
    concurrency: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        collection.starts_with("myelin_") && collection != corpus.collection(),
        "refusing to write events into {collection:?}: build them into a myelin_* COPY of \
         {}, so the shipped store and its base runs stay unchanged",
        corpus.collection()
    );
    anyhow::ensure!(
        ledger != corpus.ledger(),
        "refusing to write events into the shipped ledger {ledger}; copy it first"
    );
    let slots = event_slots(corpus, limit, questions)?;
    let map = myelin_eval::events::load_cache(cache)?;
    let r = myelin_eval::events::build(&slots, &map, collection, Path::new(ledger), concurrency).await?;
    eprintln!(
        "events-build {}: {} slots ({} resumed), {} events (stated {}, unresolved {}, said {}), \
         {} records written, {:.0}s",
        corpus.slug(),
        r.slots,
        r.resumed,
        r.events,
        r.stated,
        r.unresolved,
        r.said,
        // `episodes`, not `added`: with extraction off the write path
        // stores each event as its own record and never reaches the
        // consolidation step that `added`/`duplicates` count. The first
        // LoCoMo build printed "0 added" over 939 records it had written.
        r.total.episodes,
        r.wall_secs
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn build_cmd(
    corpus: Corpus,
    collection: Option<&str>,
    ledger: Option<&str>,
    limit: Option<usize>,
    question_types: Option<&[String]>,
    concurrency: usize,
    lmev2_dir: &str,
    pools: bool,
    trajectories_only: bool,
    repair: bool,
    allow_undated: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !trajectories_only || matches!(corpus, Corpus::LmeV2Small | Corpus::LmeV2Medium),
        "--trajectories stores agent trajectories; {} has none",
        corpus.slug()
    );
    let collection = collection.map_or_else(|| corpus.collection(), str::to_string);
    let ledger = ledger.map_or_else(|| corpus.ledger(), str::to_string);
    anyhow::ensure!(
        collection.starts_with("myelin_"),
        "refusing to build into {collection:?}: collections must be myelin_*-prefixed \
         so a typo cannot touch the production collections on big"
    );
    anyhow::ensure!(
        question_types.is_none() || corpus == Corpus::LongmemevalS,
        "--question-types is LongMemEval_S-only; no other corpus has question_type strata"
    );

    let report = match corpus {
        Corpus::Locomo => {
            let data = Path::new("data/locomo10.json");
            anyhow::ensure!(
                data.exists(),
                "missing {}; run `myelin-eval fetch` first",
                data.display()
            );
            eprintln!("ingesting LoCoMo -> collection {collection}, ledger {ledger}");
            build_locomo(data, &collection, Path::new(&ledger), limit, repair).await?
        }
        Corpus::LongmemevalS => {
            let data = Path::new("data/longmemeval_s.json");
            anyhow::ensure!(
                data.exists(),
                "missing {}; run `myelin-eval fetch` first",
                data.display()
            );
            eprintln!("ingesting LongMemEval_S -> collection {collection}, ledger {ledger}");
            myelin_eval::build::build_longmemeval_s(
                data,
                &collection,
                Path::new(&ledger),
                limit,
                question_types,
                concurrency,
                repair,
            )
            .await?
        }
        Corpus::LmeV2Small | Corpus::LmeV2Medium => {
            let dir = Path::new(lmev2_dir);
            let trajectories = dir.join("trajectories.jsonl");
            let questions = dir.join("questions.jsonl");
            let haystack = dir
                .join("haystacks")
                .join(format!("{}.json", corpus.slug()));
            for p in [&trajectories, &questions, &haystack] {
                anyhow::ensure!(p.exists(), "missing {}", p.display());
            }
            if trajectories_only {
                eprintln!(
                    "storing {} trajectories state by state -> ledger {ledger}",
                    corpus.slug()
                );
                myelin_eval::build::build_lmev2_trajectories(
                    &trajectories,
                    &haystack,
                    &questions,
                    corpus.slug(),
                    Path::new(&ledger),
                    limit,
                )
                .await?
            } else if pools {
                // D1 does not take `--concurrency`: the pool pass is one
                // batched call per trajectory per pool, sequential, against
                // a reader shared with a live household.
                eprintln!(
                    "ingesting {} typed pools -> collection {collection}, ledger {ledger}",
                    corpus.slug()
                );
                myelin_eval::build::build_lmev2_pools(
                    &trajectories,
                    &haystack,
                    &questions,
                    corpus.slug(),
                    &collection,
                    Path::new(&ledger),
                    limit,
                    repair,
                )
                .await?
            } else {
                eprintln!(
                    "ingesting {} -> collection {collection}, ledger {ledger}",
                    corpus.slug()
                );
                myelin_eval::build::build_lmev2(
                    &trajectories,
                    &haystack,
                    &questions,
                    corpus.slug(),
                    &collection,
                    Path::new(&ledger),
                    limit,
                    repair,
                )
                .await?
            }
        }
    };

    let units = report.per_unit.len();
    report.print(units);

    // A LongMemEval session is supposed to carry a date; LoCoMo's too, but
    // its `date_time` is optional in the release, so only the corpora whose
    // temporal strata depend on it are gated.
    let dated_corpus = matches!(
        corpus,
        Corpus::LongmemevalS | Corpus::LmeV2Small | Corpus::LmeV2Medium
    );
    if report.sessions_without_date > 0 && dated_corpus && !allow_undated {
        anyhow::bail!(
            "{} of {} {} sessions carried no parseable date; those episodes are stamped with \
             today's date, not the conversation's, and every temporal number measured on this \
             memory would be wrong (M19). Fix `parse_session_time` or pass --allow-undated.",
            report.sessions_without_date,
            report.sessions_total,
            corpus.slug(),
        );
    }
    Ok(())
}

/// Run the M4 ablation table against a memory that `build` already wrote.
///
/// Deliberately separate from `build`: rebuilding a memory costs GPU-hours and
/// the ablation costs minutes, so they must be independently runnable.
#[allow(clippy::too_many_arguments)]
async fn ablate_cmd(
    dataset: &str,
    collection: &str,
    ledger: &str,
    units: usize,
    k: usize,
    limit: Option<usize>,
    steps: Option<&[usize]>,
    width: bool,
    corpus: &str,
    grid: GridName,
    holdout: bool,
    dedupe_lineage: bool,
    turn_windows: Option<usize>,
) -> anyhow::Result<()> {
    // clap's value is the user's spelling; the rest of the crate keys on
    // the underscored slug `Corpus::slug` produces, so normalise once here
    // rather than matching two spellings in three places.
    let corpus: &str = &corpus.replace('-', "_");
    if let Some(steps) = steps {
        let points = myelin_eval::ablate::investigate_curve(
            Path::new(dataset),
            collection,
            Path::new(ledger),
            units,
            k,
            steps,
            limit,
            holdout,
        )
        .await?;
        myelin_eval::ablate::print_step_curve(&points, k);
        return Ok(());
    }
    if width {
        // `--corpus` picks the gold annotation and, unless the caller
        // overrode them, the dataset/collection/ledger triple that goes
        // with it. A width sweep pointed at LoCoMo's ledger and
        // LongMemEval's questions would retrieve nothing and report it as
        // a retrieval failure.
        let (dataset, collection, ledger) = match corpus {
            "longmemeval_s" => (
                defaulted(dataset, "data/locomo10.json", "data/longmemeval_s.json"),
                defaulted(collection, "myelin_locomo", "myelin_longmemeval_s"),
                defaulted(ledger, "data/locomo.ledger", "data/longmemeval_s.ledger"),
            ),
            _ => (dataset.to_string(), collection.to_string(), ledger.to_string()),
        };
        let points = myelin_eval::ablate::width_sweep(
            corpus,
            Path::new(&dataset),
            &collection,
            Path::new(&ledger),
            units,
            k,
            grid.cells(),
            limit,
            holdout,
            dedupe_lineage,
            turn_windows,
        )
        .await?;
        myelin_eval::ablate::print_width(&points, k, corpus);
        let slug = grid.slug();
        // The shipped grid is swept over `k` and compose switches, so each
        // of its runs gets its own artifact.
        let dir = match grid {
            GridName::Shipped => format!(
                "runs/{slug}_{corpus}_k{k}{}{}",
                if dedupe_lineage { "_dedupe" } else { "" },
                turn_windows.map(|r| format!("_tw{r}")).unwrap_or_default()
            ),
            _ => format!("runs/{slug}_{corpus}"),
        };
        myelin_eval::ablate::write_width(&points, Path::new(&dir))?;
        return Ok(());
    }
    let run = myelin_eval::ablate::ablate_locomo(
        Path::new(dataset),
        collection,
        Path::new(ledger),
        units,
        k,
        limit,
        holdout,
    )
    .await?;
    myelin_eval::ablate::print_table(&run, k);
    Ok(())
}

/// M69: export a tenant's trajectories and, with `check`, compare them byte
/// for byte against a harness workspace's own files.
async fn trajectories_export(
    ledger: &str,
    namespace: &str,
    tenant: &str,
    out: &str,
    check: Option<&str>,
) -> anyhow::Result<()> {
    use myelin_core::pipeline::trajectory_export::{export_tenant, TRAJECTORY_FILE};
    let ledger = myelin_core::store::ledger::Ledger::open(Path::new(ledger))
        .await
        .context("open ledger")?;
    let out = Path::new(out);
    let n = export_tenant(&ledger, namespace, tenant, out).await?;
    println!("wrote {n} trajectories under {}", out.display());
    let Some(check) = check else { return Ok(()) };
    let theirs = Path::new(check);
    let (mut same, mut differ, mut missing) = (Vec::new(), Vec::new(), Vec::new());
    for entry in std::fs::read_dir(theirs).with_context(|| format!("read {}", theirs.display()))? {
        let dir = entry?.path();
        if !dir.is_dir() {
            continue;
        }
        let Some(id) = dir.file_name().map(|n| n.to_string_lossy().to_string()) else {
            continue;
        };
        let reference = std::fs::read(dir.join(TRAJECTORY_FILE))
            .with_context(|| format!("read {}", dir.join(TRAJECTORY_FILE).display()))?;
        match std::fs::read(out.join(&id).join(TRAJECTORY_FILE)) {
            Ok(ours) if ours == reference => same.push(id),
            Ok(_) => differ.push(id),
            Err(_) => missing.push(id),
        }
    }
    println!(
        "check against {}: {} byte-identical, {} differ, {} missing",
        theirs.display(),
        same.len(),
        differ.len(),
        missing.len()
    );
    anyhow::ensure!(
        differ.is_empty() && missing.is_empty() && !same.is_empty(),
        "export is not byte-identical to the harness: differ {differ:?}, missing {missing:?}"
    );
    Ok(())
}

/// A per-corpus default: keep the caller's value unless it is still clap's
/// default for the *other* corpus.
///
/// clap cannot express "this default depends on another flag", and silently
/// pointing a LongMemEval sweep at LoCoMo's ledger would retrieve nothing
/// and report it as a retrieval failure — a null that looks exactly like a
/// finding.
fn defaulted(current: &str, clap_default: &str, wanted: &str) -> String {
    if current == clap_default {
        wanted.to_string()
    } else {
        current.to_string()
    }
}

/// Score LoCoMo end-to-end against a memory that `build` already wrote.
///
/// Separate from `ablate` for the same reason `ablate` is separate from
/// `build`: retrieval quality and answer quality are different questions and
/// answering the second costs a reader call per question.
#[allow(clippy::too_many_arguments)]
async fn bench_cmd(
    corpus: BenchCorpus,
    dataset: Option<&str>,
    collection: Option<&str>,
    ledger: Option<&str>,
    k: usize,
    budget_tokens: Option<usize>,
    prefetch_limit: Option<u64>,
    rerank_depth: Option<usize>,
    mode: &str,
    max_steps: usize,
    limit: Option<usize>,
    out: Option<&str>,
    resume: bool,
    switches: myelin_eval::bench::BenchSwitches,
    scorer: Option<myelin_eval::bench::Scorer>,
) -> anyhow::Result<()> {
    let mode = match mode {
        "recall" => Mode::Recall,
        "investigate" => Mode::Investigate,
        other => anyhow::bail!("--mode must be recall or investigate, got {other:?}"),
    };
    // The reranked pool cannot be wider than its prefetch, and every M23 arm
    // obeys `prefetch_limit ≥ rerank_depth ≥ k`; catch the typo before a GPU
    // window is spent on it.
    if let Some(depth) = rerank_depth {
        let prefetch = prefetch_limit
            .unwrap_or(myelin_core::pipeline::retrieve::RetrieveConfig::default().prefetch_limit);
        anyhow::ensure!(
            prefetch >= depth as u64,
            "--rerank-depth {depth} exceeds --prefetch-limit {prefetch}: the reranked \
             pool cannot be wider than its prefetch"
        );
    }
    anyhow::ensure!(
        !(switches.question_date && corpus == BenchCorpus::LongmemevalS),
        "--question-date is LoCoMo-only; the LongMemEval_S prompt already carries <today>"
    );
    // Before any GPU work: a judged run needs verdicts, and a live `bench`
    // has none. Discovering that after 50 minutes of reader calls would
    // throw the run away.
    anyhow::ensure!(
        scorer != Some(myelin_eval::bench::Scorer::Judge),
        "--scorer judge is rescore-only: run bench, then judge --run <out>, \
         then rescore --run <out> --scorer judge"
    );
    // `mmr_select` clamps, so `--mmr 70` would silently run pure relevance —
    // the base arm under an arm's directory name, which is the most
    // expensive kind of typo there is.
    anyhow::ensure!(
        switches.mmr.is_none_or(|l| (0.0..=1.0).contains(&l)),
        "--mmr is a lambda in [0.0, 1.0] (1.0 pure relevance, 0.0 pure diversity), got {:?}",
        switches.mmr
    );
    let d = corpus.defaults();
    let dataset = dataset.unwrap_or(d.dataset);
    let collection = collection.unwrap_or(d.collection);
    let ledger = ledger.unwrap_or(d.ledger);
    let scorer = scorer.unwrap_or(d.scorer);
    let owned_out = out.map_or_else(
        || bench_out_dir(d.slug, mode, &switches, scorer),
        str::to_string,
    );
    let out = owned_out.as_str();

    let run = match corpus {
        BenchCorpus::Locomo => {
            myelin_eval::bench::bench_locomo(
                Path::new(dataset),
                collection,
                Path::new(ledger),
                k,
                // 4096 is what every run before M23 composed to; the flag
                // exists so the width arms can raise it with `--k`.
                budget_tokens.unwrap_or(4096),
                prefetch_limit,
                rerank_depth,
                mode,
                max_steps,
                limit,
                &switches,
                scorer,
                Path::new(out),
                resume,
            )
            .await?
        }
        BenchCorpus::LongmemevalS => {
            myelin_eval::bench::bench_longmemeval_s(
                Path::new(dataset),
                collection,
                Path::new(ledger),
                k,
                // 4096 is what every run before M23 composed to; the flag
                // exists so the width arms can raise it with `--k`.
                budget_tokens.unwrap_or(4096),
                prefetch_limit,
                rerank_depth,
                mode,
                max_steps,
                limit,
                &switches,
                scorer,
                Path::new(out),
                resume,
            )
            .await?
        }
    };

    println!();
    println!(
        "  {} {} k={} over {} questions",
        run.corpus, run.mode, run.k, run.questions
    );
    // Named by the scorer the run reports, because `score` carries whichever
    // column `--scorer` selected and a fixed "token F1" label would lie.
    let score_label = match run.scorer.as_str() {
        "temporal" => "temporal (answerable)",
        _ => "token F1 (answerable)",
    };
    println!("    {score_label:<23} {:.4}", run.f1_answerable);
    println!("    exact match             {:.4}", run.em_answerable);
    // Named by what marks an item unanswerable in each corpus, not by
    // LoCoMo's category number: LongMemEval_S uses an `_abs` id suffix and
    // printing "category 5" there pointed at temporal-reasoning instead.
    let abs_label = match run.corpus.as_str() {
        "locomo" => "abstention (category 5)",
        _ => "abstention (_abs items)",
    };
    println!("    {abs_label:<23} {:.4}", run.abstention_accuracy);
    println!(
        "    query latency           p50 {:.2}s  avg {:.2}s",
        run.query_p50_seconds, run.query_avg_seconds
    );
    println!();
    println!("    {:<10}{:>7}{:>12}", "category", "n", "mean");
    for c in &run.by_category {
        println!("    {:<10}{:>7}{:>12.4}", c.category, c.count, c.mean_score);
    }
    println!();
    println!("  wrote {out}/per_question.jsonl and {out}/aggregated_metrics.json");
    Ok(())
}

/// Re-score a finished bench run. Pure CPU, and it never writes into the
/// source: a rescored directory lives under `runs/rescored/` so provenance is
/// visible from the path and cannot collide with a live `bench` directory.
fn rescore_cmd(
    run_dir: &str,
    scorer: myelin_eval::bench::Scorer,
    out: Option<&str>,
) -> anyhow::Result<()> {
    let source = Path::new(run_dir);
    let basename = source
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .with_context(|| format!("--run {run_dir} has no directory name"))?;
    let owned_out = out.map_or_else(
        || format!("runs/rescored/{basename}_{}", scorer.slug()),
        str::to_string,
    );
    let out = owned_out.as_str();
    let run = myelin_eval::bench::rescore_run(source, Path::new(out), scorer)?;

    println!();
    println!(
        "  {} {} rescored under {} over {} questions",
        run.corpus, run.mode, run.scorer, run.questions
    );
    println!("    mean score (answerable) {:.4}", run.f1_answerable);
    println!("    exact match             {:.4}", run.em_answerable);
    println!("    abstention              {:.4}", run.abstention_accuracy);
    println!();
    println!("    {:<10}{:>7}{:>12}", "category", "n", "mean");
    for c in &run.by_category {
        println!("    {:<10}{:>7}{:>12.4}", c.category, c.count, c.mean_score);
    }
    println!();
    println!("  wrote {out}/per_question.jsonl and {out}/aggregated_metrics.json");
    Ok(())
}

/// Grade a finished run's answers with the local reader.
///
/// Reader-only, so the GPU window this needs is a fraction of a bench run's:
/// no embedder, no reranker, no store, no ledger.
async fn judge_cmd(
    run: &str,
    category: Option<u8>,
    limit: Option<usize>,
    seed: Option<&str>,
) -> anyhow::Result<()> {
    let dir = Path::new(run);
    let seed = seed.map(Path::new);
    let (file, stats) = myelin_eval::judge::judge_run(dir, category, limit, seed).await?;
    let total = stats.judged + stats.cached;
    println!();
    println!("  judge {} over {run}", file.model);
    println!(
        "    {} judged, {} from cache, {} of {total} marked correct ({:.1}%)",
        stats.judged,
        stats.cached,
        stats.correct,
        if total == 0 {
            0.0
        } else {
            100.0 * stats.correct as f64 / total as f64
        }
    );
    println!("  wrote {run}/judge_verdicts.json");
    Ok(())
}

/// M16: which side of the pipeline loses an answerable question.
///
/// Reader-only, like `judge`: no embedder, no store, no ledger. The rule in
/// `docs/measurements/m16-evidence-sufficiency.md` is applied to the split,
/// so the verbatim labels are printed beside the rate.
async fn evidence_audit_cmd(run: &str, limit: Option<usize>) -> anyhow::Result<()> {
    let report = myelin_eval::evidence_audit::audit(Path::new(run), limit).await?;
    myelin_eval::evidence_audit::print(&report);
    Ok(())
}

/// M15: the injection gate's false-positive cost on real corpus text.
///
/// Reader-only, like `judge`: no embedder, no store, no ledger. The rule in
/// `docs/measurements/m15-injection-adjudication.md` is applied to this
/// number, so it prints the flagged text in full rather than a rate alone.
async fn adjudicate_probe_cmd(dataset: &str, limit: Option<usize>) -> anyhow::Result<()> {
    let report = myelin_eval::adjudicate_probe::probe(Path::new(dataset), limit).await?;
    myelin_eval::adjudicate_probe::print(&report);
    Ok(())
}
