//! `compose` — turn ranked records into a budgeted `EvidenceSet`
//! (`PLAN.md` §7.3). "The part everyone gets wrong."
//!
//! Three decisions, each forced by a measurement:
//!
//! - **Bookend the order.** GPT-3.5-Turbo on 20-document NQ-Open scores
//!   **53.8%** with the gold document in the middle — *below* its own **56.1%
//!   closed-book** score (`10.48550/arxiv.2307.03172` Table 6). Burying the
//!   best evidence mid-context is worse than supplying none. So rank 1 goes
//!   first, rank 2 goes last, and the weakest items fill the middle.
//! - **Return few items.** HiGMem retrieves 8.09 vs 99.84 turns/query at
//!   P@K 0.1909 vs 0.0101. `k` is also a *security* parameter: MINJA ASR
//!   climbs 6% → 20% → 38% as k goes 3 → 5 → 10.
//! - **Dedup before emitting.** Near-identical items waste budget twice: once
//!   in tokens and once by pushing a distinct fact out of the set.
//!
//! The wire form is fixed by R1 and produced by
//! [`crate::model::evidence::EvidenceSet::to_wire`]; everything here decides
//! *which* items and *in what order*.

use serde::{Deserialize, Serialize};

use crate::model::evidence::{EvidenceItem, EvidenceKind, EvidenceSet};
use crate::model::record::MemoryRecord;

use super::consolidate::cosine;
use super::ingest::approx_tokens;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComposeConfig {
    /// Maximum items returned. Small by evidence and by threat model.
    pub k: usize,
    /// Total token ceiling for the composed set.
    pub max_tokens: usize,
    /// Cosine at or above which two items are the same evidence.
    pub tau_near_dup: f32,
    /// Prefix `Untrusted` items with their tier in the emitted text.
    ///
    /// **Default off, because it was measured and it does not work.**
    ///
    /// The idea was sound on paper: E1 injects paraphrased poison that the
    /// pattern gate misses, the store already knows it is `Untrusted`
    /// (score 0.30) against first-party memory at `Verified` (0.90), and
    /// the read path was discarding that. Prefixing `[untrusted source]`
    /// tells the reader what the store knows.
    ///
    /// Measured effect on attack success: **none.** ASR stayed at 100%
    /// (empty) and 80% (pre-populated, k=6), identical to the unlabelled
    /// run. A 9B reader repeats the content regardless of the label.
    ///
    /// So it is off: it costs tokens in every prompt and buys nothing, and
    /// shipping a defence that provably does not defend is worse than
    /// having none, because it invites the belief that the problem is
    /// handled. `EvidenceItem::trust` still carries the tier structurally,
    /// where a consumer that can actually act on it will find it.
    ///
    /// Kept as a switch so the next attempt at a reader-side defence has a
    /// baseline to beat.
    pub label_untrusted: bool,
}

impl Default for ComposeConfig {
    fn default() -> Self {
        Self {
            k: 6,
            max_tokens: 2048,
            tau_near_dup: 0.93,
            label_untrusted: false,
        }
    }
}

/// A ranked candidate on its way into the evidence set.
pub struct Ranked {
    pub record: MemoryRecord,
    pub score: f32,
    /// Dense vector, when available, for near-duplicate suppression. Without
    /// it dedup falls back to exact text equality.
    pub vector: Option<Vec<f32>>,
}

/// Select, dedup, budget and bookend.
///
/// Input must be sorted best-first; this function does not re-rank.
/// The emitted text for one record.
///
/// `Verified` and `Asserted` pass through unchanged: labelling everything
/// would make the label meaningless, which is the same failure as a poison
/// filter that flags all text. Only material that crossed a trust boundary
/// is marked.
fn label(record: &crate::model::record::MemoryRecord, cfg: &ComposeConfig) -> String {
    use crate::model::record::TrustTier;
    if cfg.label_untrusted && record.trust.tier == TrustTier::Untrusted {
        format!("[untrusted source] {}", record.text)
    } else {
        record.text.clone()
    }
}

pub fn compose(ranked: Vec<Ranked>, cfg: &ComposeConfig) -> EvidenceSet {
    // 1. Dedup, keeping the higher-ranked copy.
    let mut kept: Vec<Ranked> = Vec::new();
    for candidate in ranked {
        let duplicate = kept.iter().any(|k| {
            if k.record.text == candidate.record.text {
                return true;
            }
            match (&k.vector, &candidate.vector) {
                (Some(a), Some(b)) => cosine(a, b) >= cfg.tau_near_dup,
                _ => false,
            }
        });
        if !duplicate {
            kept.push(candidate);
        }
    }

    // 2. Budget: take in rank order until k or the token ceiling binds.
    let mut selected: Vec<Ranked> = Vec::new();
    let mut tokens = 0usize;
    for candidate in kept {
        if selected.len() >= cfg.k {
            break;
        }
        let cost = approx_tokens(&candidate.record.text);
        // Always admit the top item: returning nothing because the single
        // best piece of evidence is large is worse than overrunning slightly.
        if !selected.is_empty() && tokens + cost > cfg.max_tokens {
            continue;
        }
        tokens += cost;
        selected.push(candidate);
    }

    // 3. Bookend. Strongest first, second-strongest last, rest in the middle.
    let ordered = bookend(selected);

    let items: Vec<EvidenceItem> = ordered
        .into_iter()
        .map(|r| EvidenceItem {
            kind: EvidenceKind::Text,
            value: label(&r.record, cfg),
            record_id: r.record.id,
            source: r.record.provenance.source.clone(),
            score: r.score,
            trust: r.record.trust.tier,
        })
        .collect();

    EvidenceSet {
        items,
        tokens,
        trace: Vec::new(),
    }
}

/// `[1st, 3rd, 5th, …, 6th, 4th, 2nd]` — best at the head, second-best at the
/// tail, weakest buried in the middle where attention is worst.
fn bookend<T>(mut ranked: Vec<T>) -> Vec<T> {
    let mut head: Vec<T> = Vec::with_capacity(ranked.len());
    let mut tail: Vec<T> = Vec::with_capacity(ranked.len());
    ranked.reverse(); // pop() now yields best-first
    let mut to_head = true;
    while let Some(item) = ranked.pop() {
        if to_head {
            head.push(item);
        } else {
            tail.push(item);
        }
        to_head = !to_head;
    }
    tail.reverse();
    head.extend(tail);
    head
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::record::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn record(text: &str) -> MemoryRecord {
        MemoryRecord {
            id: Uuid::new_v4(),
            kind: RecordKind::Semantic,
            scope: Scope::new("t", "a", "ns"),
            text: text.into(),
            entities: vec![],
            validity: Validity {
                t_valid: Utc::now(),
                t_invalid: None,
                t_ingested: Utc::now(),
                t_expired: None,
            },
            provenance: Provenance {
                source: SourceRef::doc("d"),
                contributed_by: ActorId::new("u"),
                written_by: ActorId::new("w"),
                derived_from: vec![],
            },
            trust: Trust::asserted(),
            salience: Salience::default(),
            links: vec![],
        }
    }

    fn ranked(texts: &[&str]) -> Vec<Ranked> {
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| Ranked {
                record: record(t),
                score: 1.0 - i as f32 * 0.1,
                vector: None,
            })
            .collect()
    }

    /// The lost-in-the-middle defence, stated as a test: best first,
    /// second-best last, weakest in the middle.
    #[test]
    fn the_two_strongest_items_bookend_the_set() {
        let set = compose(
            ranked(&["best", "second", "third", "fourth", "fifth"]),
            &ComposeConfig::default(),
        );
        let values: Vec<&str> = set.items.iter().map(|i| i.value.as_str()).collect();
        assert_eq!(values.first(), Some(&"best"));
        assert_eq!(values.last(), Some(&"second"));
        assert_eq!(values, vec!["best", "third", "fifth", "fourth", "second"]);
    }

    #[test]
    fn a_single_item_is_not_reordered() {
        let set = compose(ranked(&["only"]), &ComposeConfig::default());
        assert_eq!(set.items.len(), 1);
        assert_eq!(set.items[0].value, "only");
    }

    /// k is a security parameter, not just a cost one.
    #[test]
    fn k_caps_the_set() {
        let cfg = ComposeConfig {
            k: 3,
            ..Default::default()
        };
        let set = compose(ranked(&["a", "b", "c", "d", "e", "f"]), &cfg);
        assert_eq!(set.items.len(), 3);
    }

    #[test]
    fn exact_duplicates_are_dropped_keeping_the_higher_rank() {
        let set = compose(
            ranked(&["same", "other", "same"]),
            &ComposeConfig::default(),
        );
        assert_eq!(set.items.len(), 2);
        assert_eq!(set.items[0].value, "same");
    }

    /// Near-duplicates by cosine, not just exact text — paraphrases are the
    /// common case and waste the budget twice.
    #[test]
    fn near_duplicates_are_dropped_by_cosine() {
        let items = vec![
            Ranked {
                record: record("the user moved to Berlin"),
                score: 1.0,
                vector: Some(vec![1.0, 0.0, 0.0]),
            },
            Ranked {
                record: record("the user relocated to Berlin"),
                score: 0.9,
                vector: Some(vec![0.999, 0.01, 0.0]),
            },
            Ranked {
                record: record("the user likes rye bread"),
                score: 0.8,
                vector: Some(vec![0.0, 1.0, 0.0]),
            },
        ];
        let set = compose(items, &ComposeConfig::default());
        assert_eq!(set.items.len(), 2, "{:?}", set.items);
        assert_eq!(set.items[0].value, "the user moved to Berlin");
    }

    /// The token ceiling binds before k does when items are large.
    #[test]
    fn the_token_budget_binds() {
        let long = "word ".repeat(200);
        let cfg = ComposeConfig {
            k: 6,
            max_tokens: 300,
            tau_near_dup: 0.93,
            ..Default::default()
        };
        let items: Vec<Ranked> = (0..5)
            .map(|i| Ranked {
                record: record(&format!("{long}{i}")),
                score: 1.0 - i as f32 * 0.1,
                vector: None,
            })
            .collect();
        let set = compose(items, &cfg);
        assert!(set.items.len() < 5, "budget did not bind: {}", set.items.len());
        assert!(set.tokens > 0);
    }

    /// Returning nothing because the single best item is large is worse than
    /// a slight overrun.
    #[test]
    fn the_top_item_is_admitted_even_if_it_alone_exceeds_the_budget() {
        let huge = "word ".repeat(5000);
        let cfg = ComposeConfig {
            k: 6,
            max_tokens: 10,
            tau_near_dup: 0.93,
            ..Default::default()
        };
        let set = compose(
            vec![Ranked {
                record: record(&huge),
                score: 1.0,
                vector: None,
            }],
            &cfg,
        );
        assert_eq!(set.items.len(), 1, "an oversized best item must still be returned");
    }

    /// R1: the wire form is exactly `[{"type","value"}]`, nothing else.
    #[test]
    fn the_wire_form_carries_only_type_and_value() {
        let set = compose(ranked(&["a", "b"]), &ComposeConfig::default());
        let json = serde_json::to_value(set.to_wire()).unwrap();
        let first = &json[0];
        assert_eq!(first["type"], "text");
        assert_eq!(first["value"], "a");
        assert_eq!(
            first.as_object().unwrap().len(),
            2,
            "R1 forbids extra keys on the wire: {first}"
        );
    }

    #[test]
    fn an_empty_input_composes_to_an_empty_set() {
        let set = compose(Vec::new(), &ComposeConfig::default());
        assert!(set.is_empty());
        assert!(set.to_wire().is_empty());
    }
}
