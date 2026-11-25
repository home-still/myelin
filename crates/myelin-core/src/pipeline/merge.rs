//! One evidence set out of several retrievals.
//!
//! A read that names several stores is several retrievals, each scoped to its
//! own tenant (C12 holds per retrieval). What comes back from each is ranked
//! by that retrieval's own fusion, and fused rank scores from separate
//! retrievals are not comparable: an RRF score says where a record stood in
//! *its* candidate lists, not how well it answers the question. So the pool
//! is scored again, by one cross-encoder pass over `(query, item text)`, and
//! that score alone orders it.

use crate::error::{MyelinError, Result};
use crate::model::evidence::EvidenceItem;
use crate::rerank::Reranker;

use super::ingest::approx_tokens;

/// The merged pool, best first.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MergedEvidence {
    /// Each item's `score` is the cross-encoder's, not its retrieval's.
    pub items: Vec<EvidenceItem>,
    /// Aligned 1:1 with `items`: the index of the retrieval each came from.
    pub origins: Vec<usize>,
    pub tokens: usize,
}

/// Pool the evidence of several retrievals, score it against `query` in one
/// reranker pass, and keep the best `k` that fit `max_tokens`.
///
/// `sets[i]` is retrieval `i`'s items. Order: score descending, ties by
/// record id, then by retrieval index, so the result is deterministic. The
/// budget is taken from the tail, and the best item is kept even when it
/// alone exceeds the budget — the rule `compose` applies to one retrieval,
/// so naming a second store never turns an answer into an empty set.
pub async fn merge_scopes(
    reranker: &dyn Reranker,
    query: &str,
    sets: Vec<Vec<EvidenceItem>>,
    k: usize,
    max_tokens: usize,
) -> Result<MergedEvidence> {
    let mut pool: Vec<(usize, EvidenceItem)> = sets
        .into_iter()
        .enumerate()
        .flat_map(|(origin, items)| items.into_iter().map(move |item| (origin, item)))
        .collect();
    if pool.is_empty() {
        return Ok(MergedEvidence::default());
    }

    let documents: Vec<String> = pool.iter().map(|(_, item)| item.value.clone()).collect();
    let scores = reranker.rerank(query, &documents).await?;
    if scores.len() != pool.len() {
        return Err(MyelinError::Store(format!(
            "{}: scored {} of {} pooled items",
            reranker.id(),
            scores.len(),
            pool.len()
        )));
    }
    for ((_, item), score) in pool.iter_mut().zip(scores) {
        item.score = score;
    }

    pool.sort_by(|(origin_a, a), (origin_b, b)| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.record_id.cmp(&b.record_id))
            .then_with(|| origin_a.cmp(origin_b))
    });
    pool.truncate(k);

    let mut tokens: usize = pool
        .iter()
        .map(|(_, item)| approx_tokens(&item.value))
        .sum();
    while tokens > max_tokens && pool.len() > 1 {
        if let Some((_, dropped)) = pool.pop() {
            tokens -= approx_tokens(&dropped.value);
        }
    }

    let (origins, items) = pool.into_iter().unzip();
    Ok(MergedEvidence {
        items,
        origins,
        tokens,
    })
}

#[cfg(test)]
mod tests {
    use async_trait::async_trait;
    use uuid::Uuid;

    use super::*;
    use crate::model::evidence::EvidenceKind;
    use crate::model::record::{SourceRef, TrustTier};

    /// Scores a document by the number written after `score=` in its text, so
    /// a test states the cross-encoder's verdict in the fixture itself.
    struct ScoreInText {
        /// Report one score fewer than asked, to exercise the contract check.
        short: bool,
    }

    #[async_trait]
    impl Reranker for ScoreInText {
        fn id(&self) -> &str {
            "score-in-text"
        }

        async fn rerank(&self, _query: &str, documents: &[String]) -> Result<Vec<f32>> {
            let mut scores: Vec<f32> = documents
                .iter()
                .map(|d| {
                    d.split("score=")
                        .nth(1)
                        .and_then(|rest| rest.split_whitespace().next())
                        .and_then(|n| n.parse().ok())
                        .expect("fixture text carries score=<n>")
                })
                .collect();
            if self.short {
                scores.pop();
            }
            Ok(scores)
        }
    }

    const RERANKER: ScoreInText = ScoreInText { short: false };

    /// `retrieval_score` is deliberately the opposite of the text's score in
    /// every fixture below: an order that survives the merge by retrieval
    /// score would be visible.
    fn item(id: u128, text: &str, retrieval_score: f32) -> EvidenceItem {
        EvidenceItem {
            kind: EvidenceKind::Text,
            value: text.to_string(),
            record_id: Uuid::from_u128(id),
            source: SourceRef::doc("test"),
            score: retrieval_score,
            trust: TrustTier::Asserted,
        }
    }

    fn ids(merged: &MergedEvidence) -> Vec<u128> {
        merged.items.iter().map(|i| i.record_id.as_u128()).collect()
    }

    #[tokio::test]
    async fn the_pool_is_ordered_by_the_cross_encoder_not_by_each_retrievals_score() {
        let sets = vec![
            vec![item(1, "a score=0.2", 0.99), item(2, "b score=0.9", 0.10)],
            vec![item(3, "c score=0.5", 0.98), item(4, "d score=0.7", 0.05)],
        ];
        let merged = merge_scopes(&RERANKER, "q", sets, 10, 10_000)
            .await
            .expect("merge");

        assert_eq!(ids(&merged), vec![2, 4, 3, 1]);
        assert_eq!(merged.origins, vec![0, 1, 1, 0]);
        let scores: Vec<f32> = merged.items.iter().map(|i| i.score).collect();
        assert_eq!(
            scores,
            vec![0.9, 0.7, 0.5, 0.2],
            "items carry the pooled score"
        );
    }

    #[tokio::test]
    async fn equal_scores_are_ordered_by_record_id() {
        let sets = vec![
            vec![item(9, "a score=0.5", 0.0)],
            vec![item(3, "b score=0.5", 0.0), item(6, "c score=0.5", 0.0)],
        ];
        let merged = merge_scopes(&RERANKER, "q", sets, 10, 10_000)
            .await
            .expect("merge");
        assert_eq!(ids(&merged), vec![3, 6, 9]);
    }

    #[tokio::test]
    async fn k_keeps_the_best_and_the_token_budget_trims_the_tail() {
        // Every fixture text costs the same, so the budget is stated in items.
        let each = approx_tokens("one two three score=0.1");
        let sets = vec![
            vec![
                item(1, "one two three score=0.1", 0.0),
                item(2, "one two three score=0.8", 0.0),
            ],
            vec![
                item(3, "one two three score=0.6", 0.0),
                item(4, "one two three score=0.4", 0.0),
            ],
        ];
        let by_k = merge_scopes(&RERANKER, "q", sets.clone(), 3, 10_000)
            .await
            .expect("merge");
        assert_eq!(ids(&by_k), vec![2, 3, 4]);
        assert_eq!(by_k.tokens, 3 * each);

        let two_and_a_bit = 2 * each + 1;
        let by_budget = merge_scopes(&RERANKER, "q", sets.clone(), 3, two_and_a_bit)
            .await
            .expect("merge");
        assert_eq!(
            ids(&by_budget),
            vec![2, 3],
            "the third does not fit the budget"
        );
        assert_eq!(by_budget.tokens, 2 * each);
        assert_eq!(by_budget.origins, vec![0, 1]);

        // The best item survives a budget smaller than itself.
        let starved = merge_scopes(&RERANKER, "q", sets, 3, 1)
            .await
            .expect("merge");
        assert_eq!(ids(&starved), vec![2]);
        assert_eq!(starved.tokens, each);
    }

    #[tokio::test]
    async fn an_empty_pool_is_an_empty_result_and_asks_the_reranker_nothing() {
        // `short` would fail the contract check if the reranker were asked.
        let merged = merge_scopes(
            &ScoreInText { short: true },
            "q",
            vec![vec![], vec![]],
            5,
            100,
        )
        .await
        .expect("merge");
        assert_eq!(merged, MergedEvidence::default());
    }

    #[tokio::test]
    async fn a_reranker_that_scores_too_few_items_is_an_error_not_a_misaligned_ranking() {
        let sets = vec![
            vec![item(1, "a score=0.1", 0.0)],
            vec![item(2, "b score=0.2", 0.0)],
        ];
        let err = merge_scopes(&ScoreInText { short: true }, "q", sets, 5, 100)
            .await
            .expect_err("a short score list must not be zipped onto the pool");
        assert!(err.to_string().contains("scored 1 of 2"), "{err}");
    }
}
