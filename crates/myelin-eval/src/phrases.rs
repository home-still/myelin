//! Populate the phrase↔record incidence graph over an already-built ledger.
//!
//! **Pure SQLite: no Qdrant, no GPU, no model.** That is the point. The graph
//! route needs phrase nodes on corpora whose builds cost GPU-days
//! (`docs/measurements/m3-write-path.md`), and re-ingesting 162,254
//! LongMemEval_S records to add a derived table would be absurd. Incidence is
//! derived from `record.text` and `record.entities`, both already in the
//! ledger, by [`myelin_core::pipeline::phrases::incidence_rows`] — the same
//! function [`myelin_core::pipeline::index::Indexer::index`] calls, so a
//! fresh ingest and this backfill write byte-identical rows.
//!
//! Re-running after an incremental ingest is safe:
//! `Ledger::replace_incidence_batch` is authoritative per record, so a second
//! pass converges to the same graph rather than accumulating edges.

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use myelin_core::pipeline::phrases::incidence_rows;
use myelin_core::store::ledger::Ledger;

/// What one backfill pass wrote.
///
/// Because the pass visits every record in the namespace and the write is
/// authoritative per record, `edges` **is** the namespace's incidence edge
/// count — the metric `PLAN.md` §5.4 makes first-class, obtained with no
/// read-back pass and no 5M-row materialisation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhraseStats {
    pub records: usize,
    pub edges: usize,
    pub distinct_phrases: usize,
}

pub async fn backfill_phrases(
    ledger_path: &Path,
    namespace: &str,
    batch: usize,
    limit: Option<usize>,
) -> Result<PhraseStats> {
    anyhow::ensure!(
        ledger_path.exists(),
        "missing {}; run `myelin-eval build` first",
        ledger_path.display()
    );
    let ledger = Ledger::open(ledger_path).await.context("open ledger")?;
    let batch = batch.max(1);

    let mut stats = PhraseStats::default();
    let mut distinct: HashSet<String> = HashSet::new();
    let mut cursor = None;

    loop {
        // Keyset pagination, so a namespace is never materialised whole.
        let take = match limit {
            Some(n) => batch.min(n.saturating_sub(stats.records)),
            None => batch,
        };
        if take == 0 {
            break;
        }
        let page = ledger
            .records_after(namespace, cursor, take as i64)
            .await
            .context("scan records")?;
        if page.is_empty() {
            break;
        }

        let rows: Vec<_> = page.iter().flat_map(incidence_rows).collect();
        for row in &rows {
            distinct.insert(row.phrase.clone());
        }
        stats.edges += ledger
            .replace_incidence_batch(&rows)
            .await
            .context("write incidence")?;
        stats.records += page.len();
        cursor = page.last().map(|r| r.id);
        eprintln!(
            "  {:>8} records  {:>9} edges  {:>8} distinct phrases",
            stats.records,
            stats.edges,
            distinct.len()
        );

        if page.len() < take {
            break;
        }
    }

    stats.distinct_phrases = distinct.len();
    Ok(stats)
}
