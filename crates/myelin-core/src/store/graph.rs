//! The smallest graph the evidence justifies (`PLAN.md` §5.4).
//!
//! Phrase↔record incidence lives in SQLite and is loaded into `petgraph` for
//! personalized PageRank when the read policy routes a query as multi-hop.
//!
//! **1-hop expansion only.** HippoRAG-2's own numbers say 1-hop + PPR beats
//! naive neighbour expansion R@5 **72.5 vs 59.2**, and 2–3 hops *degrade*
//! (`07-graph-memory.md` §1, §5). Damping is `[UNVERIFIED]` in the corpus;
//! 0.5 is the Forgetting-pack default and is treated as a tuned parameter, not
//! a constant of nature.
//!
//! Node specificity `s_i = |P_i|^-1` down-weights phrases that touch many
//! records, which is the cheap stand-in for the synonym-edge explosion risk
//! (1,125,951 synonym vs 140,830 extracted edges on MuSiQue).

use std::collections::HashMap;

use petgraph::graph::{NodeIndex, UnGraph};
use uuid::Uuid;

use crate::error::Result;
use crate::store::ledger::{IncidenceRow, Ledger};

pub const DEFAULT_DAMPING: f32 = 0.5;
pub const DEFAULT_ITERATIONS: usize = 30;
/// All-record background seeds, per §5.4.
pub const BACKGROUND_SEED_WEIGHT: f32 = 0.05;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Node {
    Phrase(String),
    Record(Uuid),
}

/// The bipartite incidence structure, in memory.
pub struct IncidenceGraph {
    graph: UnGraph<Node, f32>,
    phrases: HashMap<String, NodeIndex>,
    records: HashMap<Uuid, NodeIndex>,
}

impl IncidenceGraph {
    pub fn from_rows(rows: &[IncidenceRow]) -> Self {
        let mut graph = UnGraph::<Node, f32>::new_undirected();
        let mut phrases = HashMap::new();
        let mut records = HashMap::new();

        for row in rows {
            let p = *phrases
                .entry(row.phrase.clone())
                .or_insert_with(|| graph.add_node(Node::Phrase(row.phrase.clone())));
            let r = *records
                .entry(row.record_id)
                .or_insert_with(|| graph.add_node(Node::Record(row.record_id)));
            graph.add_edge(p, r, row.weight);
        }

        Self {
            graph,
            phrases,
            records,
        }
    }

    pub async fn load(ledger: &Ledger, namespace: Option<&str>) -> Result<Self> {
        Ok(Self::from_rows(&ledger.incidence(namespace).await?))
    }

    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    pub fn phrase_count(&self) -> usize {
        self.phrases.len()
    }

    /// Edge count is a tracked metric: synonym-edge explosion is the storage
    /// risk this design is guarding against.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Node specificity `s_i = |P_i|^-1` — a phrase incident to many records
    /// carries less signal per edge.
    fn specificity(&self, node: NodeIndex) -> f32 {
        let degree = self.graph.neighbors(node).count();
        if degree == 0 {
            0.0
        } else {
            1.0 / degree as f32
        }
    }

    /// Personalized PageRank from phrase seeds, returning record scores in
    /// descending order.
    ///
    /// Seeds are the query's entity phrases; every record also gets a small
    /// background reset mass so a query whose phrases miss the index still
    /// produces a ranking rather than an empty one.
    pub fn personalized_pagerank(
        &self,
        seed_phrases: &[String],
        damping: f32,
        iterations: usize,
    ) -> Vec<(Uuid, f32)> {
        let n = self.graph.node_count();
        if n == 0 {
            return Vec::new();
        }

        // Reset distribution: seed phrases weighted by specificity, plus a
        // uniform background over records.
        let mut reset = vec![0.0f32; n];
        let mut seed_mass = 0.0f32;
        for phrase in seed_phrases {
            if let Some(&idx) = self.phrases.get(phrase) {
                let w = self.specificity(idx).max(f32::EPSILON);
                reset[idx.index()] += w;
                seed_mass += w;
            }
        }
        let background = BACKGROUND_SEED_WEIGHT / self.records.len().max(1) as f32;
        for &idx in self.records.values() {
            reset[idx.index()] += background;
            seed_mass += background;
        }
        if seed_mass <= 0.0 {
            return Vec::new();
        }
        for r in &mut reset {
            *r /= seed_mass;
        }

        // Weighted out-degree, precomputed.
        let out_weight: Vec<f32> = (0..n)
            .map(|i| {
                self.graph
                    .edges(NodeIndex::new(i))
                    .map(|e| *e.weight())
                    .sum::<f32>()
            })
            .collect();

        let mut rank = reset.clone();
        let mut next = vec![0.0f32; n];
        for _ in 0..iterations {
            next.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..n {
                let r = rank[i];
                if r == 0.0 || out_weight[i] == 0.0 {
                    continue;
                }
                for edge in self.graph.edges(NodeIndex::new(i)) {
                    use petgraph::visit::EdgeRef;
                    let target = if edge.source().index() == i {
                        edge.target()
                    } else {
                        edge.source()
                    };
                    next[target.index()] += damping * r * (*edge.weight() / out_weight[i]);
                }
            }
            for i in 0..n {
                next[i] += (1.0 - damping) * reset[i];
            }
            std::mem::swap(&mut rank, &mut next);
        }

        let mut scored: Vec<(Uuid, f32)> = self
            .records
            .iter()
            .map(|(id, idx)| (*id, rank[idx.index()]))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored
    }
}
