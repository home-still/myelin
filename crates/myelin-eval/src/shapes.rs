//! `myelin-eval shapes`: where a query-shape gate fires, per category.
//!
//! A gated arm reruns only the questions its gate opens (`bench --questions`)
//! and pre-registers how often the gate fires on each stratum. Both have to
//! come from the functions `bench` itself calls, or the arm would run on one
//! detector's list and be read against another's (M87,
//! `docs/measurements/m87-enumeration-depth.md`).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;
use std::str::FromStr;

use anyhow::{bail, Result};
use myelin_core::pipeline::query_shape::{
    is_aggregation_question, is_enumeration_question, is_non_recall_request,
};

use crate::datasets::{locomo, longmemeval};

/// One query-shape gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// M72: counts and sums.
    Aggregation,
    /// M87: lists.
    Enumeration,
    /// M84: advice and inference requests.
    NonRecall,
}

impl FromStr for Shape {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "aggregation" => Ok(Self::Aggregation),
            "enumeration" => Ok(Self::Enumeration),
            "non-recall" => Ok(Self::NonRecall),
            other => bail!("unknown shape {other:?} (aggregation | enumeration | non-recall)"),
        }
    }
}

impl Shape {
    pub fn fires(self, question: &str) -> bool {
        match self {
            Self::Aggregation => is_aggregation_question(question),
            Self::Enumeration => is_enumeration_question(question),
            Self::NonRecall => is_non_recall_request(question),
        }
    }
}

/// LoCoMo's category numbers, named as its paper names them.
const LOCOMO_CATEGORIES: [(u8, &str); 5] = [
    (1, "multi-hop"),
    (2, "temporal"),
    (3, "open-domain"),
    (4, "single-hop"),
    (5, "adversarial"),
];

/// Hits and totals per stratum, and the ids of every question hit, in
/// corpus order.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ShapeTable {
    pub by_stratum: BTreeMap<String, (usize, usize)>,
    pub ids: Vec<String>,
}

impl ShapeTable {
    fn count(&mut self, stratum: String, id: String, hit: bool) {
        let entry = self.by_stratum.entry(stratum).or_default();
        entry.1 += 1;
        if hit {
            entry.0 += 1;
            self.ids.push(id);
        }
    }

    pub fn render(&self) -> String {
        let mut out = String::from("| stratum | fires | of |\n|---|---|---|\n");
        for (stratum, (hits, total)) in &self.by_stratum {
            let _ = writeln!(out, "| {stratum} | {hits} | {total} |");
        }
        let (hits, total) = self
            .by_stratum
            .values()
            .fold((0, 0), |(h, t), (a, b)| (h + a, t + b));
        let _ = writeln!(out, "| **all** | **{hits}** | **{total}** |");
        out
    }
}

/// A question is hit when any of `shapes` fires on it.
fn any_fires(shapes: &[Shape], question: &str) -> bool {
    shapes.iter().any(|s| s.fires(question))
}

/// LoCoMo, with `bench`'s question ids (`<sample_id>#<index in qa>`).
pub fn locomo_table(path: &Path, shapes: &[Shape]) -> Result<ShapeTable> {
    let mut table = ShapeTable::default();
    for conv in locomo::load(path)? {
        for (i, qa) in conv.qa.iter().enumerate() {
            let Some((_, name)) = LOCOMO_CATEGORIES.iter().find(|(c, _)| *c == qa.category) else {
                bail!("{}#{i}: LoCoMo category {} is not one of 1-5", conv.sample_id, qa.category);
            };
            table.count(
                (*name).to_string(),
                format!("{}#{i}", conv.sample_id),
                any_fires(shapes, &qa.question),
            );
        }
    }
    Ok(table)
}

/// LongMemEval_S, by question type, abstention items apart.
pub fn longmemeval_table(path: &Path, shapes: &[Shape]) -> Result<ShapeTable> {
    let mut table = ShapeTable::default();
    for item in longmemeval::load(path)? {
        let stratum = if item.is_abstention() {
            format!("{} (abstention)", item.question_type)
        } else {
            item.question_type.clone()
        };
        let hit = any_fires(shapes, &item.question);
        table.count(stratum, item.question_id, hit);
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_parse_by_name_and_refuse_anything_else() {
        assert_eq!("enumeration".parse::<Shape>().unwrap(), Shape::Enumeration);
        assert_eq!("aggregation".parse::<Shape>().unwrap(), Shape::Aggregation);
        assert_eq!("non-recall".parse::<Shape>().unwrap(), Shape::NonRecall);
        assert!("lists".parse::<Shape>().is_err());
    }

    #[test]
    fn a_table_counts_hits_and_keeps_their_ids_in_order() {
        let mut t = ShapeTable::default();
        t.count("multi-hop".into(), "a#0".into(), true);
        t.count("multi-hop".into(), "a#1".into(), false);
        t.count("single-hop".into(), "a#2".into(), true);
        assert_eq!(t.by_stratum["multi-hop"], (1, 2));
        assert_eq!(t.ids, vec!["a#0", "a#2"]);
        assert!(t.render().contains("| **all** | **2** | **3** |"));
    }
}
