//! Every myelin-core schema reaches the model in the order it always has.
//!
//! Until 2026-09-28 serde_json sorted object keys, so every measured run
//! received each schema's `properties` alphabetically. With `preserve_order`
//! the source order is the wire order, and each schema's source was rewritten
//! into the order it was measured with. This test pins that. A schema that is
//! not sorted here would have silently changed what the model writes first
//! (`docs/measurements/defect-2026-09-28-schema-field-order.md`).

use myelin_core::llm::schema_property_orders;
use myelin_core::pipeline::{
    adjudicate, consolidate, decompose, events, extract, investigate, select, trajectory_agent,
};

fn assert_measured_order(name: &str, schema: &serde_json::Value) {
    for (path, keys) in schema_property_orders(schema) {
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "{name}{path}: the order every measured run received is alphabetical");
    }
}

#[test]
fn every_core_schema_keeps_the_order_it_was_measured_with() {
    assert_measured_order("reflection", &investigate::reflection_schema(false));
    assert_measured_order("reflection+typed_probes", &investigate::reflection_schema(true));
    assert_measured_order("self_ask", &investigate::self_ask_schema());
    for label in [
        investigate::DigestLabel::None,
        investigate::DigestLabel::Relevance,
        investigate::DigestLabel::Role,
    ] {
        assert_measured_order("digest", &investigate::digest_schema(4, label));
    }
    assert_measured_order("premise", &investigate::premise_schema(4));
    assert_measured_order("judgement", &consolidate::judgement_schema());
    assert_measured_order("events", &events::events_schema());
    assert_measured_order("extraction", &extract::extraction_schema());
    assert_measured_order("profile", &extract::profile_schema());
    assert_measured_order("selection", &select::selection_schema());
    assert_measured_order("verdict", &adjudicate::verdict_schema());
    assert_measured_order("decomposition", &decompose::decomposition_schema(3));
    assert_measured_order(
        "action",
        &trajectory_agent::action_schema(&[
            trajectory_agent::ToolKind::Summary,
            trajectory_agent::ToolKind::Grep,
            trajectory_agent::ToolKind::Read,
            trajectory_agent::ToolKind::Answer,
        ]),
    );
}

/// The reflection schema is the shipped investigate path's: reasoning
/// (`reason`) before the decision (`sufficient`), by accident of the sort.
#[test]
fn the_shipped_reflection_writes_its_reason_before_its_decision() {
    let orders = schema_property_orders(&investigate::reflection_schema(false));
    assert_eq!(
        orders[0].1,
        vec!["conflict", "next_query", "reason", "sufficient"],
        "{orders:?}"
    );
    let typed = schema_property_orders(&investigate::reflection_schema(true));
    assert_eq!(
        typed[0].1,
        vec!["conflict", "next_kind", "next_query", "reason", "sufficient"],
        "typed probes insert `next_kind` where the measured arm had it"
    );
}
