//! M73b and M20b: a side block ranks one kind of record from its own ledger,
//! appends the top few after the evidence behind a question gate, and never
//! touches the evidence (`myelin_core::pipeline::side_block`).

mod common;

use async_trait::async_trait;
use chrono::{NaiveDate, TimeZone, Utc};
use common::{episode, semantic};
use myelin_core::model::delta::Delta;
use myelin_core::model::evidence::{EvidenceItem, EvidenceKind, EvidenceSet};
use myelin_core::model::query::{Budget, Mode, Recall, ScopeFilter};
use myelin_core::model::record::{ActorId, MemoryRecord, RecordKind, Scope, SourceRef, TrustTier};
use myelin_core::pipeline::side_block::{SideBlock, SideKind, EVENTS_M, PROFILE_M};
use myelin_core::rerank::Reranker;
use myelin_core::store::ledger::Ledger;

const TENANT: &str = "t/side";
const NS: &str = "side";

/// Scores a document by how many of the question's longer words it shares.
struct Overlap;

#[async_trait]
impl Reranker for Overlap {
    fn id(&self) -> &str {
        "overlap"
    }
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> myelin_core::error::Result<Vec<f32>> {
        let words: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 3)
            .map(str::to_string)
            .collect();
        Ok(documents
            .iter()
            .map(|d| {
                let d = d.to_lowercase();
                words.iter().filter(|w| d.contains(w.as_str())).count() as f32
            })
            .collect())
    }
}

fn ask(text: &str, asked_on: Option<NaiveDate>) -> Recall {
    Recall {
        scope: ScopeFilter::tenant(TENANT).with_namespace(NS),
        text: text.to_string(),
        budget: Budget {
            k: 6,
            tokens: 2048,
            max_steps: 1,
        },
        mode: Mode::Recall,
        kinds: None,
        as_of: asked_on,
    }
}

fn day(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("date")
}

/// A ledger holding one parent turn and `records` derived from it.
async fn ledger_with(records: &[MemoryRecord], parent: &MemoryRecord) -> Ledger {
    let ledger = Ledger::open_memory().await.expect("ledger");
    let actor = ActorId::new("test");
    ledger
        .apply(
            &Delta::Add {
                record: Box::new(parent.clone()),
            },
            &actor,
        )
        .await
        .expect("parent turn");
    for r in records {
        ledger
            .apply(
                &Delta::Add {
                    record: Box::new(r.clone()),
                },
                &actor,
            )
            .await
            .expect("record");
    }
    ledger
}

fn dated(mut r: MemoryRecord, on: NaiveDate, kind: RecordKind) -> MemoryRecord {
    r.kind = kind;
    r.validity.t_valid = Utc.from_utc_datetime(&on.and_hms_opt(0, 0, 0).expect("midnight"));
    r
}

/// Evidence the base retrieval produced, standing in for a real recall.
fn base_evidence() -> EvidenceSet {
    EvidenceSet {
        items: vec![EvidenceItem {
            kind: EvidenceKind::Text,
            value: "[2023-03-15] user: by the way, I just got a smoker today".into(),
            record_id: uuid::Uuid::new_v4(),
            source: SourceRef::doc("s1#0"),
            score: 1.0,
            trust: TrustTier::Asserted,
        }],
        tokens: 12,
        ..Default::default()
    }
}

#[tokio::test]
async fn dated_events_come_from_the_day_the_question_names() {
    let scope = Scope::new(TENANT, "myelin", NS);
    let parent = episode(
        &scope,
        "turn-0",
        "user: by the way, I just got a smoker today",
    );
    // Five events on the named day, one a day later that matches better.
    let mut events: Vec<MemoryRecord> = (0..5)
        .map(|i| {
            dated(
                semantic(
                    &scope,
                    &format!("ev-{i}"),
                    &format!("event: the user bought kitchen item {i}"),
                    vec![parent.id],
                ),
                day(2023, 3, 15),
                RecordKind::Semantic,
            )
        })
        .collect();
    events.push(dated(
        semantic(
            &scope,
            "ev-out",
            "event: the user bought a kitchen appliance, a smoker",
            vec![parent.id],
        ),
        day(2023, 3, 16),
        RecordKind::Semantic,
    ));
    let ledger = ledger_with(&events, &parent).await;
    let block = SideBlock::new(SideKind::Events, &ledger, &Overlap);

    let q = ask(
        "What kitchen appliance did I buy 10 days ago?",
        Some(day(2023, 3, 25)),
    );
    let mut set = base_evidence();
    let before = set.clone();
    let n = block.append(&q, &mut set).await.expect("events");
    assert_eq!(n, EVENTS_M, "the top m in-window events");
    assert_eq!(
        set.items[..before.items.len()],
        before.items[..],
        "the evidence is untouched"
    );
    let header = &set.items[before.items.len()];
    assert_eq!(header.value, SideKind::Events.header());
    assert!(header.record_id.is_nil());
    assert_eq!(header.source, SourceRef::doc(SideKind::Events.mechanism()));
    let appended = &set.items[before.items.len() + 1..];
    assert!(
        appended.iter().all(|i| i.value.starts_with("[2023-03-15]")),
        "only the named day"
    );
    assert!(
        !appended.iter().any(|i| i.value.contains("smoker")),
        "the out-of-window event never competes"
    );
    assert!(set.tokens > before.tokens);
}

#[tokio::test]
async fn a_question_without_a_past_day_gets_no_events() {
    let scope = Scope::new(TENANT, "myelin", NS);
    let parent = episode(&scope, "turn-0", "user: hi");
    let ev = dated(
        semantic(
            &scope,
            "ev-0",
            "event: the user bought a smoker",
            vec![parent.id],
        ),
        day(2023, 3, 15),
        RecordKind::Semantic,
    );
    let ledger = ledger_with(&[ev], &parent).await;
    let block = SideBlock::new(SideKind::Events, &ledger, &Overlap);
    for q in [
        ask("What kitchen appliance did I buy?", Some(day(2023, 3, 25))),
        ask(
            "What kitchen appliance did I buy last month?",
            Some(day(2023, 3, 25)),
        ),
        ask("What kitchen appliance did I buy 10 days ago?", None),
    ] {
        let mut set = base_evidence();
        let before = set.clone();
        assert_eq!(
            block.append(&q, &mut set).await.expect("events"),
            0,
            "{}",
            q.text
        );
        assert_eq!(set, before, "byte-identical: {}", q.text);
    }
}

#[tokio::test]
async fn an_advice_request_gets_the_best_ranked_dispositions() {
    let scope = Scope::new(TENANT, "myelin", NS);
    let parent = episode(
        &scope,
        "turn-0",
        "user: turbinado adds a richer flavor to my cookies",
    );
    let mut prefs: Vec<MemoryRecord> = (0..12)
        .map(|i| {
            dated(
                semantic(
                    &scope,
                    &format!("p-{i}"),
                    &format!("The user is interested in unrelated topic {i}."),
                    vec![parent.id],
                ),
                day(2023, 5, 20),
                RecordKind::Profile,
            )
        })
        .collect();
    prefs.push(dated(
        semantic(
            &scope,
            "p-gold",
            "The user likes turbinado sugar in chocolate chip cookies.",
            vec![parent.id],
        ),
        day(2023, 1, 2),
        RecordKind::Profile,
    ));
    let ledger = ledger_with(&prefs, &parent).await;
    let block = SideBlock::new(SideKind::Profile, &ledger, &Overlap);

    let q = ask(
        "Any suggestions for my chocolate chip cookies?",
        Some(day(2023, 5, 30)),
    );
    let mut set = base_evidence();
    let n = block.append(&q, &mut set).await.expect("profile");
    assert_eq!(n, PROFILE_M);
    assert_eq!(set.items[1].value, SideKind::Profile.header());
    assert!(
        set.items[2].value.contains("turbinado"),
        "the oldest disposition ranks first when it answers the question: {}",
        set.items[2].value
    );

    let mut plain = base_evidence();
    let before = plain.clone();
    let q = ask(
        "What sugar did I use in my cookies?",
        Some(day(2023, 5, 30)),
    );
    assert_eq!(
        block.append(&q, &mut plain).await.expect("profile"),
        0,
        "not an advice request"
    );
    assert_eq!(plain, before);
}
