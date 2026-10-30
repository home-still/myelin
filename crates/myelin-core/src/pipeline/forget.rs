//! Erase a whole scope (C11 unlearning at namespace grain).
//!
//! `forget` erases one record and its descendants; a consumer that stores a
//! document per namespace needs "replace this document", which is every record
//! of the namespace except the ones just written. The order of operations is
//! `forget`'s: the ledger first, then the vector index once, on everything the
//! ledger erased.

use std::collections::{BTreeSet, HashSet};

use uuid::Uuid;

use crate::error::Result;
use crate::model::record::ActorId;
use crate::store::ledger::Ledger;
use crate::store::qdrant::QdrantStore;

/// Hard-delete every record of `tenant` + `namespace` that is not in `keep`,
/// with everything derived from them (**I5**), and return every erased id,
/// sorted.
///
/// The namespace is a required argument, not an `Option`: a missing namespace
/// must not be able to mean "the whole tenant". A `keep` id that is itself
/// derived from an erased record is erased with it; nothing here can retain a
/// descendant of material it was asked to erase.
pub async fn forget_scope(
    ledger: &Ledger,
    store: &QdrantStore,
    tenant: &str,
    namespace: &str,
    keep: &HashSet<Uuid>,
    actor: &ActorId,
    reason: &str,
) -> Result<Vec<Uuid>> {
    let mut gone: BTreeSet<Uuid> = BTreeSet::new();
    for id in ledger.ids_in_scope(tenant, namespace).await? {
        // Kept, or already erased as the descendant of an earlier id.
        if keep.contains(&id) || gone.contains(&id) {
            continue;
        }
        gone.extend(ledger.hard_delete(id, actor, reason).await?);
    }
    let ids: Vec<Uuid> = gone.into_iter().collect();
    store.delete_points(&ids).await?;
    Ok(ids)
}
