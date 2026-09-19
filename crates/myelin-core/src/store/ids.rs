//! Deterministic record ids: UUID v5 over (namespace, natural key).
//!
//! Same idea as `hs-distill`'s `qdrant.rs`, which derives point ids as
//! `uuid_v5(DNS_NAMESPACE, xxh3(doc_id:chunk_index))`. We drop the xxh3
//! pre-hash — v5 already hashes its name — and derive a per-namespace UUID
//! first, so two tenants using the same natural key get different ids.
//!
//! The property that matters is idempotent re-ingest: the same logical fact in
//! the same namespace always lands on the same point, so a replayed trajectory
//! updates rather than duplicates.

use uuid::Uuid;

use crate::model::record::Scope;

/// RFC 4122 DNS namespace — the same root `hs-distill` uses.
const DNS_NAMESPACE: Uuid = Uuid::from_bytes([
    0x6b, 0xa7, 0xb8, 0x10, 0x9d, 0xad, 0x11, 0xd1, 0x80, 0xb4, 0x00, 0xc0, 0x4f, 0xd4, 0x30, 0xc8,
]);

/// Stable UUID for a memory namespace.
pub fn namespace_uuid(namespace: &str) -> Uuid {
    Uuid::new_v5(&DNS_NAMESPACE, namespace.as_bytes())
}

/// Stable record id. Tenant is folded in so the same natural key in two tenants
/// never collides — a collision would be a cross-tenant leak (C12).
pub fn record_id(scope: &Scope, natural_key: &str) -> Uuid {
    let ns = namespace_uuid(&scope.namespace);
    Uuid::new_v5(&ns, format!("{}\u{1f}{}", scope.tenant, natural_key).as_bytes())
}
