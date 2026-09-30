#!/bin/bash
# Copy one Qdrant collection between two instances on this host through a
# snapshot: create it on the source, download it into myelin's snapshot
# directory, recover it on the destination, compare point counts, and delete
# the snapshot this script created on the source.
#
#   bash ops/big/qdrant-copy.sh <collection> [source_rest] [dest_rest]
#
# Defaults copy from home-still's shared instance (:6333) into myelin's own
# (:6433). Snapshot recovery copies the stored vectors as they are, so nothing
# is re-embedded and the copy is exact (measured 2026-09-23: 162,181 points in
# seconds, against ~57 minutes to re-embed).
set -euo pipefail
C=${1:?collection}
SRC=${2:-http://127.0.0.1:6333}
DST=${3:-http://127.0.0.1:6433}
SNAP_DIR="$HOME/myelin-qdrant/snapshots"
count() { curl -sf "$1/collections/$2" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["points_count"])'; }
src_n=$(count "$SRC" "$C")
name=$(curl -sf -X POST "$SRC/collections/$C/snapshots?wait=true" | python3 -c 'import json,sys; print(json.load(sys.stdin)["result"]["name"])')
mkdir -p "$SNAP_DIR/$C"
curl -sf -o "$SNAP_DIR/$C/$name" "$SRC/collections/$C/snapshots/$name"
curl -sf -X DELETE "$SRC/collections/$C/snapshots/$name?wait=true" > /dev/null
curl -sf -X PUT "$DST/collections/$C/snapshots/recover?wait=true" -H 'Content-Type: application/json' \
  -d "{\"location\":\"file:///qdrant/snapshots/$C/$name\",\"priority\":\"snapshot\"}" > /dev/null
dst_n=$(count "$DST" "$C")
if [ "$src_n" != "$dst_n" ]; then
  echo "$C: point counts differ after the copy: source $src_n, destination $dst_n" >&2
  exit 1
fi
echo "$C: $dst_n points copied ($(du -h "$SNAP_DIR/$C/$name" | cut -f1) snapshot $name)"
