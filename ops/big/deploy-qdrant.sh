#!/bin/bash
# Install and start myelin's own Qdrant on big (ops/big/myelin-qdrant.container).
# Run on big from the repository root. Idempotent: re-running reinstalls the
# unit and restarts the container; the storage under ~/myelin-qdrant is kept.
set -euo pipefail
UNIT_DIR="$HOME/.config/containers/systemd"
REST_URL=http://127.0.0.1:6433
READY_TIMEOUT_S=120
mkdir -p "$HOME/myelin-qdrant/storage" "$HOME/myelin-qdrant/snapshots" "$UNIT_DIR"
podman pull -q docker.io/qdrant/qdrant:v1.19.1 > /dev/null
install -m 0644 ops/big/myelin-qdrant.container "$UNIT_DIR/myelin-qdrant.container"
systemctl --user daemon-reload
systemctl --user restart myelin-qdrant.service
for _ in $(seq 1 "$READY_TIMEOUT_S"); do
  if curl -sf "$REST_URL/readyz" > /dev/null; then
    echo "myelin-qdrant ready: $(curl -s "$REST_URL/" | head -c 120)"
    exit 0
  fi
  sleep 1
done
echo "myelin-qdrant did not become ready within ${READY_TIMEOUT_S}s" >&2
systemctl --user status myelin-qdrant.service --no-pager | tail -20 >&2
exit 1
