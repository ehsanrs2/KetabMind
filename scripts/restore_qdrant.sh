#!/usr/bin/env bash
set -euo pipefail

: "${QDRANT_STORAGE_DIR:?QDRANT_STORAGE_DIR is required}"
: "${UPLOAD_DIR:?UPLOAD_DIR is required}"
: "${QDRANT_BACKUP_DIR:?QDRANT_BACKUP_DIR is required}"

STOP_CMD=${QDRANT_STOP_CMD:-"docker compose stop"}
START_CMD=${QDRANT_START_CMD:-"docker compose up -d"}
HEALTH_URL=${QDRANT_HEALTH_URL:-"http://localhost:6333/readyz"}
HEALTH_TIMEOUT=${QDRANT_HEALTH_TIMEOUT:-60}
HEALTH_INTERVAL=${QDRANT_HEALTH_INTERVAL:-2}

usage() {
  echo "Usage: $0 <backup-tarball>" >&2
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

backup_arg=$1

if [[ -f $backup_arg ]]; then
  backup_path=$backup_arg
elif [[ -f ${QDRANT_BACKUP_DIR%/}/$backup_arg ]]; then
  backup_path=${QDRANT_BACKUP_DIR%/}/$backup_arg
else
  echo "[ERROR] Backup file not found: $backup_arg" >&2
  exit 1
fi

if [[ ! -r $backup_path ]]; then
  echo "[ERROR] Backup file is not readable: $backup_path" >&2
  exit 1
fi

echo "[INFO] Stopping services via: $STOP_CMD"
bash -c "$STOP_CMD"

storage_parent=$(dirname "$QDRANT_STORAGE_DIR")
storage_basename=$(basename "$QDRANT_STORAGE_DIR")
upload_parent=$(dirname "$UPLOAD_DIR")
upload_basename=$(basename "$UPLOAD_DIR")

rm -rf "$QDRANT_STORAGE_DIR" "$UPLOAD_DIR"
mkdir -p "$storage_parent" "$upload_parent"

tar -xzf "$backup_path" -C "$storage_parent" "$storage_basename"
tar -xzf "$backup_path" -C "$upload_parent" "$upload_basename"

echo "[INFO] Starting services via: $START_CMD"
bash -c "$START_CMD"

echo "[INFO] Waiting for Qdrant readiness at $HEALTH_URL (timeout ${HEALTH_TIMEOUT}s)"
end_time=$((SECONDS + HEALTH_TIMEOUT))
while (( SECONDS < end_time )); do
  if curl -fsS "$HEALTH_URL" >/dev/null; then
    echo "[INFO] Qdrant is ready"
    exit 0
  fi
  sleep "$HEALTH_INTERVAL"
done

echo "[ERROR] Timed out waiting for Qdrant readiness" >&2
exit 1
