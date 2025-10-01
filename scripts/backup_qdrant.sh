#!/usr/bin/env bash
set -euo pipefail

: "${QDRANT_STORAGE_DIR:?QDRANT_STORAGE_DIR is required}"
: "${UPLOAD_DIR:?UPLOAD_DIR is required}"
: "${QDRANT_BACKUP_DIR:?QDRANT_BACKUP_DIR is required}"

if [[ ! -d "$QDRANT_STORAGE_DIR" ]]; then
  echo "[ERROR] QDRANT_STORAGE_DIR does not exist: $QDRANT_STORAGE_DIR" >&2
  exit 1
fi

if [[ ! -d "$UPLOAD_DIR" ]]; then
  echo "[ERROR] UPLOAD_DIR does not exist: $UPLOAD_DIR" >&2
  exit 1
fi

mkdir -p "$QDRANT_BACKUP_DIR"

timestamp=$(date +%Y%m%d%H%M%S)
backup_name="qdrant_backup_${timestamp}.tar.gz"
backup_path="${QDRANT_BACKUP_DIR%/}/$backup_name"

storage_parent=$(dirname "$QDRANT_STORAGE_DIR")
storage_basename=$(basename "$QDRANT_STORAGE_DIR")
upload_parent=$(dirname "$UPLOAD_DIR")
upload_basename=$(basename "$UPLOAD_DIR")

trap 'rm -f "$backup_path"' ERR

tar -czf "$backup_path" \
  -C "$storage_parent" "$storage_basename" \
  -C "$upload_parent" "$upload_basename"

trap - ERR

echo "[INFO] Backup created at $backup_path"
