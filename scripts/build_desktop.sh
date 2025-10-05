#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIST_DIR="$ROOT_DIR/dist"
UI_DIR="$ROOT_DIR/apps/ui"
BACKEND_ENTRY="$ROOT_DIR/backend/main.py"
PYTHON_BIN=${PYTHON_BIN:-python3}

mkdir -p "$DIST_DIR"

read_version() {
  "$PYTHON_BIN" - <<PY
import pathlib
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore
root = pathlib.Path(r"$ROOT_DIR")
with (root / "pyproject.toml").open("rb") as handle:
    data = tomllib.load(handle)
print(data["tool"]["poetry"]["version"])
PY
}

VERSION=$(read_version)

map_platform() {
  local uname_out
  uname_out=$(uname -s | tr '[:upper:]' '[:lower:]')
  case "$uname_out" in
    linux*)
      echo "linux"
      ;;
    darwin*)
      echo "darwin"
      ;;
    msys*|mingw*|cygwin*|windows*)
      echo "windows"
      ;;
    *)
      echo "";
      ;;
  esac
}

PLATFORM=$(map_platform)
if [[ -z "$PLATFORM" ]]; then
  echo "[ERROR] Unsupported platform: $(uname -s)" >&2
  exit 1
fi

CONFIG_PATH="$ROOT_DIR/installer/${PLATFORM}.json"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[ERROR] Electron Builder config not found: $CONFIG_PATH" >&2
  exit 1
fi

verify_artifacts() {
  shopt -s nullglob
  local matches=("$DIST_DIR"/ketabmind-desktop-"${VERSION}"*)
  shopt -u nullglob
  if (( ${#matches[@]} == 0 )); then
    echo "[ERROR] No installer artifacts found for version $VERSION in $DIST_DIR" >&2
    exit 1
  fi

  local file_count=0
  for artifact in "${matches[@]}"; do
    if [[ -f "$artifact" ]]; then
      echo "[INFO] Found installer artifact: $(basename "$artifact")"
      ((file_count += 1))
    fi
  done

  if (( file_count == 0 )); then
    echo "[ERROR] Installer artifacts located in $DIST_DIR but none are regular files" >&2
    exit 1
  fi

  return 0
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  placeholder="$DIST_DIR/ketabmind-desktop-${VERSION}-${PLATFORM}.placeholder"
  echo "[DRY RUN] Creating placeholder artifact: $placeholder"
  > "$placeholder"
  verify_artifacts
  exit 0
fi

echo "[INFO] Building KetabMind Desktop version $VERSION for $PLATFORM"

echo "[INFO] Building UI"
pushd "$UI_DIR" > /dev/null
npm run build
popd > /dev/null

PYINSTALLER_WORK="$DIST_DIR/.pyinstaller-build"
PYINSTALLER_SPEC="$DIST_DIR/.pyinstaller-spec"
rm -rf "$PYINSTALLER_WORK" "$PYINSTALLER_SPEC"

pyinstaller "$BACKEND_ENTRY" --onefile --name "ketabmind-desktop" \
  --distpath "$DIST_DIR" --workpath "$PYINSTALLER_WORK" --specpath "$PYINSTALLER_SPEC"

if [[ -f "$DIST_DIR/ketabmind-desktop" ]]; then
  mv "$DIST_DIR/ketabmind-desktop" "$DIST_DIR/ketabmind-desktop-${VERSION}"
elif [[ -f "$DIST_DIR/ketabmind-desktop.exe" ]]; then
  mv "$DIST_DIR/ketabmind-desktop.exe" "$DIST_DIR/ketabmind-desktop-${VERSION}.exe"
fi

rm -rf "$PYINSTALLER_WORK" "$PYINSTALLER_SPEC"

electron-builder --config "$CONFIG_PATH" \
  --config.extraMetadata.version="$VERSION" \
  --config.buildVersion="$VERSION" \
  --config.directories.output="$DIST_DIR"

verify_artifacts

echo "[INFO] Build completed. Artifacts are located in $DIST_DIR"
