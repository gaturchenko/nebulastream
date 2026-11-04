#!/usr/bin/env bash
# Usage: resolve-external-data.sh [--verify] [ROOT ...]
#   ROOTs default to $NAVI_DIR/models-benchmarking/models, $NAVI_DIR/datasets, and
#   $NAVI_DIR/bin (NAVI_DIR defaults to ~/navi). Files resolved into a bin/ directory
#   (e.g. the systest binary) get their executable bit restored after download.
#   --verify  re-hash targets that already exist; by default an existing file is trusted,
#             since downloads land via a temp file and are only renamed after verification.
set -euo pipefail

NAVI_DIR="${NAVI_DIR:-$HOME/navi}"

# Same shares, order, and timeout as ExternalData_URL_TEMPLATES / ExternalData_TIMEOUT_ABSOLUTE.
SHARES=(28Tr2wTd73Ggeed ZkAMpxFBEeNAnzS aQnjYag7mxxjJaM)
TIMEOUT_ABSOLUTE=1200

VERIFY=false
ROOTS=()
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY=true ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) ROOTS+=("$arg") ;;
  esac
done
if [[ ${#ROOTS[@]} -eq 0 ]]; then
  ROOTS=("$NAVI_DIR/models-benchmarking/models" "$NAVI_DIR/datasets" "$NAVI_DIR/bin")
fi

command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 1; }
command -v md5sum >/dev/null || { echo "ERROR: md5sum is required" >&2; exit 1; }

md5_of() { md5sum "$1" | cut -d' ' -f1; }

# fetch_object <hash> <destination>: try each share in order, verify md5, atomically rename.
fetch_object() {
  local hash="$1" dest="$2" tmp="$2.fetching" share url
  for share in "${SHARES[@]}"; do
    url="https://tubcloud.tu-berlin.de/s/${share}/download?files=MD5_${hash}"
    if ! curl -fsL --retry 3 --max-time "$TIMEOUT_ABSOLUTE" -o "$tmp" "$url"; then
      rm -f "$tmp"
      continue
    fi
    if [[ "$(md5_of "$tmp")" == "$hash" ]]; then
      mv "$tmp" "$dest"
      return 0
    fi
    # Wrong content (e.g. an HTML error page served with HTTP 200) — try the next share.
    rm -f "$tmp"
  done
  return 1
}

# extract_saved_model <tar> <dest-dir>: mirror of cmake/ExtractExternalDataArchive.cmake.
extract_saved_model() {
  local archive="$1" dest="$2" staging="$2.extracting"
  rm -rf "$staging" "$dest"
  mkdir -p "$staging"
  tar -xf "$archive" -C "$staging"
  if [[ -d "$staging/saved_model" ]]; then
    mv "$staging/saved_model" "$dest"
  else
    local entries=("$staging"/*)
    if [[ ${#entries[@]} -eq 1 && -d "${entries[0]}" ]]; then
      mv "${entries[0]}" "$dest"
    else
      mkdir -p "$dest"
      mv "${entries[@]}" "$dest/"
    fi
  fi
  rm -rf "$staging"
  touch "$dest.extract-stamp"
}

failures=0
fetched=0
skipped=0

for root in "${ROOTS[@]}"; do
  [[ -d "$root" ]] || { echo "WARNING: skipping missing root $root" >&2; continue; }
  while IFS= read -r -d '' link; do
    target="${link%.md5}"
    hash="$(tr -d '[:space:]' < "$link")"
    if [[ ! "$hash" =~ ^[0-9a-fA-F]{32}$ ]]; then
      echo "ERROR: $link does not contain a valid md5 hash" >&2
      failures=$((failures + 1))
      continue
    fi
    hash="${hash,,}"

    need_fetch=true
    if [[ -f "$target" ]]; then
      if ! $VERIFY || [[ "$(md5_of "$target")" == "$hash" ]]; then
        need_fetch=false
      else
        echo "STALE  ${target#"$root"/} (checksum mismatch, re-downloading)"
      fi
    fi

    if $need_fetch; then
      echo "FETCH  ${target#"$root"/}"
      if fetch_object "$hash" "$target"; then
        fetched=$((fetched + 1))
      else
        echo "ERROR: could not fetch MD5_${hash} for $target from any share" >&2
        failures=$((failures + 1))
        continue
      fi
    else
      skipped=$((skipped + 1))
    fi

    # Executables (systest et al.) lose their exec bit on the cloud round-trip.
    if [[ "${target%/*}" == */bin ]]; then
      chmod +x "$target"
    fi

    # SavedModel archives are unpacked next to the tar
    if [[ "$(basename "$target")" == "saved_model.tar" ]]; then
      dest="${target%.tar}"
      if $need_fetch || [[ ! -d "$dest" ]] || [[ "$target" -nt "$dest.extract-stamp" ]]; then
        echo "UNTAR  ${dest#"$root"/}"
        extract_saved_model "$target" "$dest"
      fi
    fi
  done < <(find "$root" -name '*.md5' -type f -print0 | sort -z)
done

echo "Done: $fetched fetched, $skipped up-to-date, $failures failed."
[[ $failures -eq 0 ]]
