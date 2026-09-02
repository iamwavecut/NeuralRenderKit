#!/usr/bin/env bash
set -euo pipefail

tree="${1:-.}"
if [[ "$(git -C "$tree" rev-parse --is-inside-work-tree 2>/dev/null || true)" != "true" ]]; then
  echo "public tree audit: not a git worktree: $tree" >&2
  exit 2
fi

tree="$(cd "$tree" && pwd -P)"
maximum_file_bytes=5242880
maximum_fixture_bytes=1048576
failed=0
mac_home_prefix="/""Users/"

report() {
  local relative_path="$1"
  local reason="$2"
  printf 'forbidden: %q (%s)\n' "$relative_path" "$reason"
  failed=1
}

file_size() {
  local path="$1"
  if stat -f '%z' "$path" >/dev/null 2>&1; then
    stat -f '%z' "$path"
  else
    stat -c '%s' "$path"
  fi
}

while IFS= read -r -d '' relative_path; do
  absolute_path="$tree/$relative_path"
  [[ -f "$absolute_path" ]] || continue

  lowercase_path="$(printf '%s' "$relative_path" | tr '[:upper:]' '[:lower:]')"
  size="$(file_size "$absolute_path")"
  magic="$(od -An -tx1 -N4 "$absolute_path" | tr -d '[:space:]')"

  case "$lowercase_path" in
    *.dll)
      report "$relative_path" "proprietary DLL filename"
      ;;
  esac
  case "$lowercase_path" in
    *private-oracle*)
      report "$relative_path" "private oracle filename"
      ;;
  esac
  case "$lowercase_path" in
    *.safetensors)
      case "$relative_path" in
        Tests/Fixtures/Public/*.safetensors)
          if (( size > maximum_fixture_bytes )); then
            report "$relative_path" "allowlisted safetensors fixture exceeds 1 MiB"
          fi
          ;;
        *)
          report "$relative_path" "safetensors are allowed only in Tests/Fixtures/Public"
          ;;
      esac
      ;;
  esac

  if (( size > maximum_file_bytes )); then
    report "$relative_path" "unreviewed file exceeds 5 MiB"
  fi

  case "$magic" in
    4d5a*)
      report "$relative_path" "PE executable magic"
      ;;
    7f454c46*)
      report "$relative_path" "ELF executable magic"
      ;;
    feedface*|cefaedfe*|feedfacf*|cffaedfe*|cafebabe*|bebafeca*|cafebabf*|bfbafeca*)
      report "$relative_path" "Mach-O executable magic"
      ;;
    50ed55ba*)
      report "$relative_path" "CUDA fatbin magic"
      ;;
  esac

  if LC_ALL=C grep -a -q -F "$mac_home_prefix" "$absolute_path"; then
    report "$relative_path" "absolute local macOS home path"
  fi
done < <(git -C "$tree" ls-files -z --cached --others --exclude-standard)

if (( failed != 0 )); then
  exit 1
fi

echo "public tree audit: clean"
