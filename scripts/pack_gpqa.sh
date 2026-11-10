#!/usr/bin/env bash
# Re-create the password-protected GPQA-Diamond archive from the eight probe
# files. For maintainers: use it to set or change the archive password. The
# probe files themselves are never committed (see .gitignore).
#
#   GPQA_PASSWORD=<new password> bash scripts/pack_gpqa.sh
#
# The eight files must be present in experiments/ntc/. The existing archive is
# replaced only after the new one has been written and read back.
set -euo pipefail
cd "$(dirname "$0")/../experiments/ntc"
FILES=(h2h2_gpqa_Qwen3-4B.json h2h2_gpqa_Qwen3-8B.json
       w1sh_gpqa_Qwen3-4B.json w1sh_gpqa_Qwen3-4B_s43.json w1sh_gpqa_Qwen3-4B_s44.json
       w1sh_gpqa_Qwen3-8B.json w1sh_gpqa_Qwen3-8B_s43.json w1sh_gpqa_Qwen3-8B_s44.json)

if [ -z "${GPQA_PASSWORD:-}" ]; then
  echo "Set GPQA_PASSWORD to the password the new archive should have." >&2
  exit 1
fi
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Missing probe file: experiments/ntc/$f" >&2
    exit 1
  fi
done

tmp=gpqa_diamond.new.zip
rm -f "$tmp"
zip -q -9 -P "$GPQA_PASSWORD" "$tmp" "${FILES[@]}"
# Read the archive back with the same password before replacing the old one.
unzip -q -t -P "$GPQA_PASSWORD" "$tmp" >/dev/null
mv "$tmp" gpqa_diamond.zip
echo "Wrote experiments/ntc/gpqa_diamond.zip with ${#FILES[@]} files."
echo "Keep the password somewhere safe: it is not stored in the repository."
