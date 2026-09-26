#!/usr/bin/env bash
# Extract the GPQA-Diamond probe files from their password-protected archive.
#
# The authors of GPQA ask that its questions not be posted online in plain text,
# so the eight probe files that contain them are distributed in the encrypted
# archive experiments/ntc/gpqa_diamond.zip. The password is available from the
# author on request (abraheemrashid@outlook.com).
#
#   GPQA_PASSWORD=<password> bash scripts/unpack_gpqa.sh
set -euo pipefail
cd "$(dirname "$0")/../experiments/ntc"
FILES=(h2h2_gpqa_Qwen3-4B.json h2h2_gpqa_Qwen3-8B.json
       w1sh_gpqa_Qwen3-4B.json w1sh_gpqa_Qwen3-4B_s43.json w1sh_gpqa_Qwen3-4B_s44.json
       w1sh_gpqa_Qwen3-8B.json w1sh_gpqa_Qwen3-8B_s43.json w1sh_gpqa_Qwen3-8B_s44.json)

missing=0
for f in "${FILES[@]}"; do [ -f "$f" ] || missing=1; done
if [ "$missing" -eq 0 ]; then
  echo "GPQA-Diamond probe files are present."
  exit 0
fi
if [ -z "${GPQA_PASSWORD:-}" ]; then
  cat >&2 <<'MSG'
The GPQA-Diamond probe files are distributed in the password-protected archive
experiments/ntc/gpqa_diamond.zip, because the authors of GPQA ask that its
questions not be posted online in plain text. To obtain the password, email
abraheemrashid@outlook.com, then run:

    GPQA_PASSWORD=<password> make all
MSG
  exit 1
fi
if ! unzip -q -o -P "$GPQA_PASSWORD" gpqa_diamond.zip "${FILES[@]}"; then
  rm -f "${FILES[@]}"
  echo "Could not extract gpqa_diamond.zip: the password is incorrect." >&2
  exit 1
fi
echo "Extracted ${#FILES[@]} GPQA-Diamond probe files."
