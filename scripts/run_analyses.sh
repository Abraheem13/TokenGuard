#!/usr/bin/env bash
# Regenerate every result file in experiments/ntc from the committed probe
# streams. Runs on a CPU in a few minutes; needs no GPU, network or model weights.
# The GPQA-Diamond probe files are extracted first from their password-protected
# archive (see scripts/unpack_gpqa.sh).
#
#   GPQA_PASSWORD=<password> bash scripts/run_analyses.sh
set -euo pipefail
cd "$(dirname "$0")/.."
NTC=experiments/ntc
bash scripts/unpack_gpqa.sh
export PYTHONHASHSEED=0

# Expand probe-file stems into repeated command-line flags.
flags () { local flag=$1; shift; for f in "$@"; do printf -- "%s %s/%s.json " "$flag" "$NTC" "$f"; done; }
probes () { flags --probes "$@"; }

# The twelve primary settings, one generation seed each.
PRIMARY=(w1_gsm8k_Qwen3-4B w1_gsm8k_Qwen3-8B
         w1_math500_Qwen3-1.7B w1_math500_Qwen3-4B w1_math500_Qwen3-8B
         w1sh_gpqa_Qwen3-4B w1sh_gpqa_Qwen3-8B
         w1_mmlupro_Qwen3-4B_s42 w1_mmlupro_Qwen3-8B_s42
         w1_aime24_Qwen3-4B_s100 w1_aime25_Qwen3-4B_s100
         w1_math500_DeepSeek-R1-Distill-Qwen-7B)

step () { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

step "Per-setting results over generation seeds (GENSEEDS_*.md)"
agg () { local tag=$1; shift; python scripts/ntc_genseed_agg.py $(probes "$@") --tag "$tag"; }
agg gsm8k_Qwen3-4B     w1_gsm8k_Qwen3-4B     w1_gsm8k_Qwen3-4B_s43     w1_gsm8k_Qwen3-4B_s44
agg gsm8k_Qwen3-8B     w1_gsm8k_Qwen3-8B     w1_gsm8k_Qwen3-8B_s43     w1_gsm8k_Qwen3-8B_s44
agg math500_Qwen3-1.7B w1_math500_Qwen3-1.7B w1_math500_Qwen3-1.7B_s43 w1_math500_Qwen3-1.7B_s44
agg math500_Qwen3-4B   w1_math500_Qwen3-4B   w1_math500_Qwen3-4B_s43   w1_math500_Qwen3-4B_s44
agg math500_Qwen3-8B   w1_math500_Qwen3-8B   w1_math500_Qwen3-8B_s43   w1_math500_Qwen3-8B_s44
agg mmlupro_Qwen3-4B   w1_mmlupro_Qwen3-4B_s42 w1_mmlupro_Qwen3-4B_s43 w1_mmlupro_Qwen3-4B_s44
agg mmlupro_Qwen3-8B   w1_mmlupro_Qwen3-8B_s42 w1_mmlupro_Qwen3-8B_s43 w1_mmlupro_Qwen3-8B_s44
agg gpqash_Qwen3-4B    w1sh_gpqa_Qwen3-4B    w1sh_gpqa_Qwen3-4B_s43    w1sh_gpqa_Qwen3-4B_s44
agg gpqash_Qwen3-8B    w1sh_gpqa_Qwen3-8B    w1sh_gpqa_Qwen3-8B_s43    w1sh_gpqa_Qwen3-8B_s44
python scripts/ntc_genseed_agg.py $(probes w1_aime24_Qwen3-4B_s10{0..7}) --pool-calib --tag aime24_avg8
python scripts/ntc_genseed_agg.py $(probes w1_aime25_Qwen3-4B_s10{0..7}) --pool-calib --tag aime25_avg8

step "Target attainment (SLO_ATTAINMENT.md) and the price of the tail (TAIL_PRICE.md)"
python scripts/ntc_slo_report.py
python scripts/ntc_tail_price.py

step "Operating curves (OPERATING_CURVES.md)"
python scripts/ntc_operating_curves.py $(probes "${PRIMARY[@]}")

step "Primary comparison and minimax regret (PRIMARY_STATS.md)"
python scripts/ntc_primary_stats.py $(for f in "$NTC"/w1_*.json "$NTC"/w1sh_*.json; do printf -- "--probes %s " "$f"; done)

step "Paired tests, overhead, gate diagnostic (DISSERTATION_NUMBERS.md)"
python scripts/ntc_dissertation_numbers.py

step "Error stickiness (PROP2_VALIDATION.md, PROP2_WITHIN.md)"
python scripts/ntc_prop2_validate.py $(probes "${PRIMARY[@]}")
python scripts/ntc_prop2_within.py

step "Serving regimes (COST_REGIMES.md)"
python scripts/ntc_cost_regimes.py $(probes w1_gsm8k_Qwen3-4B w1_gsm8k_Qwen3-8B \
    w1_math500_Qwen3-4B w1_math500_Qwen3-8B w1sh_gpqa_Qwen3-4B w1sh_gpqa_Qwen3-8B)

step "Matched comparison with DEER (H2H_TABLE.md) and checkpoint density (CKPT_DENSITY.md)"
python scripts/ntc_h2h_table.py $(flags --h2h h2h2_math500_Qwen3-4B h2h2_math500_Qwen3-8B \
    h2h2_gpqa_Qwen3-4B h2h2_gpqa_Qwen3-8B h2h2_aime24_Qwen3-4B h2h2_aime24_Qwen3-8B)
python scripts/ntc_ckpt_density.py

step "Certificate and fusion tier under domain shift (SHIFT_CERTIFICATE.md, FUSION_TRANSFER.md)"
python scripts/ntc_shift_certificate.py
python scripts/ntc_fusion_transfer.py

step "Joint tier (JOINT.md)"
python scripts/ntc_joint_router.py --small "$NTC/w1_math500_Qwen3-1.7B.json" \
    --large "$NTC/w1_math500_Qwen3-4B.json" --small-params 1.7 --large-params 4.0

step "Corpus accounting and provenance (CORPUS_TRACKS.md, DATA_INVENTORY.md, PROVENANCE.md)"
python scripts/ntc_corpus_tracks.py
python scripts/ntc_data_inventory.py
python scripts/ntc_provenance.py

step "Grader reproducibility (GRADER_CHECK.md)"
python scripts/ntc_grader_check.py

step "Done: every result file under $NTC has been regenerated."
