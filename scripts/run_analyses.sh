#!/usr/bin/env bash
# Regenerate every committed result file in experiments/ntc from the frozen
# probe streams. CPU only; no network access and no model weights are needed.
#
#   bash scripts/run_analyses.sh
set -euo pipefail
cd "$(dirname "$0")/.."
NTC=experiments/ntc
AGG="python scripts/ntc_genseed_agg.py"

probes () { for f in "$@"; do printf -- "--probes %s/%s.json " "$NTC" "$f"; done; }

# --- per-setting aggregates over generation seeds ------------------------
$AGG $(probes w1_gsm8k_Qwen3-4B w1_gsm8k_Qwen3-4B_s43 w1_gsm8k_Qwen3-4B_s44) \
     --tag gsm8k_Qwen3-4B
$AGG $(probes w1_gsm8k_Qwen3-8B w1_gsm8k_Qwen3-8B_s43 w1_gsm8k_Qwen3-8B_s44) \
     --tag gsm8k_Qwen3-8B
$AGG $(probes w1_math500_Qwen3-1.7B w1_math500_Qwen3-1.7B_s43 w1_math500_Qwen3-1.7B_s44) \
     --tag math500_Qwen3-1.7B
$AGG $(probes w1_math500_Qwen3-4B w1_math500_Qwen3-4B_s43 w1_math500_Qwen3-4B_s44) \
     --tag math500_Qwen3-4B
$AGG $(probes w1_math500_Qwen3-8B w1_math500_Qwen3-8B_s43 w1_math500_Qwen3-8B_s44) \
     --tag math500_Qwen3-8B
$AGG $(probes w1_mmlupro_Qwen3-4B_s42 w1_mmlupro_Qwen3-4B_s43 w1_mmlupro_Qwen3-4B_s44) \
     --tag mmlupro_Qwen3-4B
$AGG $(probes w1_mmlupro_Qwen3-8B_s42 w1_mmlupro_Qwen3-8B_s43 w1_mmlupro_Qwen3-8B_s44) \
     --tag mmlupro_Qwen3-8B
$AGG $(probes w1sh_gpqa_Qwen3-4B w1sh_gpqa_Qwen3-4B_s43 w1sh_gpqa_Qwen3-4B_s44) \
     --tag gpqash_Qwen3-4B
$AGG $(probes w1sh_gpqa_Qwen3-8B w1sh_gpqa_Qwen3-8B_s43 w1sh_gpqa_Qwen3-8B_s44) \
     --tag gpqash_Qwen3-8B
$AGG $(probes w1_aime24_Qwen3-4B_s100 w1_aime24_Qwen3-4B_s101 w1_aime24_Qwen3-4B_s102 \
              w1_aime24_Qwen3-4B_s103 w1_aime24_Qwen3-4B_s104 w1_aime24_Qwen3-4B_s105 \
              w1_aime24_Qwen3-4B_s106 w1_aime24_Qwen3-4B_s107) --pool-calib --tag aime24_avg8
$AGG $(probes w1_aime25_Qwen3-4B_s100 w1_aime25_Qwen3-4B_s101 w1_aime25_Qwen3-4B_s102 \
              w1_aime25_Qwen3-4B_s103 w1_aime25_Qwen3-4B_s104 w1_aime25_Qwen3-4B_s105 \
              w1_aime25_Qwen3-4B_s106 w1_aime25_Qwen3-4B_s107) --pool-calib --tag aime25_avg8

# --- everything that reads those aggregates or the probe files directly ---
python scripts/ntc_slo_report.py
python scripts/ntc_primary_stats.py
python scripts/ntc_dissertation_numbers.py
python scripts/ntc_prop2_validate.py
python scripts/ntc_prop2_within.py
python scripts/ntc_cost_regimes.py
python scripts/ntc_h2h_table.py
python scripts/ntc_ckpt_density.py
python scripts/ntc_shift_certificate.py
python scripts/ntc_fusion_transfer.py
python scripts/ntc_tail_price.py
python scripts/ntc_v2_diagnostic.py --probes $NTC/w1_math500_Qwen3-4B.json || true
python scripts/ntc_data_inventory.py
python scripts/ntc_corpus_tracks.py
python scripts/ntc_grader_check.py
echo "All result files regenerated under $NTC."
