# Manuscript source

`main.tex` + `refs.bib` build the TIST submission (ACM `acmsmall`, 25 pages
including references).  Build with `make`, or `latexmk -pdf main.tex`.

Every number in the paper is produced by a script in `../scripts/` from the
frozen probe files in `../experiments/ntc/`:

| paper element | produced by |
|---|---|
| Table 5, Figure 1 (collapse map) | `ntc_genseed_agg.py` -> `GENSEEDS_*.md`, `ntc_slo_report.py` |
| Table 6 (stickiness) | `ntc_prop2_validate.py` |
| Table 7 (within-benchmark) | `ntc_prop2_within.py` |
| Table 8 (primary comparison) | `ntc_primary_stats.py` |
| Table 10 (SLO attainment) | `ntc_slo_report.py` |
| Table 11 (serving regimes) | `ntc_cost_regimes.py` |
| Table 12 (head-to-head vs DEER) | `ntc_h2h_table.py` |
| Table 13 (certificate under shift), Fig. 4a | `ntc_shift_certificate.py` |
| Figure 4b (pricing the tail) | `ntc_tail_price.py` |
| Figure 3 (operating curves) | `ntc_operating_curves.py` |
| joint tier | `ntc_joint_router.py` |

Figures are regenerated with `make figures`.
