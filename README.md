# Class-specific residential time-of-use tariffs for Ukraine — code

The code and inputs used to compute the results of the article *Designing a class-specific residential
time-of-use tariff for demand flexibility in Ukraine* (Dmytro Monakov, Taras Shevchenko National University of
Kyiv; ORCID [0009-0007-9472-960X](https://orcid.org/0009-0007-9472-960X)).

| Data | DOI |
|---|---|
| Source dataset of the archetypal load profiles (ARLPs) and end-use load profiles (EULPs) | [10.5281/zenodo.20386981](https://doi.org/10.5281/zenodo.20386981) |
| Data of this study | [10.5281/zenodo.23988472](https://doi.org/10.5281/zenodo.23988472) |

## Installation

Tested with Python 3.10.10 (64-bit) on Windows 11.

```
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Data

The small inputs are in `inputs/`. Four files are not in the repository; download them and put them in `inputs/`:

| File | Content | Source |
|---|---|---|
| `EULP_4435_hourly_load.csv` | hourly loads of the 4,435 weighted EULPs (350 MB) | source dataset, file `EULP_4435_hourly_load.rar` (unpack it, e.g. with 7-Zip) |
| `DAM_2019-2025_v05_Full.xlsx` | day-ahead-market prices 2019–2025 with the network tariffs and the exchange rate | data of this study (the version published with this code; earlier versions hold an older workbook) |
| `BASE_M_2024_Ukraine.xlsx`, `BASE_M_2025_Ukraine.xlsx` | BASE_M prices of the supply months 2024 and 2025 (DAM BASE index of month M−3) | data of this study |

## Run

```
.venv\Scripts\python run_all.py
```

About 30 minutes. The steps:

1. `build_dam_prices_uah.py` and 2. `build_base_m_prices_uah.py` rebuild the price inputs
   `inputs/hourly_dam_prices_uah.csv` and `inputs/base_m_prices_uah.csv` from the workbooks;
3. the notebook `ARLP_clustering_analysis_autonomous.ipynb`: clustering, tariff classes, proposed tariff, demand
   response, bills, PSO compensation and supplier balance, with its built-in checks against the article (the
   executed copy is written to `outputs/ARLP_Clustering/notebook_run/`);
4. `run_eulp_bill_distribution.py`: the within-class distribution of the passive bill changes;
5. `run_eulp_bill_response.py`: the shift and combined effects within the classes;
6. `run_robustness_eulp.py` and 7. `make_robustness_outputs.py`: the sensitivity of the tariff multipliers.

Every result is written to `outputs/ARLP_Clustering/`. Each runner also writes its quality checks
(`11_qa_gates.csv`, `07_qa_gates.csv`, `08_gates.csv`) and stops with a failed one.

## Tables and figures of the article

Paths below `outputs/ARLP_Clustering/`; all code in `code/`.

| Item | § | File | Code |
|---|---|---|---|
| Fig. 1 | 3 | — (diagram) | — |
| Table 1 | 3.1 | — (data table) | household counts `inputs/arlp_household_counts.csv`; other columns from the source publication, typed in the notebook (`TABLE2`) |
| Table 2 | 3.6 | `notebook_only/layer13_peed_sensitivity/peed_source_registry.csv` | published elasticities in `inputs/peed_matrices_source_specific/` (`SOURCES.md`) |
| Fig. 2 | 4.1 | `non_esh/lens1_compositional/jsd_heatmap_recolored.png` | `make_jsd_heatmap.py` |
| Table 3 | 4.1 | — (displayed in the notebook, `TABLE4`) | notebook, `arlp_families.py` |
| Fig. 3 | 4.1 | `tou_annotation/representative_load_shapes_grid.png` | `make_representative_load_shapes.py` |
| Fig. 4 | 4.2 | `non_esh/lens2_families_kselection/kselection_grid.png` | `make_kselection_grid.py`, `k_selection.py` |
| Fig. 5 | 4.2 | `tou_annotation/pca_maps_combined.png` | `make_pca_maps_combined.py` |
| Fig. 6 | 4.3 | — (diagram; the household and energy shares are computed in the notebook, `TABLE6`) | notebook |
| Fig. 7 | 4.3 | `notebook_figures/assignment_tree_full.png` | `make_assignment_tree_full.py`, `make_assignment_tree.py` |
| Table 4 | 4.4 | `notebook_only/layer13_peed_dr/table4_proposed_tariff.csv` | notebook |
| Fig. 8 | 4.5.1 | `notebook_only/layer13_peed_dr/price_signals_grid.png` | `make_price_signals_grid.py` |
| Fig. 9 | 4.5.2 | `notebook_only/layer13_peed_sensitivity/peed_source_loadshift_panels.png` | `make_peed_source_loadshift.py` |
| Table 5 | 4.5.2 | `notebook_only/layer13_peed_sensitivity/peed_sensitivity_summary.csv`, `peed_sensitivity_peak.csv` | notebook, `peed_sensitivity.py` |
| Table 6 | 4.5.4 | `notebook_only/layer13_peed_dr/table5_bill_impacts.csv` (Panel A), `notebook_only/layer13_peed_sensitivity/peed_sensitivity_retailer.csv` (Panels B–C) | notebook, `peed_sensitivity.py` |
| Table 7 | 4.5.5 | `notebook_only/layer13_peed_sensitivity/peed_sensitivity_retailer.csv`, `peed_pso_levels.csv` | notebook, `peed_pso.py` |
| Fig. 10 | 4.5.5 | `notebook_only/layer13_peed_sensitivity/peed_retailer_lambda.png` | `make_peed_retailer_lambda.py` |
| Fig. 11 | 4.5.6 | `notebook_only/layer13_peed_sensitivity/participation_trajectory.png` | `make_participation_trajectory.py`, `peed_rollout_retailer.py` |
| Fig. 12 | 4.6 | `eulp_bill_response/fig_passive_both_cohorts_histograms.png` | `run_eulp_bill_response.py` |
| Fig. 13 | 4.6 | `eulp_bill_response/fig_shift_both_cohorts_histograms.png` | `run_eulp_bill_response.py` |
| Fig. 14 | 4.6 | `eulp_bill_response/fig_combined_both_cohorts_histograms.png` | `run_eulp_bill_response.py` |
| Fig. 15 | 5.2 | `robustness_eulp/fig_coefficient_deviations.png` | `run_robustness_eulp.py`, `make_robustness_outputs.py` |
| Fig. 16 | 5.2 | `robustness_eulp/fig_ls1_timing.png` | `run_robustness_eulp.py`, `make_robustness_outputs.py` |
| Fig. 17 | 5.2 | `robustness_eulp/fig_ls2_one_class.png` | `run_robustness_eulp.py`, `make_robustness_outputs.py` |
| Fig. 18 | 5.2 | `robustness_eulp/fig_ab_regimes.png` | `run_robustness_eulp.py`, `make_robustness_outputs.py` |

Printed values that are arithmetic on the files:

- Table 5: the change in maximum load is (`max_MW` − `base_max_MW`) / `base_max_MW` × 100 of
  `peed_sensitivity_peak.csv` (series `combined P3` and `combined E3`, p_t = 1).
- Table 6: positive values are bill increases. Panel A is minus the passive columns of `table5_bill_impacts.csv`
  (stored as savings); Panels B and C are 100 × `response_MUAH` / `base_revenue_MUAH` and
  100 × `total_retail_MUAH` / `base_revenue_MUAH` of `peed_sensitivity_retailer.csv`.
- Table 7: Panel A is `dV_PSO_MUAH` and `dPi_MUAH` of `peed_sensitivity_retailer.csv` in ₴ bn; Panel B divides them
  by `V0_MUAH` and by |`Pi0_MUAH`| of `peed_pso_levels.csv` (row `total`, convention `stamp`), × 100.

Some output files keep the names they were given when they were first written (for example
`table5_bill_impacts.csv` holds the bill impacts of Table 6); the notebook's section headings follow the article.

## Notes

- Money is in Ukrainian hryvnia (₴), excluding VAT.
- The clustering uses `RANDOM_STATE = 42` (k-medoids, PAM); the sensitivity study draws its replicates with the
  seeds of `code/robustness_registry.csv`.
- The code locates the folder that holds `inputs/` by walking up from its own folder, so `code/` and `inputs/`
  must stay side by side; the results go to `outputs/` next to them.

## License and citation

Code: MIT (`LICENSE`). Please cite the article and this repository (`CITATION.cff`).
