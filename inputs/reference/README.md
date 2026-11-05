# Reference inputs

| File | Content | Read by |
|---|---|---|
| `bill_impacts_no_esh_threshold.csv` | The class bill-impact table of the notebook (same layout as `notebook_only/layer13_peed_dr/table5_bill_impacts.csv`) under the billing convention without the ESH consumption threshold: every ESH winter month at ₴2.20/kWh. Only its consumption, average-tariff, bill and passive-change columns are used; its shift and combined columns are not. | `run_eulp_bill_distribution.py`: case L of the within-class analysis reproduces it (gate C, `07_aggregation_identity.csv`). |
