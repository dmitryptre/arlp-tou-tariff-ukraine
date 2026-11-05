# Source-specific PEED matrices

Three published 2×2 price-elasticity matrices of residential peak and off-peak electricity demand, used one at a time
as source-specific PEED specifications in the autonomous notebook (§4.6.6). They are used instead of the averaged matrix
of `../peed_matrices_residential_avg/` in every result of the article; that folder is kept unchanged as a computational
reference for one regression check only (the same code with the averaged matrix, λ = 1, R, proportional allocation,
must reproduce the notebook's reference objects exactly; no result comes from it).

**Layout** (as the files of `../peed_matrices_residential_avg/`): index `zone_t` = the zone of consumption *i*;
columns `peak`, `off-peak` = the zone of the price change *j*. A cell is ε_ij = ∂ln q_i / ∂ln p_j. Values exactly as published (three decimals for Filippini, two
for Auray et al.); nothing is rounded, averaged or derived. The notebook extends each matrix to the three Ukrainian zones
by the shoulder-inert rule (ε_SS = ½(ε_PP + ε_OO), shoulder cross terms 0; cell 22, `build_3x3`), a modelling assumption
common to all three specifications.

| File | ID | Source | Table, column / row | Estimator and type | ε_PP | ε_PO | ε_OP | ε_OO |
|---|---|---|---|---|---|---|---|---|
| `filippini2011_lsdvc_sr.csv` | F11-SR | Filippini [24] | Table 4, "Short run own price elasticity" and "Short run cross price elasticity", columns LSDVC peak (ε_PP, ε_PO) and LSDVC off-peak (ε_OO, ε_OP); the same coefficients in Table 3 (ln P_EP and ln PE_OP rows of the LSDVC peak and off-peak equations) | dynamic log–log demand equations, LSDV corrected by Kiviet's method (LSDVC); short run; Marshallian | −0.778 | 0.793 | 0.363 | −0.652 |
| `filippini2011_lsdvc_lr.csv` | F11-LR | Filippini [24] | Table 4, "Long run own price elasticity" and "Long run cross price elasticity", columns LSDVC peak and LSDVC off-peak | as F11-SR; long run = short run / (1 − lag coefficient); the author: the long-run values "should be considered carefully" | −2.266 | 2.311 | 0.919 | −1.652 |
| `auray2019_aids_table4.csv` | A19 | Auray et al. [22] | Table 4 ("Price elasticity of electricity demand (Two-price model)"), column "This study": price elasticity peak (ε_PP), off-peak (ε_OO); cross-price elasticity peak/off-peak (ε_PO, peak demand w.r.t. the off-peak price), off-peak/peak (ε_OP) | two-price AIDS budget-share system, homogeneity and symmetry imposed, iterative SUR; conditional on total electricity expenditure (no short-run/long-run status) | −1.47 | 0.46 | 0.85 | −1.87 |

**Orientation.** In [24] the rows of Tables 3–4 are the equations (consumption zone) and the price regressors the
columns; the text (p. 5816) states that the off-peak price affects peak consumption more than the peak price affects
off-peak consumption, so ε_PO > ε_OP. In [22] "peak/off-peak" is peak demand with respect to the off-peak price; the
AIDS formulas confirm it (the larger cross elasticity belongs to the zone with the smaller expenditure share).
`auray2019_aids_table4.csv` is byte-identical to `../peed_matrices_residential_avg/source_auray2019.csv`.

**Citations.**
- [22] S. Auray, V. Caponi, B. Ravel, Price elasticity of electricity demand in France, Écon. Stat. 513 (2019) 91–103.
  https://doi.org/10.24187/ecostat.2019.513.2002
- [24] M. Filippini, Short- and long-run time-of-use price elasticities in Swiss residential electricity demand,
  Energy Policy 39 (2011) 5811–5817. https://doi.org/10.1016/j.enpol.2011.06.002
- Not used in the computation: [23] M. Filippini (1995), Energy Econ. 17, 197–204 (the AIDS model that [22] re-estimates;
  its homothetic estimates, in the averaged matrix of `../peed_matrices_residential_avg/`, are those its author does not
  prefer).

**Verification**: the twelve values and their orientation were read from the text extractions of the papers
(Filippini 2011, Tables 3–4; Auray et al. 2019, Table 4) and equal the values above. `test_peed_sensitivity.py` types
the same values with their table references and checks the files against them.
