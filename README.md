# Uncoupled Linear Regression using Optimal Transport

All files are copies; originals remain in Downloads.

## figures/ -> generating code
| Figure | Content | Script (code/matlab/) |
|---|---|---|
| fig1 | Parametric rate: oracle sigma / 4th cumulant / NPMLE profile | noise_estimation_linear_map.m (uses uir_1d.m, nndensity.m, NWcorner.m) |
| fig2 | Parametric-rate check, linear case (sort raw / VCB+sort / BW moment) | uncoupled_linear_regression_EB.m |
| fig3 | CLT vs bootstrap 95% pointwise bands (known noise) | code_1.m |
| fig4 | Coverage, estimated noise: updated vs naive bootstrap | code_2.m |
| fig5 | 2D: NPMLE+LP OT vs parametric BW + ICA | code_3.m |
| fig6 | 2D BW map convergence rate | code_4.m |
| fig7 | Eigendecomposition noise estimator rate (d=3, r=2) | code7.m |
| fig8 | Joint parameter coverage, multivariate CLT vs bootstrap | code6.m |
| fig9 | Multivariate cumulant noise estimator rate (d=3) | code8.m |

code5.m and cod9.m (multidimensional noise scale, Sec 6.3; joint-parameter rate, Sec 6.4) are related but no figure matched them.
Figure-to-script mapping was made by matching plot titles/content and file timestamps; the .m scripts do not save the images themselves.

## Other figures -> code
- baseball_estimates.png, baseball_variance_risk.png -> code/baseball/simple_baseball.R (lahman_og_vs_ebvcb.R is a related Lahman salary comparison)
- apogee_spectrum.png -> code/apogee/apogee_bw_analysis.py
- unlabeled_sensing_{comparison,dimension,diagnostics}.png -> code/unlabeled_sensing/ (unlabeled_sensing.py, CSVs, LaTeX)
- code/acic/ : acic_ate_variance.py + its png
- code/single_cell_kang/ : kang_bw_experiment.py, results/ (h5ad input removed; script re-downloads it)
