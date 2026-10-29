# Uncoupled Linear Regression using Optimal Transport

Research code and numerical experiments for estimating linear relationships from **unpaired covariate and response samples** using optimal transport. The project studies moment-based Bures–Wasserstein (BW) maps, estimation of unknown noise levels, and uncertainty quantification in one and multiple dimensions.

The repository includes simulation studies and applications to baseball data, stellar abundances, unlabeled sensing, treatment-effect variance estimation, and single-cell perturbation data.

## Overview

Classical regression uses paired observations of predictors and responses. In uncoupled regression, only the two samples are observed; their individual correspondences are unavailable. The underlying model can be written as

$$
Y \overset{d}{=} AX+b+\varepsilon,
$$

where the noise is independent of the covariates. The goal is to learn the relationship from the marginal distributions of $X$ and $Y$.

The BW approach uses means and noise-adjusted covariance matrices to construct an affine transport map. Recovering the underlying regression map requires identifiability assumptions, such as a positive slope in one dimension or a symmetric positive-definite coefficient matrix in the multivariate setting. The experiments also examine more general settings through comparisons with deconvolution, nonparametric transport, and independent component analysis.

The numerical studies address three main questions:

- **Estimation:** How accurately can the transport map and noise level be estimated as sample size increases?
- **Inference:** How do asymptotic normal approximations and bootstrap procedures perform, including when noise must be estimated?
- **Applications:** How can BW maps support distributional calibration, covariance recovery, and matching in applied problems?

## Repository guide

| Location | Contents |
|---|---|
| [`figures/`](figures/) | Saved figures from the simulation studies |
| [`code/matlab/`](code/matlab/) | Core simulations for map estimation, noise estimation, convergence rates, and confidence regions |
| [`code/baseball/`](code/baseball/) | Baseball analyses in R |
| [`code/apogee/`](code/apogee/) | APOGEE stellar-abundance analysis in Python |
| [`code/unlabeled_sensing/`](code/unlabeled_sensing/) | Unlabeled-sensing experiments, CSV results, and LaTeX writeup |
| [`code/acic/`](code/acic/) | Treatment-effect variance experiment and its figure |
| [`code/single_cell_kang/`](code/single_cell_kang/) | Single-cell perturbation experiment and saved results |

## Simulation studies

The table below connects each numbered figure in `figures/` to its generating script. Script links are relative to the repository root.

| Figure | Experiment | Generating script |
|---|---|---|
| **fig1** | Parametric-rate comparison using the true noise level, a fourth-cumulant noise estimator, and NPMLE profile estimation | [`noise_estimation_linear_map.m`](code/matlab/noise_estimation_linear_map.m) |
| **fig2** | Linear-model rate comparison: raw sorting, variance-constrained empirical Bayes followed by sorting, and the BW moment estimator | [`uncoupled_linear_regression_EB.m`](code/matlab/uncoupled_linear_regression_EB.m) |
| **fig3** | Central limit theorem and bootstrap 95% pointwise bands with known noise | [`code_1.m`](code/matlab/code_1.m) |
| **fig4** | Coverage with estimated noise: updated and naive bootstrap procedures | [`code_2.m`](code/matlab/code_2.m) |
| **fig5** | Two-dimensional comparison of NPMLE plus linear-programming OT and parametric BW plus ICA | [`code_3.m`](code/matlab/code_3.m) |
| **fig6** | Convergence rate of the two-dimensional BW map estimator | [`code_4.m`](code/matlab/code_4.m) |
| **fig7** | Rate of the eigendecomposition-based noise estimator with dimension $d=3$ and signal rank $r=2$ | [`code7.m`](code/matlab/code7.m) |
| **fig8** | Multivariate joint-parameter coverage: asymptotic normal approximation and bootstrap | [`code6.m`](code/matlab/code6.m) |
| **fig9** | Rate of the multivariate cumulant-based noise estimator with $d=3$ | [`code8.m`](code/matlab/code8.m) |

The figure-to-script index was reconstructed from plot content and file timestamps. The MATLAB scripts display plots but do not automatically save the corresponding figure files; export plots after running the relevant script.

### Supporting MATLAB files

The `fig1` experiment uses [`uir_1d.m`](code/matlab/uir_1d.m), [`nndensity.m`](code/matlab/nndensity.m), and [`NWcorner.m`](code/matlab/NWcorner.m). Keep these files on the MATLAB path when running `noise_estimation_linear_map.m`.

Two additional scripts cover related experiments without an assigned numbered figure in this index:

- [`code5.m`](code/matlab/code5.m): multidimensional noise-scale estimation, associated with Section 6.3.
- [`cod9.m`](code/matlab/cod9.m): joint-parameter convergence rates, associated with Section 6.4. The filename is `cod9.m`.

## Applications

| Application | Entry point | Outputs and supporting material |
|---|---|---|
| **Baseball** | [`simple_baseball.R`](code/baseball/simple_baseball.R) | `baseball_estimates.png` and `baseball_variance_risk.png`. [`lahman_og_vs_ebvcb.R`](code/baseball/lahman_og_vs_ebvcb.R) contains a related Lahman salary comparison. |
| **APOGEE stellar abundances** | [`apogee_bw_analysis.py`](code/apogee/apogee_bw_analysis.py) | `apogee_spectrum.png` |
| **Unlabeled sensing** | [`unlabeled_sensing.py`](code/unlabeled_sensing/unlabeled_sensing.py) | `unlabeled_sensing_comparison.png`, `unlabeled_sensing_dimension.png`, and `unlabeled_sensing_diagnostics.png`; CSV results and LaTeX are in the same application directory. |
| **Treatment-effect variance estimation** | [`acic_ate_variance.py`](code/acic/acic_ate_variance.py) | Experiment figure in `code/acic/` |
| **Single-cell perturbation** | [`kang_bw_experiment.py`](code/single_cell_kang/kang_bw_experiment.py) | Saved outputs in [`results/`](code/single_cell_kang/results/). The large `.h5ad` input is excluded; the script downloads it again when run. |

## Running the experiments

### Requirements

Use **MATLAB** for the core simulations, **R** for the baseball analyses, and **Python** for the other applications. Package, toolbox, solver, and data requirements depend on the selected experiment; inspect its imports and setup code before running it.

### MATLAB example

From the repository root, run the two-dimensional map-rate experiment with:

```matlab
cd('code/matlab')
addpath(pwd)
run('code_4.m')
```

Replace `code_4.m` with a script from the figure index to run another experiment. Review the sample sizes, number of simulation replications, and bootstrap settings in the script before starting a larger run. Export the resulting plots from MATLAB to retain the figures.

### R and Python entry points

The application scripts can be launched from the repository root with commands such as:

```bash
Rscript code/baseball/simple_baseball.R
python code/apogee/apogee_bw_analysis.py
python code/unlabeled_sensing/unlabeled_sensing.py
python code/acic/acic_ate_variance.py
python code/single_cell_kang/kang_bw_experiment.py
```

Check each script's working-directory assumptions and input/output paths before running it, and adjust local paths where necessary. The single-cell experiment requires network access to retrieve its input data.

### Reproduction notes

- Experiments are organized as individual research scripts; run the script associated with the figure or application of interest.
- For stochastic experiments, record the random seed and simulation settings when comparing runs. Regenerated results may differ when these settings change.
- Keep downloaded data and generated outputs in the locations expected by the corresponding script.
- The numbered figure index refers to the saved simulation outputs. Application figures are listed separately above.

## Abbreviations

| Abbreviation | Meaning |
|---|---|
| BW | Bures–Wasserstein |
| OT | Optimal transport |
| EB | Empirical Bayes |
| VCB | Variance-constrained empirical Bayes |
| NPMLE | Nonparametric maximum likelihood estimation |
| ICA | Independent component analysis |
| CLT | Central limit theorem |
| LP | Linear programming |
