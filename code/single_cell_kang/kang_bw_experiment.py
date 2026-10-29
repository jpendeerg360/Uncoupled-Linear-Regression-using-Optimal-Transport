#!/usr/bin/env python3
"""Held-out Bures-Wasserstein experiment on Kang IFN-beta PBMC data.

The script downloads a processed Kang et al. AnnData file once, restricts the
analysis to one cell type, creates independent train/validation/test splits in
the control and stimulated groups, learns a common PCA representation from the
training cells only, and evaluates the transport map on untouched test cells.

Install dependencies with:
    python -m pip install anndata gdown numpy pandas scipy scikit-learn matplotlib

Run with:
    python kang_bw_experiment.py

The processed file used here does not retain donor identifiers. Consequently,
the script reports held-out distribution-prediction metrics, not cell-level
causal effects or donor-level inferential p-values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


GOOGLE_DRIVE_FILE_ID = "1z8gGKQ6oDoi2blCU2IVihKA38h5fORRp"
KNOWN_IFN_GENES = {
    "ISG15", "IFIT1", "IFIT2", "IFIT3", "IFI6", "IFI44", "IFI44L",
    "ISG20", "MX1", "MX2", "OAS1", "OAS2", "OAS3", "STAT1", "IRF7",
    "CXCL10", "CCL8", "LY6E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/kang_normalized_hvg.h5ad"),
        help="Cached AnnData path. It is downloaded only when absent.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/kang_bw")
    )
    parser.add_argument("--cell-type", default="CD14 Mono")
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--sigma-grid-size", type=int, default=31)
    parser.add_argument("--n-projections", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def download_if_needed(path: Path) -> None:
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"Using cached data: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kang data to {path} ...")
    result = gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=str(path), quiet=False)
    if result is None or not path.exists():
        raise RuntimeError("The Kang dataset download failed.")


def split_three_way(
    indices: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return independent 60/20/20 train/validation/test index splits."""
    train, remainder = train_test_split(
        indices, test_size=0.40, random_state=seed, shuffle=True
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=seed + 1, shuffle=True
    )
    return np.sort(train), np.sort(validation), np.sort(test)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def psd_matrix_power(
    matrix: np.ndarray, power: float, floor: float = 0.0
) -> np.ndarray:
    values, vectors = eigh(symmetrize(matrix), check_finite=True)
    values = np.maximum(values, floor)
    return symmetrize((vectors * (values**power)) @ vectors.T)


def covariance(x: np.ndarray) -> np.ndarray:
    cov = np.cov(x, rowvar=False, ddof=1)
    ridge = 1e-8 * max(float(np.trace(cov) / cov.shape[0]), 1.0)
    return symmetrize(cov + ridge * np.eye(cov.shape[0]))


def bw_operator(
    sigma_x: np.ndarray, sigma_y: np.ndarray, noise_variance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the BW linear operator and the clipped latent covariance."""
    dimension = sigma_x.shape[0]
    omega_raw = symmetrize(sigma_y - noise_variance * np.eye(dimension))
    omega = psd_matrix_power(omega_raw, 1.0, floor=0.0)

    x_half = psd_matrix_power(sigma_x, 0.5, floor=1e-12)
    x_inv_half = psd_matrix_power(sigma_x, -0.5, floor=1e-12)
    middle = psd_matrix_power(x_half @ omega @ x_half, 0.5, floor=0.0)
    operator = symmetrize(x_inv_half @ middle @ x_inv_half)
    return operator, omega


def apply_affine(
    x: np.ndarray, mean_x: np.ndarray, mean_y: np.ndarray, operator: np.ndarray
) -> np.ndarray:
    return (x - mean_x) @ operator.T + mean_y


def make_directions(dimension: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, dimension))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def sliced_wasserstein(
    x: np.ndarray, y: np.ndarray, directions: np.ndarray
) -> float:
    distances = [
        wasserstein_distance(x @ direction, y @ direction)
        for direction in directions
    ]
    return float(np.mean(distances))


def energy_distance_multivariate(x: np.ndarray, y: np.ndarray) -> float:
    # The diagonal zero terms are included, matching the empirical V-statistic.
    return float(
        2.0 * cdist(x, y).mean()
        - cdist(x, x).mean()
        - cdist(y, y).mean()
    )


def median_bandwidth(reference: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    if len(reference) > 800:
        reference = reference[rng.choice(len(reference), 800, replace=False)]
    distances = cdist(reference, reference, metric="sqeuclidean")
    positive = distances[distances > 0]
    median_sq = float(np.median(positive)) if positive.size else 1.0
    return max(median_sq, 1e-12)


def rbf_mmd_squared(x: np.ndarray, y: np.ndarray, bandwidth_sq: float) -> float:
    gamma = 1.0 / (2.0 * bandwidth_sq)
    kxx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
    kyy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean"))
    kxy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))
    return float(max(kxx.mean() + kyy.mean() - 2.0 * kxy.mean(), 0.0))


def distribution_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    directions: np.ndarray,
    bandwidth_sq: float,
) -> dict[str, float]:
    mean_error = np.linalg.norm(prediction.mean(0) - target.mean(0))
    target_mean_scale = max(np.linalg.norm(target.mean(0)), 1e-12)
    pred_cov = np.cov(prediction, rowvar=False, ddof=1)
    target_cov = np.cov(target, rowvar=False, ddof=1)
    covariance_error = np.linalg.norm(pred_cov - target_cov, ord="fro")
    covariance_scale = max(np.linalg.norm(target_cov, ord="fro"), 1e-12)
    return {
        "sliced_wasserstein": sliced_wasserstein(prediction, target, directions),
        "energy_distance": energy_distance_multivariate(prediction, target),
        "mmd_squared": rbf_mmd_squared(prediction, target, bandwidth_sq),
        "relative_mean_error": float(mean_error / target_mean_scale),
        "relative_covariance_error": float(covariance_error / covariance_scale),
    }


def load_cell_type(
    path: Path, requested_cell_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    adata = ad.read_h5ad(path)
    required = {"cell_type", "condition"}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(f"Missing required metadata columns: {sorted(missing)}")

    available = adata.obs["cell_type"].value_counts()
    if requested_cell_type not in available.index:
        raise ValueError(
            f"Unknown cell type {requested_cell_type!r}. Available values:\n{available}"
        )

    keep = np.asarray(adata.obs["cell_type"] == requested_cell_type)
    subset = adata[keep].copy()
    expression = subset.X
    if hasattr(expression, "toarray"):
        expression = expression.toarray()
    expression = np.asarray(expression, dtype=np.float64)

    condition = subset.obs["condition"].astype(str).str.lower().to_numpy()
    control = condition == "ctrl"
    stimulated = condition == "stimulated"
    if control.sum() < 100 or stimulated.sum() < 100:
        raise ValueError("The selected cell type has too few cells in one condition.")

    if "symbol" in subset.var.columns:
        gene_names = subset.var["symbol"].astype(str).to_numpy()
    else:
        gene_names = subset.var_names.astype(str).to_numpy()
    return expression, control, stimulated, gene_names


def save_summary_figure(
    output_path: Path,
    control_test: np.ndarray,
    stimulated_test: np.ndarray,
    mean_shift_test: np.ndarray,
    bw_test: np.ndarray,
    metrics: pd.DataFrame,
    sigma_curve: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    comparisons = [
        (control_test, "Unmapped control vs stimulated"),
        (mean_shift_test, "Mean shift vs stimulated"),
        (bw_test, "Tuned BW map vs stimulated"),
    ]
    colors = ["#2364AA", "#F28E2B", "#59A14F"]
    for ax, (prediction, title), color in zip(axes.flat[:3], comparisons, colors):
        ax.scatter(
            stimulated_test[:, 0], stimulated_test[:, 1], s=10,
            alpha=0.35, color="#777777", label="Stimulated target",
        )
        ax.scatter(
            prediction[:, 0], prediction[:, 1], s=10,
            alpha=0.35, color=color, label=title.split(" vs")[0],
        )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(frameon=False, fontsize=8)

    ax = axes.flat[3]
    ordered = metrics.sort_values("sliced_wasserstein")
    ax.barh(ordered["method"], ordered["sliced_wasserstein"], color="#4E79A7")
    ax.set_xlabel("Held-out sliced Wasserstein distance")
    ax.set_title("Lower is better")

    fig.suptitle("Kang IFN-beta: held-out CD14 monocyte experiment", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        sigma_curve["noise_variance"],
        sigma_curve["validation_sliced_wasserstein"],
        marker="o", markersize=3,
    )
    best = sigma_curve.loc[sigma_curve["selected"]].iloc[0]
    ax.axvline(best["noise_variance"], color="#E15759", linestyle="--")
    ax.set_xlabel(r"Candidate noise variance $\sigma^2$")
    ax.set_ylabel("Validation sliced Wasserstein distance")
    ax.set_title("Noise-variance selection on validation cells")
    fig.tight_layout()
    fig.savefig(output_path.with_name("sigma_validation.png"), dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_if_needed(args.data_path)

    expression, control_mask, stimulated_mask, gene_names = load_cell_type(
        args.data_path, args.cell_type
    )
    control_indices = np.flatnonzero(control_mask)
    stimulated_indices = np.flatnonzero(stimulated_mask)
    c_train, c_validation, c_test = split_three_way(control_indices, args.seed)
    s_train, s_validation, s_test = split_three_way(stimulated_indices, args.seed + 10)

    # Feature selection, centering/scaling, and PCA use only training cells.
    pooled_train_indices = np.concatenate([c_train, s_train])
    train_variances = expression[pooled_train_indices].var(axis=0, ddof=1)
    n_genes = min(args.n_genes, expression.shape[1])
    selected_genes = np.argsort(train_variances)[-n_genes:]
    expression = expression[:, selected_genes]
    selected_gene_names = gene_names[selected_genes]

    scaler = StandardScaler().fit(expression[pooled_train_indices])
    train_scaled = scaler.transform(expression[pooled_train_indices])
    max_components = min(
        args.n_components, train_scaled.shape[0] - 1, train_scaled.shape[1]
    )
    pca = PCA(n_components=max_components, svd_solver="randomized", random_state=args.seed)
    pca.fit(train_scaled)
    scores = pca.transform(scaler.transform(expression))

    xc_train, xc_val, xc_test = scores[c_train], scores[c_validation], scores[c_test]
    ys_train, ys_val, ys_test = scores[s_train], scores[s_validation], scores[s_test]

    mean_x, mean_y = xc_train.mean(0), ys_train.mean(0)
    sigma_x, sigma_y = covariance(xc_train), covariance(ys_train)
    directions = make_directions(max_components, args.n_projections, args.seed + 100)

    # Tune sigma^2 on validation cells only. The upper limit preserves a
    # positive target signal covariance before numerical clipping.
    min_target_eigenvalue = max(float(eigh(sigma_y, eigvals_only=True).min()), 0.0)
    sigma_grid = np.linspace(0.0, 0.95 * min_target_eigenvalue, args.sigma_grid_size)
    sigma_rows: list[dict[str, float | bool]] = []
    for noise_variance in sigma_grid:
        operator, _ = bw_operator(sigma_x, sigma_y, float(noise_variance))
        predicted_validation = apply_affine(xc_val, mean_x, mean_y, operator)
        score = sliced_wasserstein(predicted_validation, ys_val, directions)
        sigma_rows.append(
            {
                "noise_variance": float(noise_variance),
                "validation_sliced_wasserstein": score,
                "selected": False,
            }
        )
    sigma_curve = pd.DataFrame(sigma_rows)
    best_index = int(sigma_curve["validation_sliced_wasserstein"].idxmin())
    sigma_curve.loc[best_index, "selected"] = True
    selected_sigma = float(sigma_curve.loc[best_index, "noise_variance"])

    identity = xc_test
    mean_shift = xc_test - mean_x + mean_y
    x_sd = np.sqrt(np.diag(sigma_x))
    y_sd = np.sqrt(np.diag(sigma_y))
    diagonal_operator = np.diag(y_sd / np.maximum(x_sd, 1e-12))
    diagonal_map = apply_affine(xc_test, mean_x, mean_y, diagonal_operator)
    operator_zero, _ = bw_operator(sigma_x, sigma_y, 0.0)
    bw_zero = apply_affine(xc_test, mean_x, mean_y, operator_zero)
    operator_tuned, omega_tuned = bw_operator(sigma_x, sigma_y, selected_sigma)
    bw_tuned = apply_affine(xc_test, mean_x, mean_y, operator_tuned)

    methods = {
        "unmapped_control": identity,
        "mean_shift": mean_shift,
        "diagonal_location_scale": diagonal_map,
        "bw_sigma_zero": bw_zero,
        "bw_sigma_tuned": bw_tuned,
    }
    bandwidth_sq = median_bandwidth(np.vstack([identity, ys_test]), args.seed + 200)
    metric_rows = []
    for name, prediction in methods.items():
        row = {"method": name}
        row.update(distribution_metrics(prediction, ys_test, directions, bandwidth_sq))
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows).sort_values("sliced_wasserstein")

    # Back-transform the tuned prediction to normalized log-expression space.
    predicted_scaled = pca.inverse_transform(bw_tuned)
    predicted_expression = scaler.inverse_transform(predicted_scaled)
    actual_stimulated_expression = expression[s_test]
    control_test_expression = expression[c_test]
    actual_delta = actual_stimulated_expression.mean(0) - control_test_expression.mean(0)
    predicted_delta = predicted_expression.mean(0) - control_test_expression.mean(0)
    gene_table = pd.DataFrame(
        {
            "gene": selected_gene_names,
            "actual_mean_shift": actual_delta,
            "predicted_mean_shift": predicted_delta,
            "absolute_actual_shift": np.abs(actual_delta),
            "known_interferon_gene": [g.upper() in KNOWN_IFN_GENES for g in selected_gene_names],
        }
    ).sort_values("absolute_actual_shift", ascending=False)

    sigma_curve.to_csv(args.output_dir / "sigma_validation.csv", index=False)
    metrics.to_csv(args.output_dir / "test_metrics.csv", index=False)
    gene_table.head(100).to_csv(args.output_dir / "top_gene_shifts.csv", index=False)
    np.save(args.output_dir / "bw_operator.npy", operator_tuned)
    np.save(args.output_dir / "latent_signal_covariance.npy", omega_tuned)

    summary = {
        "dataset": "Kang IFN-beta PBMC",
        "cell_type": args.cell_type,
        "split_sizes": {
            "control": {"train": len(c_train), "validation": len(c_validation), "test": len(c_test)},
            "stimulated": {"train": len(s_train), "validation": len(s_validation), "test": len(s_test)},
        },
        "selected_genes": n_genes,
        "pca_components": max_components,
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "selected_noise_variance": selected_sigma,
        "best_test_method": str(metrics.iloc[0]["method"]),
        "best_test_sliced_wasserstein": float(metrics.iloc[0]["sliced_wasserstein"]),
        "limitations": [
            "The processed file omits donor identifiers, so this is distribution prediction, not donor-level inference.",
            "The Gaussian isotropic-noise model is applied in standardized PCA space, not to raw RNA counts.",
            "The estimated map is a population-level coupling and is not an observed cell-to-cell causal map.",
        ],
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    save_summary_figure(
        args.output_dir / "experiment_summary.png",
        identity,
        ys_test,
        mean_shift,
        bw_tuned,
        metrics,
        sigma_curve,
    )

    print("\nHeld-out test metrics (lower is better):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"\nSelected noise variance: {selected_sigma:.6f}")
    print(f"Outputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
