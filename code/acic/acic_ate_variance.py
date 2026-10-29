"""Reproduce a concise ACIC 2016 illustration of ATE randomization variance.

Run: python acic_ate_variance.py --repetitions 10000
Dependencies: numpy pandas scipy matplotlib. Cached data are used when present.
No counterfactual pairing or original treatment assignment enters estimation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

SOURCE = ("https://raw.githubusercontent.com/BiomedSciAI/causallib/master/"
          "causallib/datasets/data/acic_challenge_2016/")
ROOT = Path(__file__).resolve().parent


def kappa4(a):
    z = a - np.mean(a)
    return np.mean(z**4) - 3*np.mean(z*z)**2


def load_data(instance):
    path = ROOT / "data" / f"zymu_{instance}.csv"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(urllib.request.urlopen(SOURCE + path.name, timeout=60).read())
    return pd.read_csv(path), {
        "instance": instance, "url": SOURCE + path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
    }


def screen():
    """Record the exploratory diagnostic screen, including unsuccessful cases."""
    rows = []
    for instance in range(1, 11):
        data, _ = load_data(instance)
        a, b = data.y0.to_numpy(), data.y1.to_numpy()
        va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
        cov = np.cov(a, b, ddof=1)[0, 1]
        ka, kb = kappa4(a), kappa4(b)
        beta = (kb/ka)**.25 if ka != 0 and kb/ka > 0 else np.nan
        rows.append({"instance": instance, "n": len(a), "paired_slope": cov/va,
                     "paired_r2": cov*cov/(va*vb),
                     "oracle_variance_reduction_vs_neyman_pct":
                     100*np.var(b-a, ddof=1)/(2*(va+vb)),
                     "full_population_cumulant_beta": beta,
                     "full_population_cumulant_noise": vb-beta*beta*va})
    pd.DataFrame(rows).to_csv(ROOT/"screening_diagnostics.csv", index=False)


def run(instance, repetitions, seed):
    data, provenance = load_data(instance)
    y0, y1 = data.y0.to_numpy(), data.y1.to_numpy()
    n = len(y0)
    assert n % 2 == 0
    m = n // 2
    s0, s1 = np.var(y0, ddof=1), np.var(y1, ddof=1)
    c01 = np.cov(y0, y1, ddof=1)[0, 1]
    slope = c01 / s0
    residual = y1 - y1.mean() - slope*(y0-y0.mean())
    oracle = s0/m + s1/m - np.var(y1-y0, ddof=1)/n
    truth = np.mean(y1-y0)
    rng = np.random.default_rng(seed)
    records = []
    zcrit = norm.ppf(.975)
    for b in range(repetitions):
        order = rng.permutation(n)
        control, treated = y0[order[:m]], y1[order[m:]]
        v0, v1 = np.var(control, ddof=1), np.var(treated, ddof=1)
        ate = treated.mean()-control.mean()
        neyman = v0/m + v1/m
        k0, k1 = kappa4(control), kappa4(treated)
        beta = (k1/k0)**.25 if k0 != 0 and k1/k0 > 0 else np.nan
        noise = v1-beta**2*v0
        tau_var = (beta-1)**2*v0+noise
        vbw = neyman-tau_var/n
        valid = bool(np.isfinite(beta) and noise >= 0 and vbw > 0)
        # Prespecified, observable fallback. Do not discard failed estimates.
        deployed = vbw if valid else neyman
        # Aronow--Green--Lee upper-bound plug-in, n=N, equal arm sizes.
        # The empirical CDF moments use denominator m, and population n
        # is converted to finite-population variance with denominator n-1.
        a = np.sort(control)-control.mean()
        d = np.sort(treated)-treated.mean()
        agl = (np.mean(a*a)+np.mean(d*d)+2*np.mean(a*d))/(n-1)
        records.append((ate, neyman, agl, deployed, vbw, valid, beta, noise))
    draws = pd.DataFrame(records, columns=["ate", "Neyman", "AGL upper bound",
                         "BW with fallback", "bw_raw", "admissible", "beta", "noise"])
    draws.to_csv(ROOT / f"draws_instance_{instance}.csv.gz", index=False)
    summaries = []
    for label in ["Neyman", "AGL upper bound", "BW with fallback", "Oracle"]:
        values = np.full(repetitions, oracle) if label == "Oracle" else draws[label].to_numpy()
        cover = np.abs(draws.ate.to_numpy()-truth) <= zcrit*np.sqrt(values)
        summaries.append({
            "instance": instance, "method": label,
            "mean_variance": values.mean(), "oracle_variance": oracle,
            "relative_bias_pct": 100*(values.mean()/oracle-1),
            "variance_rmse": np.sqrt(np.mean((values-oracle)**2)),
            "coverage_pct": 100*cover.mean(),
            "coverage_mcse_pct": 100*np.sqrt(cover.mean()*(1-cover.mean())/repetitions),
            "mean_interval_length": np.mean(2*zcrit*np.sqrt(values)),
            "length_reduction_pct": 100*(1-np.mean(np.sqrt(values))/np.mean(np.sqrt(draws.Neyman))),
        })
    diagnostics = {
        "instance": instance, "n": n, "n1": m, "repetitions": repetitions,
        "seed": seed, "paired_r2": c01*c01/(s0*s1), "paired_slope": slope,
        "oracle_ate": truth, "oracle_treatment_effect_variance": np.var(y1-y0, ddof=1),
        "oracle_ate_variance": oracle,
        "empirical_randomization_variance": draws.ate.var(ddof=1),
        "admissible_pct": 100*draws.admissible.mean(),
        "mean_admissible_beta": draws.loc[draws.admissible, "beta"].mean(),
        "mean_admissible_noise": draws.loc[draws.admissible, "noise"].mean(),
        "paired_residual_spearman_y0": float(spearmanr(residual, y0).statistic),
        "paired_residual_spearman_y0_squared": float(spearmanr(residual**2, y0).statistic),
        "provenance": provenance,
    }
    print(json.dumps(diagnostics, indent=2), flush=True)
    return draws, summaries, diagnostics


def figure(draws, summaries, diagnostics):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    data, _ = load_data(1)
    ax = axes[0]
    ax.scatter(data.y0, data.y1, s=5, alpha=.18, rasterized=True, color="#31688e")
    x = np.linspace(data.y0.min(), data.y0.max(), 100)
    d = diagnostics[0]
    ax.plot(x, data.y1.mean()+d["paired_slope"]*(x-data.y0.mean()), color="#cf5c36", lw=2)
    ax.set(xlabel=r"Control potential outcome $Y(0)$", ylabel=r"Treatment potential outcome $Y(1)$",
           title=f"ACIC instance 1: paired diagnostic\n$R^2={d['paired_r2']:.3f}$; slope = {d['paired_slope']:.3f}")
    ax = axes[1]
    methods = ["Neyman", "AGL upper bound", "BW with fallback"]
    colors = ["#9ca3af", "#31688e", "#cf5c36"]
    oracle = d["oracle_ate_variance"]
    for k, (method, color) in enumerate(zip(methods, colors)):
        values = draws[method].to_numpy()/oracle
        lo, median, hi = np.quantile(values, [.025,.5,.975])
        ax.errorbar(k, median, yerr=[[median-lo],[hi-median]], fmt="o", capsize=5, color=color)
    ax.axhline(1, ls="--", color="black", lw=1)
    ax.set(xticks=range(3), xticklabels=["Neyman", "AGL upper", "BW*"],
           ylabel="Estimated variance / oracle variance", title="Instance 1: randomization variability")
    ax.text(.03, .97, "Points: median; bars: central 95%", transform=ax.transAxes, va="top", fontsize=8)
    ax = axes[2]
    table = pd.DataFrame(summaries)
    for i, method in enumerate(methods):
        s = table[table.method == method]
        ax.bar(np.arange(2)+(i-1)*.24, s.length_reduction_pct, width=.24, color=colors[i],
               label={"AGL upper bound":"AGL upper", "BW with fallback":"BW*"}.get(method, method))
    ax.set(xticks=[0,1], xticklabels=["Instance 1\n$R^2=0.399$", "Instance 2\n$R^2=0.922$"],
           ylabel="Mean interval reduction vs. Neyman (%)", title="Interval reduction and BW coverage")
    for j, case in enumerate([1, 2]):
        row = table[(table.instance == case) & (table.method == "BW with fallback")].iloc[0]
        ax.text(j+.24, row.length_reduction_pct+.15, f"{row.coverage_pct:.1f}%\ncoverage",
                ha="center", va="bottom", fontsize=8, color="#9b3518")
    ax.set_ylim(0, 10.5)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(ROOT/"acic_ate_variance.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    screen()
    rows, diagnostic_rows, first_draws = [], [], None
    for instance in [1, 2]:
        draws, summary, diagnostic = run(instance, args.repetitions, args.seed+instance)
        rows.extend(summary)
        diagnostic_rows.append(diagnostic)
        if instance == 1:
            first_draws = draws
    pd.DataFrame(rows).to_csv(ROOT/"acic_variance_results.csv", index=False)
    (ROOT/"acic_diagnostics.json").write_text(json.dumps(diagnostic_rows, indent=2))
    figure(first_draws, rows, diagnostic_rows)
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
