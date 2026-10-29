#!/usr/bin/env python3
"""Reproduce the SPD unlabeled-sensing application using NumPy/SciPy.

Run: OPENBLAS_NUM_THREADS=1 python unlabeled_sensing.py --reps 200
The default runs both sweeps, writes trial-level and summary CSVs, and makes
the manuscript figures and LaTeX table. No dataset download is required.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

METHODS = ("Direct matching", "Known-A matching", "BW matching")
COLORS = {METHODS[0]: "#777777", METHODS[1]: "#D48321", METHODS[2]: "#1565A6"}


def symmetric(a):
    return (a + a.T) / 2


def spd_power(a, power):
    w, u = np.linalg.eigh(symmetric(a))
    if w.min() <= 0:
        raise ValueError("A strictly positive-definite matrix is required.")
    return symmetric((u * w**power) @ u.T)


def sample_covariance(x):
    xc = x - x.mean(axis=0)
    return xc.T @ xc / (len(x) - 1)


def bw_estimate(x, y, noise_covariance, floor_relative=1e-6):
    """Permutation-invariant coefficient estimate with explicit spectral floor.

    All covariances have denominator n-1. The raw noise-corrected covariance
    may be indefinite. The implemented extension replaces eigenvalues below
    tau = floor_relative * tr(S_Y)/d by tau and reports every such event.
    The raw-matrix positivity diagnostic is recorded separately.
    """
    if len(x) != len(y) or x.shape[1] != y.shape[1]:
        raise ValueError("This application requires equal sample sizes/dimensions.")
    sx, sy = sample_covariance(x), sample_covariance(y)
    omega_raw = symmetric(sy - noise_covariance)
    w, u = np.linalg.eigh(omega_raw)
    tau = floor_relative * float(np.trace(sy)) / sy.shape[0]
    if tau <= 0:
        raise ValueError("Nonpositive response covariance scale.")
    omega = symmetric((u * np.maximum(w, tau)) @ u.T)
    sx_half, sx_inverse_half = spd_power(sx, .5), spd_power(sx, -.5)
    a_hat = symmetric(sx_inverse_half @ spd_power(sx_half @ omega @ sx_half, .5)
                      @ sx_inverse_half)
    centers = y.mean(axis=0) + (x - x.mean(axis=0)) @ a_hat.T
    return a_hat, centers, {
        "raw_positive_definite": int(w.min() > 0),
        "regularized": int(np.any(w < tau)),
        "eigenvalues_floored": int(np.sum(w < tau)),
        "raw_min_eigenvalue": float(w.min()),
        "eigenvalue_floor": tau,
    }


def assignment(centers, responses):
    """Output p with response i matched to predictor p[i]."""
    cost = cdist(responses, centers, metric="sqeuclidean")
    rows, cols = linear_sum_assignment(cost)
    p = np.empty(len(responses), dtype=int)
    p[rows] = cols
    return p


def coefficient_matrix(d, condition_number, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, 7401]))
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eigenvalues = np.geomspace(1., condition_number, d)
    eigenvalues /= np.sqrt(np.mean(eigenvalues**2))
    return symmetric((q * eigenvalues) @ q.T)


def run_case(n, d, kappa, sigma, reps, seed, floor_relative):
    a = coefficient_matrix(d, kappa, seed)
    b = .25 * np.cos(np.arange(d))
    gamma = sigma**2 * np.eye(d)
    records = []
    for rep in range(reps):
        # Common X, standardized errors, and permutation across kappa/sigma
        # at a given (n,d,rep) make method comparisons paired.
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, d, rep]))
        x = rng.normal(size=(n, d))
        truth = rng.permutation(n)
        eps = sigma * rng.normal(size=(n, d))
        y = x[truth] @ a.T + b + eps
        a_hat, estimated_centers, diag = bw_estimate(x, y, gamma, floor_relative)
        centers = (x, x @ a.T + b, estimated_centers)
        coefficient_error = np.linalg.norm(a_hat - a, 2) / np.linalg.norm(a, 2)
        for method, z in zip(METHODS, centers):
            p = assignment(z, y)
            accuracy = float(np.mean(p == truth))
            records.append(dict(n=n, d=d, kappa=kappa, sigma=sigma, rep=rep,
                                method=method, accuracy=accuracy,
                                hamming_error=1-accuracy,
                                exact_recovery=int(np.all(p == truth)),
                                relative_operator_error=float(coefficient_error),
                                **diag))
    return records


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    den = 1 + z*z/total
    mid = (p + z*z/(2*total)) / den
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / den
    return max(0., mid-half), min(1., mid+half)


def summarize(records):
    groups = {}
    for r in records:
        key = tuple(r[x] for x in ("n", "d", "kappa", "sigma", "method"))
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in sorted(groups.items()):
        acc = np.array([r["accuracy"] for r in rs])
        successes = sum(r["exact_recovery"] for r in rs)
        low, high = wilson(successes, len(rs))
        out.append(dict(zip(("n", "d", "kappa", "sigma", "method"), key)) |
                   dict(reps=len(rs), mean_accuracy=float(acc.mean()),
                        accuracy_mcse=float(acc.std(ddof=1)/np.sqrt(len(rs))),
                        mean_hamming_error=float(1-acc.mean()),
                        exact_recovery_probability=successes/len(rs),
                        exact_wilson_low=low, exact_wilson_high=high,
                        regularization_fraction=np.mean([r["regularized"] for r in rs]),
                        raw_pd_fraction=np.mean([r["raw_positive_definite"] for r in rs]),
                        mean_relative_operator_error=np.mean(
                            [r["relative_operator_error"] for r in rs])))
    return out


def paired_summary(records):
    groups = {}
    for r in records:
        key = tuple(r[x] for x in ("n", "d", "kappa", "sigma"))
        groups.setdefault(key, {}).setdefault(r["rep"], {})[r["method"]] = r
    out = []
    for key, reps in sorted(groups.items()):
        ds = np.array([rs[METHODS[2]]["accuracy"]-rs[METHODS[0]]["accuracy"]
                       for rs in reps.values()])
        de = np.array([rs[METHODS[2]]["exact_recovery"]-rs[METHODS[0]]["exact_recovery"]
                       for rs in reps.values()])
        out.append(dict(zip(("n", "d", "kappa", "sigma"), key)) |
                   dict(reps=len(ds), bw_minus_direct_accuracy=float(ds.mean()),
                        accuracy_difference_mcse=float(ds.std(ddof=1)/np.sqrt(len(ds))),
                        bw_minus_direct_exact=float(de.mean()),
                        exact_difference_mcse=float(de.std(ddof=1)/np.sqrt(len(de)))))
    return out


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .17,
                         "figure.dpi": 130, "savefig.dpi": 240})


def plot_noise(summary, output):
    setup_style()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey="row")
    for col, k in enumerate((1., 3., 10.)):
        for method in METHODS:
            rs = sorted((r for r in summary if r["n"] == 300 and r["d"] == 12
                         and r["kappa"] == k and r["method"] == method),
                        key=lambda r: r["sigma"])
            if not rs:
                continue
            x = np.array([r["sigma"] for r in rs])
            y = np.array([r["mean_accuracy"] for r in rs])
            se = np.array([r["accuracy_mcse"] for r in rs])
            axes[0,col].plot(x, y, "o-", color=COLORS[method], label=method, ms=4)
            axes[0,col].fill_between(x, np.maximum(0,y-1.96*se),
                                    np.minimum(1,y+1.96*se), color=COLORS[method], alpha=.12)
            y = np.array([r["exact_recovery_probability"] for r in rs])
            lo = np.array([r["exact_wilson_low"] for r in rs])
            hi = np.array([r["exact_wilson_high"] for r in rs])
            axes[1,col].plot(x, y, "o-", color=COLORS[method], ms=4)
            axes[1,col].fill_between(x, lo, hi, color=COLORS[method], alpha=.10)
        axes[0,col].set_title(r"$\kappa(A)=%g$" % k)
        axes[1,col].set_xlabel(r"Noise standard deviation $\sigma$")
    axes[0,0].set_ylabel("Fraction correctly matched")
    axes[1,0].set_ylabel("Exact recovery probability")
    for ax in axes.flat:
        ax.set_ylim(-.025,1.025)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5,1.015))
    fig.tight_layout(rect=(0,0,1,.96))
    for extension in ("png", "pdf"):
        fig.savefig(output / ("unlabeled_sensing_comparison." + extension), bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(summary, output):
    setup_style()
    fig, axes = plt.subplots(1,2,figsize=(8.5,3.2))
    for k, color in zip((1.,3.,10.), ("#777777", "#579C6E", "#1565A6")):
        rs = sorted((r for r in summary if r["n"]==300 and r["d"]==12
                     and r["kappa"]==k and r["method"]==METHODS[2]), key=lambda r:r["sigma"])
        if not rs:
            continue
        x = [r["sigma"] for r in rs]
        axes[0].plot(x,[r["regularization_fraction"] for r in rs],"o-",color=color,
                     label=r"$\kappa(A)=%g$" % k)
        axes[1].plot(x,[r["mean_relative_operator_error"] for r in rs],"o-",color=color)
    axes[0].set_ylabel("Eigenvalue flooring frequency")
    axes[0].set_ylim(-.025,1.025)
    axes[1].set_ylabel("Mean relative operator error")
    for ax in axes:
        ax.set_xlabel(r"Noise standard deviation $\sigma$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(output/("unlabeled_sensing_diagnostics."+ext),bbox_inches="tight")
    plt.close(fig)


def plot_dimension(summary, output):
    setup_style()
    fig, axes = plt.subplots(1,2,figsize=(8.5,3.2),sharey=True)
    for ax,sigma in zip(axes,(.3,.6)):
        for method in METHODS:
            rs=sorted((r for r in summary if r["n"]==300 and r["kappa"]==10
                       and r["sigma"]==sigma and r["method"]==method),key=lambda r:r["d"])
            if not rs:
                continue
            ax.plot([r["d"] for r in rs],[r["mean_accuracy"] for r in rs],"o-",
                    color=COLORS[method],label=method)
        ax.set_title(r"$\sigma=%g$"%sigma)
        ax.set_xlabel("Dimension d")
        ax.set_ylim(-.025,1.025)
    axes[0].set_ylabel("Fraction correctly matched")
    axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(output/("unlabeled_sensing_dimension."+ext),bbox_inches="tight")
    plt.close(fig)


def write_latex_table(summary, output):
    selected=((1.,.6),(3.,.6),(10.,.3),(10.,.6))
    lookup={(r["kappa"],r["sigma"],r["method"]):r for r in summary
            if r["n"]==300 and r["d"]==12}
    lines=[r"\begin{tabular}{cc@{\hspace{1.4em}}rrrr}", r"\hline",
           r"$\kappa(A)$ & $\sigma$ & Direct & Known $A$ & BW & Floored \\",r"\hline"]
    for k,s in selected:
        if (k,s,METHODS[2]) not in lookup:
            continue
        rs=[lookup[k,s,m] for m in METHODS]
        lines.append(f"{k:g} & {s:g} & "+" & ".join(f"{100*r['mean_accuracy']:.2f}" for r in rs)
                     +f" & {100*rs[2]['regularization_fraction']:.1f}"+r" \\")
    lines += [r"\hline",r"\end{tabular}"]
    (output/"unlabeled_sensing_results_table.tex").write_text("\n".join(lines)+"\n")


def validate():
    """Check mathematical invariants and assignment indexing."""
    rng=np.random.default_rng(24)
    x=rng.normal(size=(70,5))
    a=coefficient_matrix(5,8,17)
    p=rng.permutation(len(x))
    b=np.arange(5)*.1
    y=x[p]@a.T+b
    ah,z,diag=bw_estimate(x,y,np.zeros((5,5)))
    assert diag["regularized"]==0
    assert np.allclose(ah,a,atol=1e-10)
    assert np.array_equal(assignment(z,y),p)
    ah2,_,_=bw_estimate(x[rng.permutation(len(x))],y,np.zeros((5,5)))
    assert np.allclose(ah2,ah,atol=1e-10)
    y2=y+.4*rng.normal(size=y.shape)
    assert np.array_equal(assignment(x,y2),assignment(3*x+2,y2))
    sx=sample_covariance(x)
    assert np.allclose(ah@sx@ah,sample_covariance(y),atol=1e-10)
    print("Validation passed: exact noiseless A/permutation recovery, permutation "
          "invariance, scalar-map equivalence, and covariance equation.",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps",type=int,default=200)
    parser.add_argument("--seed",type=int,default=20261020)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--sweep",choices=("all","noise","dimension"),default="all")
    parser.add_argument("--floor-relative",type=float,default=1e-6)
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args()
    validate()
    if args.validate_only:
        return
    if args.reps<2 or args.floor_relative<=0:
        parser.error("--reps must be >=2; --floor-relative must be >0")
    args.output.mkdir(parents=True,exist_ok=True)
    cases=[]
    if args.sweep in ("all","noise"):
        cases += [(300,12,k,s) for k in (1.,3.,10.) for s in (.15,.3,.45,.6,.75,.9)]
    if args.sweep in ("all","dimension"):
        cases += [(300,d,10.,s) for d in (6,12,24,36) for s in (.3,.6)]
    cases=list(dict.fromkeys(cases))
    records=[]
    for case in cases:
        records += run_case(*case,args.reps,args.seed,args.floor_relative)
        write_csv(args.output/"unlabeled_sensing_trials.csv",records)
        print(f"Completed n={case[0]}, d={case[1]}, kappa={case[2]}, "
              f"sigma={case[3]} ({args.reps} replications)",flush=True)
    summary=summarize(records)
    paired=paired_summary(records)
    write_csv(args.output/"unlabeled_sensing_summary.csv",summary)
    write_csv(args.output/"unlabeled_sensing_paired_comparisons.csv",paired)
    if args.sweep in ("all","noise"):
        plot_noise(summary,args.output)
        plot_diagnostics(summary,args.output)
        write_latex_table(summary,args.output)
    if args.sweep in ("all","dimension"):
        plot_dimension(summary,args.output)
    config=dict(reps=args.reps,seed=args.seed,floor_relative=args.floor_relative,
                cases=cases,numpy=np.__version__)
    (args.output/"unlabeled_sensing_config.json").write_text(json.dumps(config,indent=2)+"\n")
    print("All outputs written to",args.output,flush=True)


if __name__=="__main__":
    main()
