"""Paper figures, rendered headless (Agg) into model_artifacts/paper/."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import paths  # noqa: E402
from experiments import data as D  # noqa: E402

FIG_DIR = Path(paths.ARTIFACT_DIR) / "paper" / "figures"

# Colorblind-safe, consistent across all figures.
C_MARGINAL = "#E69F00"
C_MONDRIAN = "#0072B2"
C_ACCENT = "#009E73"
C_GREY = "#999999"


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str((FIG_DIR / name).relative_to(paths.PROJECT_ROOT))


def conditional_coverage(conf: dict) -> str:
    labels = D.PRICE_BRACKET_LABELS
    marg = [conf["coverage_by_bracket"]["marginal_cqr"].get(b, 0) for b in labels]
    mond = [conf["coverage_by_bracket"]["mondrian"].get(b, 0) for b in labels]
    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - w / 2 for i in x], marg, w, label="Marginal CQR", color=C_MARGINAL)
    ax.bar([i + w / 2 for i in x], mond, w, label="Mondrian CQR", color=C_MONDRIAN)
    ax.axhline(conf["target_coverage"], ls="--", color=C_GREY, lw=1,
               label=f"target {conf['target_coverage']:.0%}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("empirical coverage")
    ax.set_xlabel("price bracket")
    ax.set_title("Conditional coverage by price bracket")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    return _save(fig, "conditional_coverage.png")


def reliability(conf: dict) -> str:
    rc = conf["reliability_curve"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0.4, 1.0], [0.4, 1.0], ls="--", color=C_GREY, lw=1, label="ideal")
    ax.plot(rc["nominal"], rc["empirical"], "o-", color=C_MONDRIAN,
            label="calibrated band")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability (marginal calibration)")
    ax.legend(frameon=False)
    return _save(fig, "reliability.png")


def bin_sweep(conf: dict) -> str:
    sweep = conf["bin_count_sweep"]
    nb = [s["n_bins"] for s in sweep]
    ce = [s["cond_err"] for s in sweep]
    wd = [s["mean_width"] for s in sweep]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(nb, ce, "o-", color=C_MONDRIAN, label="conditional-coverage error")
    ax1.set_xlabel("number of Mondrian bins")
    ax1.set_ylabel("conditional-coverage error", color=C_MONDRIAN)
    ax1.tick_params(axis="y", labelcolor=C_MONDRIAN)
    ax2 = ax1.twinx()
    ax2.plot(nb, wd, "s--", color=C_MARGINAL, label="mean band width ($)")
    ax2.set_ylabel("mean band width ($)", color=C_MARGINAL)
    ax2.tick_params(axis="y", labelcolor=C_MARGINAL)
    ax1.set_title("Mondrian: coverage/width tradeoff vs bin count")
    return _save(fig, "bin_sweep.png")


def model_comparison(mc: dict) -> str:
    rows = sorted(mc["rows"], key=lambda r: r["metrics"]["MAE"]["point"])
    names = [r["model"] for r in rows]
    pts = [r["metrics"]["MAE"]["point"] for r in rows]
    lo = [r["metrics"]["MAE"]["point"] - r["metrics"]["MAE"]["lo"] for r in rows]
    hi = [r["metrics"]["MAE"]["hi"] - r["metrics"]["MAE"]["point"] for r in rows]
    colors = [C_ACCENT if "ours" in n else C_GREY for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, pts, xerr=[lo, hi], color=colors, capsize=4)
    ax.set_xlabel("test MAE ($, 95% CI)")
    ax.set_title("Model comparison (identical features & split)")
    ax.invert_yaxis()
    return _save(fig, "model_comparison.png")


def ablation(ab: dict) -> str:
    rows = [r for r in ab["rows"] if "MAE_luxury" in r["metrics"]]
    names = [r["name"].replace("+ ", "+") for r in rows]
    overall = [r["metrics"]["MAE"]["point"] for r in rows]
    lux = [r["metrics"]["MAE_luxury"]["point"] for r in rows]
    x = range(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - w / 2 for i in x], overall, w, label="overall MAE", color=C_MONDRIAN)
    ax.bar([i + w / 2 for i in x], lux, w, label="luxury MAE", color=C_MARGINAL)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("test MAE ($)")
    ax.set_title("Ablation: overall vs luxury MAE as components are added")
    ax.legend(frameon=False)
    return _save(fig, "ablation.png")


def learning_curve(lc: dict) -> str:
    curve = lc["curve"]
    ns = [c["n"] for c in curve]
    maes = [c["mae_mean"] for c in curve]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ns, maes, "o-", color=C_MONDRIAN, label="observed (test MAE)")
    if "a" in lc.get("power_law", {}):
        p = lc["power_law"]
        import numpy as np
        top = max(int(k) for k in lc["extrapolation"]) if lc["extrapolation"] else max(ns)
        xs = np.linspace(min(ns), top, 200)
        ax.plot(xs, p["a"] * xs ** -p["b"] + p["c_floor"], "--", color=C_ACCENT,
                label="power-law fit + extrapolation")
        for N, v in lc["extrapolation"].items():
            ax.scatter([int(N)], [v], color=C_MARGINAL, zorder=5)
            ax.annotate(f"${v:,.0f}", (int(N), v), textcoords="offset points",
                        xytext=(4, 6), fontsize=8, color=C_MARGINAL)
    ax.set_xscale("log")
    ax.set_xlabel("training rows (log scale)")
    ax.set_ylabel("test MAE ($)")
    ax.set_title("Learning curve: MAE vs data size, extrapolated to 175k")
    ax.legend(frameon=False)
    return _save(fig, "learning_curve.png")


def monotonicity(mono: dict) -> str:
    sweeps = list(mono["audits"]["unconstrained"].keys())
    unc = [mono["audits"]["unconstrained"][s]["violation_rate"] for s in sweeps]
    con = [mono["audits"]["constrained"][s]["violation_rate"] for s in sweeps]
    x = range(len(sweeps))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([i - w / 2 for i in x], unc, w, label="unconstrained", color=C_MARGINAL)
    ax.bar([i + w / 2 for i in x], con, w, label="monotone-constrained", color=C_MONDRIAN)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s} sweep" for s in sweeps])
    ax.set_ylabel("fraction of cars with a nonsensical prediction")
    ax.set_title("Monotonicity violations, ceteris paribus")
    ax.legend(frameon=False)
    return _save(fig, "monotonicity.png")


def robustness(rob: dict) -> str:
    rk, gk = rob["repeated_kfold"], rob["grouped_kfold"]
    labels = ["repeated K-fold\n(seen model lines)", "grouped K-fold\n(unseen model lines)"]
    means = [rk["mae_mean"], gk["mae_mean"]]
    err = [[rk["mae_mean"] - rk["mae_ci_lo"]], [gk["mae_std"]]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, means, color=[C_MONDRIAN, C_MARGINAL],
                  yerr=[[rk["mae_mean"] - rk["mae_ci_lo"], gk["mae_std"]],
                        [rk["mae_ci_hi"] - rk["mae_mean"], gk["mae_std"]]], capsize=5)
    ax.set_ylabel("cross-validated MAE ($)")
    ax.set_title("Generalization: seen vs unseen model lines")
    for b, m in zip(bars, means):
        ax.annotate(f"${m:,.0f}", (b.get_x() + b.get_width() / 2, m),
                    ha="center", va="bottom", fontsize=9)
    return _save(fig, "robustness.png")


def render_all(conf, ab, mc, lc=None, mono=None, rob=None) -> dict:
    figs = {
        "conditional_coverage": conditional_coverage(conf),
        "reliability": reliability(conf),
        "bin_sweep": bin_sweep(conf),
        "model_comparison": model_comparison(mc),
        "ablation": ablation(ab),
    }
    if lc is not None:
        figs["learning_curve"] = learning_curve(lc)
    if mono is not None:
        figs["monotonicity"] = monotonicity(mono)
    if rob is not None:
        figs["robustness"] = robustness(rob)
    return figs
