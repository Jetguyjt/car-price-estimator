"""
Conformal coverage study: marginal CQR vs predicted-price-group CQR.

This study measures whether separate calibration adjustments reduce empirical
coverage imbalance across price ranges while preserving useful overall coverage
and band width. The results describe this evaluation sample; they do not establish
a coverage guarantee for every segment.
"""

import json
from pathlib import Path

import numpy as np

import conformal
import conformal_mondrian as cm
import paths
from experiments import data as D
from experiments import scoring
from experiments.models import fit_bundle

TARGET = 0.80
N_BINS = 10                     # operating point (see bin-count sweep in results)
BIN_SWEEP = [5, 8, 10, 12, 15]
OUT_DIR = Path(paths.ARTIFACT_DIR) / "paper"


def _coverage(actual, lo, hi):
    return float(np.mean((actual >= lo) & (actual <= hi)))


def run() -> dict:
    d = D.load()
    bundle = fit_bundle(d)
    te, cal = bundle["test"], bundle["cal"]
    actual = d.y_test.to_numpy()

    # --- marginal CQR (production method) ---
    g = conformal.compute_log_delta(cal["q10_log"], cal["q90_log"], d.y_cal_log, TARGET)
    marg_lo = np.expm1(te["q10_log"] - g)
    marg_hi = np.expm1(te["q90_log"] + g)

    # --- Mondrian (bins keyed on predicted q50) ---
    mond = cm.fit(cal["q10_log"], cal["q90_log"], cal["q50_log"], d.y_cal_log,
                  target_coverage=TARGET, n_bins=N_BINS)
    lo_d, hi_d = cm.deltas_for(mond, te["q50_log"])
    mond_lo = np.expm1(te["q10_log"] - lo_d)
    mond_hi = np.expm1(te["q90_log"] + hi_d)

    # --- raw (uncalibrated) band for reference ---
    raw_lo = np.expm1(te["q10_log"])
    raw_hi = np.expm1(te["q90_log"])

    methods = {
        "raw": (raw_lo, raw_hi),
        "marginal_cqr": (marg_lo, marg_hi),
        "mondrian": (mond_lo, mond_hi),
    }

    mid = np.expm1(te["q50_log"])
    brackets = d.test_meta["price_bracket"]
    per_bracket, overall, widths, cond_err, scores = {}, {}, {}, {}, {}
    for name, (lo, hi) in methods.items():
        overall[name] = _coverage(actual, lo, hi)
        widths[name] = float(np.mean(hi - lo))
        # Proper scoring rules: Winkler interval score + pinball, overall and per bracket.
        scores[name] = {"overall": scoring.score_band(actual, lo, hi, mid, alpha=1 - TARGET)}
        covs = {}
        for b in D.PRICE_BRACKET_LABELS:
            mask = (brackets == b).to_numpy()
            if mask.sum() == 0:
                continue
            covs[b] = _coverage(actual[mask], lo[mask], hi[mask])
            scores[name][b] = scoring.winkler_score(actual[mask], lo[mask], hi[mask],
                                                    alpha=1 - TARGET)
        per_bracket[name] = covs
        cond_err[name] = float(np.mean([abs(c - TARGET) for c in covs.values()]))

    reliability = _reliability_curve(d, bundle)
    bin_sweep = _bin_sweep(d, bundle, brackets, actual)

    results = {
        "target_coverage": TARGET,
        "n_bins": N_BINS,
        "overall_coverage": overall,
        "mean_band_width": widths,
        "coverage_by_bracket": per_bracket,
        "conditional_coverage_error": cond_err,
        "proper_scores": scores,
        "mondrian_model": mond,
        "reliability_curve": reliability,
        "bin_count_sweep": bin_sweep,
        "n_test": int(len(actual)),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "conformal_study.json").write_text(json.dumps(results, indent=2))
    _print(results)
    return results


def _bin_sweep(d, bundle, brackets, actual) -> dict:
    """Conditional-coverage error and band width vs bin count - evidence for the
    operating-point choice and the coverage/width tradeoff."""
    te, cal = bundle["test"], bundle["cal"]
    rows = []
    for nb in BIN_SWEEP:
        m = cm.fit(cal["q10_log"], cal["q90_log"], cal["q50_log"], d.y_cal_log,
                   target_coverage=TARGET, n_bins=nb)
        lo_d, hi_d = cm.deltas_for(m, te["q50_log"])
        lo = np.expm1(te["q10_log"] - lo_d)
        hi = np.expm1(te["q90_log"] + hi_d)
        covs = [_coverage(actual[(brackets == b).to_numpy()],
                          lo[(brackets == b).to_numpy()],
                          hi[(brackets == b).to_numpy()])
                for b in D.PRICE_BRACKET_LABELS]
        rows.append({"n_bins": nb, "overall": _coverage(actual, lo, hi),
                     "cond_err": float(np.mean([abs(c - TARGET) for c in covs])),
                     "top_bracket_cov": covs[-1],
                     "mean_width": float(np.mean(hi - lo))})
    return rows


def _reliability_curve(d, bundle) -> dict:
    """Empirical coverage of the two-sided central band at several nominal levels,
    each independently marginal-calibrated - the standard calibration diagnostic."""
    te, cal = bundle["test"], bundle["cal"]
    actual = d.y_test.to_numpy()
    nominal = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    emp = []
    for lvl in nominal:
        g = conformal.compute_log_delta(cal["q10_log"], cal["q90_log"], d.y_cal_log, lvl)
        lo = np.expm1(te["q10_log"] - g)
        hi = np.expm1(te["q90_log"] + g)
        emp.append(_coverage(actual, lo, hi))
    return {"nominal": nominal, "empirical": emp}


def _print(r: dict) -> None:
    print(f"\n=== Conformal coverage (target {r['target_coverage']:.0%}, "
          f"n_test={r['n_test']}) ===")
    print(f"{'method':<14}{'overall':>9}{'cond.err':>10}{'width':>12}")
    for m in ("raw", "marginal_cqr", "mondrian"):
        print(f"{m:<14}{r['overall_coverage'][m]:>8.1%}"
              f"{r['conditional_coverage_error'][m]:>10.3f}"
              f"${r['mean_band_width'][m]:>10,.0f}")
    print("\ncoverage by price bracket:")
    print(f"{'bracket':<10}" + "".join(f"{m:>14}" for m in
          ("marginal_cqr", "mondrian")))
    for b in D.PRICE_BRACKET_LABELS:
        row = f"{b:<10}"
        for m in ("marginal_cqr", "mondrian"):
            c = r["coverage_by_bracket"][m].get(b)
            row += f"{c:>13.1%} " if c is not None else f"{'-':>14}"
        print(row)
    print("\nproper scoring rules (lower is better):")
    print(f"{'method':<14}{'Winkler':>12}{'pinball q50':>13}")
    for m in ("raw", "marginal_cqr", "mondrian"):
        s = r["proper_scores"][m]["overall"]
        print(f"{m:<14}${s['winkler']:>10,.0f}${s.get('pinball_q50', 0):>11,.0f}")


if __name__ == "__main__":
    run()
