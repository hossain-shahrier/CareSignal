"""Run multi-case prediction validation against trained model and holdout data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from caresignal.schemas import FEATURE_COLUMNS  # noqa: E402

def load_threshold(artifacts: Path) -> float:
    manifest_path = artifacts / "manifest.json"
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return float(data.get("metrics", {}).get("threshold", 0.29))
    return 0.29

# UI presets + user screenshot case + clinical edge cases
TEST_CASES: list[tuple[str, dict]] = [
    ("ui_preset_low", {
        "age": 34,
        "sex": "female",
        "length_of_stay_days": 3,
        "prior_admissions": 0,
        "condition_count": 1,
        "procedure_count": 0,
        "days_since_last_discharge": 0,
    }),
    ("ui_preset_high", {
        "age": 72,
        "sex": "male",
        "length_of_stay_days": 9,
        "prior_admissions": 2,
        "condition_count": 6,
        "procedure_count": 2,
        "days_since_last_discharge": 12,
    }),
    ("user_screenshot", {
        "age": 56,
        "sex": "male",
        "length_of_stay_days": 7.2,
        "prior_admissions": 2,
        "condition_count": 9,
        "procedure_count": 13,
        "days_since_last_discharge": 389,
    }),
    ("first_admission_young", {
        "age": 25,
        "sex": "female",
        "length_of_stay_days": 2,
        "prior_admissions": 0,
        "condition_count": 1,
        "procedure_count": 0,
        "days_since_last_discharge": 0,
    }),
    ("frequent_flyer_recent", {
        "age": 68,
        "sex": "male",
        "length_of_stay_days": 10,
        "prior_admissions": 5,
        "condition_count": 12,
        "procedure_count": 8,
        "days_since_last_discharge": 5,
    }),
    ("complex_but_long_gap", {
        "age": 56,
        "sex": "male",
        "length_of_stay_days": 7.2,
        "prior_admissions": 2,
        "condition_count": 9,
        "procedure_count": 13,
        "days_since_last_discharge": 12,
    }),
    ("extreme_procedures_only", {
        "age": 56,
        "sex": "male",
        "length_of_stay_days": 7.2,
        "prior_admissions": 0,
        "condition_count": 1,
        "procedure_count": 20,
        "days_since_last_discharge": 0,
    }),
    ("elderly_long_stay_no_prior", {
        "age": 85,
        "sex": "female",
        "length_of_stay_days": 21,
        "prior_admissions": 0,
        "condition_count": 8,
        "procedure_count": 4,
        "days_since_last_discharge": 0,
    }),
]


def load_model(path: Path):
    return joblib.load(path)


def score(model, row: dict) -> float:
    frame = pd.DataFrame([row])
    return float(model.predict_proba(frame[FEATURE_COLUMNS])[0, 1])


def base_pipeline(model):
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return model


def feature_importances(model) -> list[tuple[str, float]]:
    pipe = base_pipeline(model)
    clf = pipe.named_steps["classifier"]
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    imp = clf.feature_importances_
    return sorted(zip(names, imp), key=lambda x: -x[1])


def holdout_analysis(model, holdout_path: Path, threshold: float) -> dict:
    df = pd.read_parquet(holdout_path)
    x = df[FEATURE_COLUMNS]
    y = df["readmitted_30d"].to_numpy()
    probas = model.predict_proba(x)[:, 1]
    preds = (probas >= threshold).astype(int)

    # Calibration bins
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(probas, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 9)

    calibration = []
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        calibration.append({
            "bin_lo": float(bins[b]),
            "bin_hi": float(bins[b + 1]),
            "n": int(mask.sum()),
            "mean_pred": float(probas[mask].mean()),
            "actual_rate": float(y[mask].mean()),
        })

    return {
        "n": len(df),
        "positive_rate": float(y.mean()),
        "mean_pred": float(probas.mean()),
        "roc_auc_proxy": float(
            np.corrcoef(probas, y)[0, 1] if len(np.unique(y)) > 1 else 0
        ),
        "pred_elevated_rate": float(preds.mean()),
        "calibration_bins": calibration,
        "min_prob": float(probas.min()),
        "max_prob": float(probas.max()),
    }


def monotonicity_checks(model, base_row: dict) -> list[dict]:
    """One-at-a-time sweeps; flag if risk moves opposite to expected direction."""
    checks = []

    sweeps = [
        ("age", [40, 56, 72, 85], "increasing"),
        ("prior_admissions", [0, 1, 2, 5], "increasing"),
        ("condition_count", [1, 4, 9, 15], "increasing"),
        ("procedure_count", [0, 2, 8, 15], "increasing"),
        ("length_of_stay_days", [2, 5, 10, 20], "increasing"),
        ("days_since_last_discharge", [5, 30, 120, 389], "decreasing"),
    ]

    for field, values, expected in sweeps:
        scores = [score(model, {**base_row, field: v}) for v in values]
        if expected == "increasing":
            violations = sum(
                1 for i in range(len(scores) - 1) if scores[i + 1] < scores[i] - 0.02
            )
        else:
            violations = sum(
                1 for i in range(len(scores) - 1) if scores[i + 1] > scores[i] + 0.02
            )
        checks.append({
            "field": field,
            "values": values,
            "scores": [round(s, 4) for s in scores],
            "expected_trend": expected,
            "step_violations": violations,
        })
    return checks


def main() -> int:
    artifacts = PROJECT_ROOT / "artifacts"
    model_path = artifacts / "model.joblib"
    holdout_path = artifacts / "holdout.parquet"
    features_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"

    if not model_path.is_file():
        print(f"Missing {model_path}")
        return 1

    model = load_model(model_path)
    threshold = load_threshold(artifacts)
    cohort = pd.read_parquet(features_path) if features_path.is_file() else None

    report: dict = {
        "threshold": threshold,
        "cohort_positive_rate": float(cohort["readmitted_30d"].mean()) if cohort is not None else None,
        "feature_importances": [
            {"feature": n, "importance": float(v)} for n, v in feature_importances(model)
        ],
        "test_cases": [],
        "monotonicity_from_user_base": None,
        "holdout": None,
        "cohort_percentiles_user_case": None,
        "issues": [],
    }

    print("=" * 72)
    print("CareSignal prediction validation")
    print("=" * 72)
    print(
        f"Threshold: {threshold:.0%}  |  Cohort readmit rate: {report['cohort_positive_rate']:.1%}"
        if report["cohort_positive_rate"]
        else f"Threshold: {threshold:.0%}"
    )

    print("\n--- Feature importances (split) ---")
    for name, val in feature_importances(model)[:10]:
        print(f"  {name:42s} {val:.0f}")

    print("\n--- Test cases ---")
    for label, row in TEST_CASES:
        p = score(model, row)
        elevated = p >= threshold
        report["test_cases"].append({
            "label": label,
            "features": row,
            "risk_score": round(p, 4),
            "predicted_readmission": elevated,
        })
        flag = "ELEVATED" if elevated else "lower"
        print(f"  {label:28s}  {p*100:5.1f}%  [{flag}]")

    # Preset ordering sanity
    low_p = next(c["risk_score"] for c in report["test_cases"] if c["label"] == "ui_preset_low")
    high_p = next(c["risk_score"] for c in report["test_cases"] if c["label"] == "ui_preset_high")
    user_p = next(c["risk_score"] for c in report["test_cases"] if c["label"] == "user_screenshot")
    gap_p = next(
        c["risk_score"] for c in report["test_cases"] if c["label"] == "complex_but_long_gap"
    )

    if not (low_p < high_p):
        report["issues"].append(
            f"UI presets inverted: low={low_p:.3f} high={high_p:.3f} (low should be < high)"
        )
    if gap_p <= user_p + 0.05:
        report["issues"].append(
            f"days_since_last_discharge may be weak: same patient 389d={user_p:.3f} vs 12d={gap_p:.3f}"
        )

    user_row = next(row for lbl, row in TEST_CASES if lbl == "user_screenshot")
    report["monotonicity_from_user_base"] = monotonicity_checks(model, user_row)

    print("\n--- Monotonicity (user base case, one feature at a time) ---")
    for chk in report["monotonicity_from_user_base"]:
        trend = "OK" if chk["step_violations"] == 0 else f"{chk['step_violations']} violations"
        print(f"  {chk['field']:30s} {chk['scores']}  ({chk['expected_trend']}) {trend}")
        if chk["step_violations"] > 0:
            report["issues"].append(
                f"Non-monotonic {chk['field']}: {chk['scores']} (expected {chk['expected_trend']})"
            )

    if holdout_path.is_file():
        report["holdout"] = holdout_analysis(model, holdout_path, threshold)
        h = report["holdout"]
        print("\n--- Holdout test set (254 rows) ---")
        print(f"  Actual readmit rate: {h['positive_rate']:.1%}")
        print(f"  Mean predicted prob: {h['mean_pred']:.1%}")
        print(f"  Predicted elevated:  {h['pred_elevated_rate']:.1%}")
        print(f"  Prob range: [{h['min_prob']:.3f}, {h['max_prob']:.3f}]")
        print("  Calibration (pred bin -> actual rate):")
        for b in h["calibration_bins"]:
            gap = b["actual_rate"] - b["mean_pred"]
            print(
                f"    [{b['bin_lo']:.1f}-{b['bin_hi']:.1f}) n={b['n']:3d} "
                f"pred={b['mean_pred']:.2f} actual={b['actual_rate']:.2f} gap={gap:+.2f}"
            )

    if cohort is not None:
        user = next(row for _, row in TEST_CASES if _ == "user_screenshot")
        pct = {}
        for col in FEATURE_COLUMNS:
            if col == "sex":
                pct[col] = user[col]
            else:
                pct[col] = float((cohort[col] < user[col]).mean())
        report["cohort_percentiles_user_case"] = pct
        print("\n--- User case vs training cohort (percentile rank) ---")
        for col in FEATURE_COLUMNS:
            if col == "sex":
                print(f"  {col}: {user[col]}")
            else:
                print(f"  {col}: {user[col]} (above {pct[col]*100:.0f}% of rows)")

    print("\n--- Issues flagged ---")
    if report["issues"]:
        for issue in report["issues"]:
            print(f"  ! {issue}")
    else:
        print("  None from automated checks.")

    out_path = artifacts / "validation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0 if not report["issues"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
