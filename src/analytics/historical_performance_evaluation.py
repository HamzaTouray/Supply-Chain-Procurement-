
"""
Procurement AI - Historical Performance Evaluation
Enhancement 1G

Evaluates selected forecasting models against naive baselines.
Read-only: does not modify models, forecasts, or planning logic.

Important:
These are model-selection backtest results, not an independent
holdout evaluation or an inventory policy backtest.
"""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

BACKTEST_PATH = DATA / "forecast_backtest_results.csv"
SELECTION_PATH = DATA / "forecast_model_selection.csv"
POLICY_PATH = DATA / "inventory_policy.csv"
RISK_PATH = DATA / "forecast_risk_analysis.csv"

DETAIL_PATH = DATA / "historical_performance_detail.csv"
CLASS_PATH = DATA / "historical_performance_by_class.csv"
MODEL_PATH = DATA / "historical_performance_by_model.csv"
SUMMARY_PATH = DATA / "historical_performance_summary.csv"

TOL = 1e-9


def check(condition, message):
    if not bool(condition):
        raise AssertionError(message)
    print(f"PASS - {message}")


def required(df, columns, name):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )


def load_data():
    print("\nLOADING HISTORICAL EVALUATION DATA")

    for path in [
        BACKTEST_PATH,
        SELECTION_PATH,
        POLICY_PATH,
        RISK_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    backtest = pd.read_csv(BACKTEST_PATH)
    selection = pd.read_csv(SELECTION_PATH)
    policy = pd.read_csv(POLICY_PATH)
    risk = pd.read_csv(RISK_PATH)

    required(
        backtest,
        [
            "sku", "model", "final_demand_class",
            "successful_windows", "actual_units",
            "forecast_units", "wape", "mae",
            "rmse", "forecast_bias",
        ],
        "backtest",
    )

    required(
        selection,
        [
            "sku", "selected_model", "selected_wape",
            "selected_mae", "selected_rmse",
            "selected_bias", "successful_backtest_windows",
            "naive_wape", "naive_mae",
        ],
        "selection",
    )

    required(
        policy,
        ["sku", "production_model"],
        "inventory policy",
    )

    required(
        risk,
        ["sku", "risk_tier", "risk_score"],
        "forecast risk",
    )

    for name, df, keys in [
        ("backtest", backtest, ["sku", "model"]),
        ("selection", selection, ["sku"]),
        ("policy", policy, ["sku"]),
        ("risk", risk, ["sku"]),
    ]:
        check(
            not df.duplicated(keys).any(),
            f"{name} has unique keys",
        )

    print(
        f"Backtests: {len(backtest)} | "
        f"Selected: {len(selection)} | "
        f"Policy: {len(policy)} | "
        f"Risk: {len(risk)}"
    )

    return backtest, selection, policy, risk


def build_detail(backtest, selection, policy, risk):
    print("\nBUILDING SKU PERFORMANCE DETAIL")

    selected = backtest.rename(
        columns={
            "model": "backtested_selected_model",
            "successful_windows": "selected_windows",
            "actual_units": "selected_actual_units",
            "forecast_units": "selected_forecast_units",
            "wape": "backtest_selected_wape",
            "mae": "backtest_selected_mae",
            "rmse": "backtest_selected_rmse",
            "forecast_bias": "backtest_selected_bias",
        }
    )

    detail = selection.merge(
        selected[
            [
                "sku", "backtested_selected_model",
                "selected_windows", "selected_actual_units",
                "selected_forecast_units",
                "backtest_selected_wape",
                "backtest_selected_mae",
                "backtest_selected_rmse",
                "backtest_selected_bias",
            ]
        ],
        left_on=["sku", "selected_model"],
        right_on=["sku", "backtested_selected_model"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    check(
        (detail["_merge"] == "both").all(),
        "Every selected model has a backtest",
    )
    detail.drop(columns="_merge", inplace=True)

    naive = backtest.loc[
        backtest["model"] == "naive",
        [
            "sku", "successful_windows", "actual_units",
            "forecast_units", "wape", "mae",
            "rmse", "forecast_bias",
        ],
    ].rename(
        columns={
            "successful_windows": "naive_windows",
            "actual_units": "naive_actual_units",
            "forecast_units": "naive_forecast_units",
            "wape": "backtest_naive_wape",
            "mae": "backtest_naive_mae",
            "rmse": "naive_rmse",
            "forecast_bias": "naive_bias",
        }
    )

    detail = detail.merge(
        naive,
        on="sku",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    check(
        (detail["_merge"] == "both").all(),
        "Every selected SKU has a naive baseline",
    )
    detail.drop(columns="_merge", inplace=True)

    detail = detail.merge(
        policy[["sku", "production_model"]],
        on="sku",
        how="left",
        validate="one_to_one",
    )

    detail = detail.merge(
        risk[["sku", "risk_tier", "risk_score"]],
        on="sku",
        how="left",
        validate="one_to_one",
    )

    comparisons = [
        ("selected_wape", "backtest_selected_wape"),
        ("selected_mae", "backtest_selected_mae"),
        ("selected_rmse", "backtest_selected_rmse"),
        ("selected_bias", "backtest_selected_bias"),
        ("naive_wape", "backtest_naive_wape"),
        ("naive_mae", "backtest_naive_mae"),
        ("successful_backtest_windows", "selected_windows"),
    ]

    for a, b in comparisons:
        equal = np.isclose(
            detail[a].to_numpy(dtype=float),
            detail[b].to_numpy(dtype=float),
            atol=1e-8,
            rtol=1e-8,
            equal_nan=True,
        )
        check(equal.all(), f"{a} reconciles with backtests")

    # Equal windows and actual volumes are required for
    # comparable per-SKU WAPE results.
    check(
        np.isclose(
            detail["selected_windows"],
            detail["naive_windows"],
        ).all(),
        "Selected and naive use equal window counts",
    )

    check(
        np.isclose(
            detail["selected_actual_units"],
            detail["naive_actual_units"],
            atol=1e-7,
        ).all(),
        "Selected and naive have equal actual units",
    )

    detail["wape_improvement_pp"] = (
        (detail["naive_wape"] - detail["selected_wape"])
        * 100
    )

    detail["mae_improvement_units"] = (
        detail["naive_mae"] - detail["selected_mae"]
    )

    detail["rmse_improvement_units"] = (
        detail["naive_rmse"] - detail["selected_rmse"]
    )

    detail["relative_wape_improvement_pct"] = np.where(
        detail["naive_wape"] > TOL,
        100
        * (detail["naive_wape"] - detail["selected_wape"])
        / detail["naive_wape"],
        np.nan,
    )

    detail["absolute_selected_bias_pct"] = (
        detail["selected_bias"].abs() * 100
    )

    detail["absolute_naive_bias_pct"] = (
        detail["naive_bias"].abs() * 100
    )

    detail["selected_bias_direction"] = np.select(
        [
            detail["selected_bias"] < -TOL,
            detail["selected_bias"] > TOL,
        ],
        ["UNDERFORECAST", "OVERFORECAST"],
        default="NEUTRAL",
    )

    detail["wape_result"] = np.select(
        [
            detail["wape_improvement_pp"] > 1e-7,
            detail["wape_improvement_pp"] < -1e-7,
        ],
        ["BETTER", "WORSE"],
        default="TIED",
    )

    detail["selected_matches_production"] = (
        detail["selected_model"]
        == detail["production_model"]
    )

    detail["evaluation_status"] = np.select(
        [
            detail["wape_result"] == "WORSE",
            detail["selected_wape"] > 0.60,
            detail["selected_bias"] <= -0.10,
            detail["selected_wape"] > 0.30,
        ],
        [
            "REVIEW BASELINE UNDERPERFORMANCE",
            "REVIEW VERY HIGH FORECAST ERROR",
            "REVIEW MATERIAL UNDERFORECAST",
            "REVIEW HIGH FORECAST ERROR",
        ],
        default="MONITOR",
    )

    detail["evaluation_limitation"] = (
        "Model-selection rolling backtest; not an "
        "independent holdout or inventory outcome evaluation."
    )

    return detail.sort_values(
        ["risk_score", "selected_wape"],
        ascending=[False, False],
    ).reset_index(drop=True)


def weighted_wape(df, prefix):
    """
    WAPE across SKUs using backtest actual units.
    Each SKU's WAPE already aggregates its evaluation
    windows, so this is an actual-volume-weighted result.
    """
    actual = df[f"{prefix}_actual_units"].sum()

    if actual <= TOL:
        return np.nan

    errors = (
        df[f"{prefix}_actual_units"]
        * df[
            "selected_wape"
            if prefix == "selected"
            else "naive_wape"
        ]
    ).sum()

    return float(errors / actual)


def aggregate_group(group):
    selected_wape = weighted_wape(group, "selected")
    naive_wape = weighted_wape(group, "naive")

    return pd.Series(
        {
            "skus": len(group),
            "better_skus": int(
                (group["wape_result"] == "BETTER").sum()
            ),
            "tied_skus": int(
                (group["wape_result"] == "TIED").sum()
            ),
            "worse_skus": int(
                (group["wape_result"] == "WORSE").sum()
            ),
            "mean_selected_wape": group[
                "selected_wape"
            ].mean(),
            "mean_naive_wape": group[
                "naive_wape"
            ].mean(),
            "weighted_selected_wape": selected_wape,
            "weighted_naive_wape": naive_wape,
            "weighted_wape_improvement_pp": (
                100 * (naive_wape - selected_wape)
            ),
            "material_underforecast_skus": int(
                (group["selected_bias"] <= -0.10).sum()
            ),
            "high_error_skus": int(
                (group["selected_wape"] > 0.30).sum()
            ),
            "production_model_mismatches": int(
                (~group["selected_matches_production"]).sum()
            ),
        }
    )


def build_outputs(detail):
    print("\nBUILDING PERFORMANCE SUMMARIES")

    by_class = pd.DataFrame(
        [
            {
                "final_demand_class": name,
                **aggregate_group(group).to_dict(),
            }
            for name, group in detail.groupby(
                "final_demand_class"
            )
        ]
    )

    by_model = pd.DataFrame(
        [
            {
                "selected_model": name,
                **aggregate_group(group).to_dict(),
            }
            for name, group in detail.groupby(
                "selected_model"
            )
        ]
    )

    overall = aggregate_group(detail).to_dict()
    overall["scope"] = "ALL_BACKTESTED_SKUS"
    overall["evaluation_type"] = (
        "MODEL_SELECTION_BACKTEST_NOT_INDEPENDENT_HOLDOUT"
    )

    summary = pd.DataFrame([overall])

    check(
        int(by_class["skus"].sum()) == len(detail),
        "Demand-class counts reconcile",
    )

    check(
        int(by_model["skus"].sum()) == len(detail),
        "Model counts reconcile",
    )

    check(
        (
            summary["better_skus"].iloc[0]
            + summary["tied_skus"].iloc[0]
            + summary["worse_skus"].iloc[0]
        ) == len(detail),
        "Performance categories reconcile",
    )

    check(
        (detail["wape_result"] != "WORSE").all(),
        "No selected model underperforms naive on WAPE",
    )

    check(
        detail["risk_tier"].notna().all(),
        "Forecast risk classification reconciles",
    )

    return by_class, by_model, summary


def save_outputs(detail, by_class, by_model, summary):
    print("\nSAVING HISTORICAL PERFORMANCE REPORTS")

    for path, df in [
        (DETAIL_PATH, detail),
        (CLASS_PATH, by_class),
        (MODEL_PATH, by_model),
        (SUMMARY_PATH, summary),
    ]:
        df.to_csv(path, index=False)
        print(f"Saved: {path.name} ({len(df)} rows)")


def main():
    backtest, selection, policy, risk = load_data()

    detail = build_detail(
        backtest, selection, policy, risk
    )

    by_class, by_model, summary = build_outputs(
        detail
    )

    save_outputs(
        detail, by_class, by_model, summary
    )

    print("\nOVERALL PERFORMANCE")
    print(
        summary[
            [
                "skus",
                "better_skus",
                "tied_skus",
                "worse_skus",
                "weighted_selected_wape",
                "weighted_naive_wape",
                "weighted_wape_improvement_pp",
                "production_model_mismatches",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nPERFORMANCE BY DEMAND CLASS")
    print(
        by_class[
            [
                "final_demand_class",
                "skus",
                "better_skus",
                "weighted_selected_wape",
                "weighted_naive_wape",
                "weighted_wape_improvement_pp",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        "\nIMPORTANT: These results reuse the "
        "model-selection backtests. Independent "
        "holdout performance and inventory outcomes "
        "have not been established."
    )

    print("\nENHANCEMENT 1G EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
