
"""
Procurement AI - Forecast Risk Analysis
Enhancement 1F

Combines historical forecast accuracy with inventory and
supplier exposure to prioritize SKUs for planner review.

Read-only analysis. Does not modify forecasts, inventory
policies, purchase proposals, or supplier commitments.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

SELECTION_FILE = PROCESSED / "forecast_model_selection.csv"
POLICY_FILE = PROCESSED / "inventory_policy.csv"
PLANNING_FILE = PROCESSED / "planning_master.csv"

DETAIL_OUTPUT = PROCESSED / "forecast_risk_analysis.csv"
SUMMARY_OUTPUT = PROCESSED / "forecast_risk_summary.csv"

TOL = 1e-9


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def require_columns(df, columns, name):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def load_inputs():
    section("LOADING FORECAST RISK INPUTS")

    paths = {
        "selection": SELECTION_FILE,
        "policy": POLICY_FILE,
        "planning": PLANNING_FILE,
    }

    data = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        data[name] = pd.read_csv(path)
        print(f"{name}: {len(data[name]):,} rows")

    require_columns(
        data["selection"],
        [
            "sku",
            "selected_model",
            "selected_wape",
            "selected_mae",
            "selected_bias",
            "successful_backtest_windows",
            "naive_wape",
        ],
        "forecast_model_selection",
    )

    require_columns(
        data["policy"],
        [
            "sku",
            "final_demand_class",
            "forecast_readiness",
            "forecast_difficulty",
            "production_model",
            "planning_weekly_demand",
            "planning_lead_time_weeks",
            "safety_stock",
            "reorder_point",
            "net_available",
            "total_open_po_qty",
            "below_reorder_point",
            "supplier_risk_class",
            "supplier_risk_score",
        ],
        "inventory_policy",
    )

    require_columns(
        data["planning"],
        [
            "sku",
            "unit_cost",
            "currently_stocked_out",
            "has_overdue_po",
            "current_weeks_of_supply",
            "13w_lost_sales_rate",
        ],
        "planning_master",
    )

    for name, df in data.items():
        if df["sku"].isna().any():
            raise ValueError(f"{name}: missing SKUs")
        if df["sku"].duplicated().any():
            raise ValueError(f"{name}: duplicate SKUs")

    if len(data["policy"]) != 200:
        raise ValueError(
            "Expected 200 SKUs in the synthetic policy dataset"
        )

    return data


def parse_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    normalized = (
        series.astype(str).str.strip().str.lower()
    )

    valid = {
        "true", "false", "1", "0",
        "yes", "no", "nan", "none", "",
    }

    unexpected = set(normalized.unique()) - valid

    if unexpected:
        raise ValueError(
            f"Unexpected boolean values: {sorted(unexpected)}"
        )

    return normalized.isin({"true", "1", "yes"})


def build_analysis(data):
    section("BUILDING FORECAST RISK ANALYSIS")

    selection = data["selection"][
        [
            "sku",
            "selected_model",
            "selected_wape",
            "selected_mae",
            "selected_bias",
            "successful_backtest_windows",
            "naive_wape",
        ]
    ].copy()

    policy = data["policy"][
        [
            "sku",
            "final_demand_class",
            "forecast_readiness",
            "forecast_difficulty",
            "production_model",
            "planning_weekly_demand",
            "planning_lead_time_weeks",
            "safety_stock",
            "reorder_point",
            "net_available",
            "total_open_po_qty",
            "below_reorder_point",
            "supplier_risk_class",
            "supplier_risk_score",
        ]
    ].copy()

    planning = data["planning"][
        [
            "sku",
            "unit_cost",
            "currently_stocked_out",
            "has_overdue_po",
            "current_weeks_of_supply",
            "13w_lost_sales_rate",
        ]
    ].copy()

    result = policy.merge(
        selection,
        on="sku",
        how="left",
        validate="one_to_one",
        indicator="_selection_merge",
    )

    result["has_backtested_model"] = (
        result["_selection_merge"] == "both"
    )

    result = result.drop(columns="_selection_merge")

    result = result.merge(
        planning,
        on="sku",
        how="left",
        validate="one_to_one",
        indicator="_planning_merge",
    )

    if not (result["_planning_merge"] == "both").all():
        raise ValueError("Planning master missing policy SKUs")

    result = result.drop(columns="_planning_merge")

    numeric_columns = [
        "selected_wape",
        "selected_mae",
        "selected_bias",
        "successful_backtest_windows",
        "naive_wape",
        "planning_weekly_demand",
        "planning_lead_time_weeks",
        "safety_stock",
        "reorder_point",
        "net_available",
        "total_open_po_qty",
        "supplier_risk_score",
        "unit_cost",
        "current_weeks_of_supply",
        "13w_lost_sales_rate",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )

    required_numeric = [
        "planning_weekly_demand",
        "planning_lead_time_weeks",
        "safety_stock",
        "reorder_point",
        "net_available",
        "total_open_po_qty",
        "unit_cost",
    ]

    if result[required_numeric].isna().any().any():
        raise ValueError("Missing required planning values")

    if (
        result[
            [
                "planning_weekly_demand",
                "planning_lead_time_weeks",
                "safety_stock",
                "total_open_po_qty",
                "unit_cost",
            ]
        ] < 0
    ).any().any():
        raise ValueError("Negative planning inputs detected")

    for column in [
        "below_reorder_point",
        "currently_stocked_out",
        "has_overdue_po",
    ]:
        result[column] = parse_bool(result[column])

    # Bias convention in the existing backtest engine:
    # (forecast - actual) / actual
    # Negative bias therefore means underforecasting.
    result["underforecast_bias_pct"] = (
        (-result["selected_bias"]).clip(lower=0)
    )

    result["overforecast_bias_pct"] = (
        result["selected_bias"].clip(lower=0)
    )

    result["material_underforecast"] = (
        result["underforecast_bias_pct"] >= 0.10
    )

    result["high_forecast_error"] = (
        result["selected_wape"] > 0.30
    )

    result["very_high_forecast_error"] = (
        result["selected_wape"] > 0.60
    )

    result["long_lead_time"] = (
        result["planning_lead_time_weeks"] >= 10
    )

    # Available stock coverage is distinct from inventory
    # position and excludes unconfirmed incoming supply.
    demand = result["planning_weekly_demand"]

    result["available_coverage_weeks"] = np.where(
        demand > TOL,
        result["net_available"].clip(lower=0) / demand,
        np.nan,
    )

    result["coverage_below_lead_time"] = (
        result["available_coverage_weeks"]
        < result["planning_lead_time_weeks"]
    )

    result["forecast_error_exposure_units"] = (
        result["selected_mae"]
        * result["planning_lead_time_weeks"]
    )

    # MAE is an average absolute historical weekly error.
    # This multiplication is a heuristic exposure indicator,
    # not a forecast of future shortage quantity.
    result["forecast_error_exposure_value"] = (
        result["forecast_error_exposure_units"]
        * result["unit_cost"]
    )

    result["risk_score"] = 0

    rules = [
        ("material_underforecast", 25),
        ("high_forecast_error", 15),
        ("very_high_forecast_error", 10),
        ("coverage_below_lead_time", 20),
        ("below_reorder_point", 10),
        ("currently_stocked_out", 10),
        ("has_overdue_po", 5),
        ("long_lead_time", 5),
    ]

    for column, points in rules:
        result["risk_score"] += (
            result[column].fillna(False).astype(int)
            * points
        )

    # Missing backtests require review; do not assume
    # missing accuracy metrics indicate good performance.
    result.loc[
        ~result["has_backtested_model"],
        "risk_score",
    ] += 15

    result["risk_tier"] = np.select(
        [
            result["risk_score"] >= 60,
            result["risk_score"] >= 40,
            result["risk_score"] >= 20,
        ],
        [
            "CRITICAL",
            "HIGH",
            "MODERATE",
        ],
        default="LOW",
    )

    def recommend_action(row):
        if not row["has_backtested_model"]:
            return "REVIEW FORECAST FALLBACK AND DATA HISTORY"

        if (
            row["material_underforecast"]
            and row["coverage_below_lead_time"]
        ):
            return "INVESTIGATE UNDERFORECAST AND SUPPLY COVERAGE"

        if (
            row["high_forecast_error"]
            and row["coverage_below_lead_time"]
        ):
            return "REVIEW FORECAST MODEL AND REPLENISHMENT TIMING"

        if row["material_underforecast"]:
            return "INVESTIGATE SYSTEMATIC UNDERFORECASTING"

        if row["high_forecast_error"]:
            return "REVIEW FORECAST ACCURACY"

        if row["coverage_below_lead_time"]:
            return "REVIEW INVENTORY COVERAGE"

        return "MONITOR"

    result["recommended_review_action"] = result.apply(
        recommend_action, axis=1
    )

    result["analysis_limitation"] = (
        "Heuristic risk prioritization using synthetic data; "
        "not a validated probability of stockout or "
        "an optimized replenishment recommendation."
    )

    result = result.sort_values(
        ["risk_score", "forecast_error_exposure_value"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    print(f"Analyzed {len(result):,} SKUs")
    return result


def validate_analysis(data, result):
    section("VALIDATING FORECAST RISK ANALYSIS")

    checks = []

    def check(condition, label):
        if not bool(condition):
            raise AssertionError(label)
        checks.append(label)
        print(f"PASS - {label}")

    check(
        len(result) == len(data["policy"]),
        "All inventory policy SKUs preserved",
    )

    check(
        result["sku"].is_unique,
        "Exactly one risk record per SKU",
    )

    check(
        set(result["sku"]) == set(data["policy"]["sku"]),
        "SKU population reconciles",
    )

    check(
        int(result["has_backtested_model"].sum())
        == len(data["selection"]),
        "Backtested SKU count reconciles",
    )

    check(
        result["risk_score"].between(0, 115).all(),
        "Risk scores are within expected bounds",
    )

    check(
        result["risk_tier"].isin(
            ["CRITICAL", "HIGH", "MODERATE", "LOW"]
        ).all(),
        "All SKUs assigned valid risk tiers",
    )

    check(
        result["recommended_review_action"].notna().all(),
        "All SKUs assigned a review action",
    )

    check(
        (
            result.loc[
                result["selected_bias"] < 0,
                "underforecast_bias_pct",
            ] > 0
        ).all(),
        "Negative bias correctly identifies underforecasting",
    )

    check(
        (
            result.loc[
                result["selected_bias"] > 0,
                "underforecast_bias_pct",
            ] == 0
        ).all(),
        "Positive bias is not classified as underforecasting",
    )

    check(
        (
            result.loc[
                result["material_underforecast"],
                "selected_bias",
            ] <= -0.10 + TOL
        ).all(),
        "Material underforecast threshold reconciles",
    )

    check(
        (
            result.loc[
                result["high_forecast_error"],
                "selected_wape",
            ] > 0.30
        ).all(),
        "High forecast error threshold reconciles",
    )

    check(
        (
            result["forecast_error_exposure_value"].dropna()
            >= 0
        ).all(),
        "No negative forecast exposure values",
    )

    check(
        result["risk_score"].notna().all(),
        "No missing risk scores",
    )

    check(
        (
            result.loc[
                ~result["has_backtested_model"],
                "selected_model",
            ].isna()
        ).all(),
        "Missing backtests remain explicitly identifiable",
    )

    print(
        f"\nVALIDATION COMPLETE: {len(checks)} PASS, 0 FAIL"
    )


def build_summary(result):
    section("FORECAST RISK SUMMARY")

    summary = (
        result.groupby("risk_tier", as_index=False)
        .agg(
            skus=("sku", "count"),
            material_underforecast_skus=(
                "material_underforecast", "sum"
            ),
            high_error_skus=(
                "high_forecast_error", "sum"
            ),
            below_lead_time_coverage_skus=(
                "coverage_below_lead_time", "sum"
            ),
            average_wape=("selected_wape", "mean"),
            average_risk_score=("risk_score", "mean"),
            heuristic_exposure_value=(
                "forecast_error_exposure_value", "sum"
            ),
        )
    )

    tier_order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MODERATE": 2,
        "LOW": 3,
    }

    summary["_order"] = summary["risk_tier"].map(
        tier_order
    )

    summary = (
        summary.sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    print(
        summary.round(2).to_string(index=False)
    )

    print("\nADDITIONAL FINDINGS")
    print(
        "SKUs with selected backtests:",
        int(result["has_backtested_model"].sum()),
    )
    print(
        "SKUs without selected backtests:",
        int((~result["has_backtested_model"]).sum()),
    )
    print(
        "Material underforecast SKUs:",
        int(result["material_underforecast"].sum()),
    )
    print(
        "High forecast error SKUs:",
        int(result["high_forecast_error"].sum()),
    )
    print(
        "Coverage below lead time:",
        int(result["coverage_below_lead_time"].sum()),
    )

    print(
        "\nExposure value is a prioritization proxy, "
        "not an expected financial loss."
    )

    return summary


def save_outputs(result, summary):
    section("SAVING FORECAST RISK OUTPUTS")

    DETAIL_OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    result.to_csv(
        DETAIL_OUTPUT, index=False
    )

    summary.to_csv(
        SUMMARY_OUTPUT, index=False
    )

    print(f"Saved: {DETAIL_OUTPUT}")
    print(f"Saved: {SUMMARY_OUTPUT}")


def main():
    data = load_inputs()

    result = build_analysis(data)

    validate_analysis(data, result)

    summary = build_summary(result)

    save_outputs(result, summary)

    section("ENHANCEMENT 1F ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
