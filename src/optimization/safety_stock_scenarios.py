
"""
Procurement AI - Safety Stock Scenario Analyzer
Enhancement 1E

Compares alternative service-level targets against the existing
inventory policy without modifying the baseline.

All scenarios are illustrative and use synthetic data.
"""

from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "data" / "processed" / "inventory_policy.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "safety_stock_scenarios.csv"

SCENARIO_LEVELS = [0.90, 0.95, 0.97, 0.99]
MAX_SAFETY_STOCK_WEEKS = 26.0
TOL = 1e-6

REQUIRED_COLUMNS = [
    "sku",
    "final_demand_class",
    "service_level_target",
    "service_level_z",
    "planning_weekly_demand",
    "planning_weekly_demand_std",
    "planning_lead_time_weeks",
    "lead_time_std_weeks",
    "policy_floor_safety_stock",
    "statistical_safety_stock",
    "safety_stock",
    "unit_cost",
]


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def load_data():
    section("LOADING SAFETY STOCK POLICY")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["sku"].isna().any():
        raise ValueError("Missing SKU values")

    if df["sku"].duplicated().any():
        raise ValueError("Duplicate SKUs detected")

    numeric_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in ["sku", "final_demand_class"]
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or invalid numeric policy values"
        )

    if not df["service_level_target"].between(
        0.50, 0.9999
    ).all():
        raise ValueError("Invalid service-level targets")

    nonnegative = [
        "planning_weekly_demand",
        "planning_weekly_demand_std",
        "lead_time_std_weeks",
        "policy_floor_safety_stock",
        "statistical_safety_stock",
        "safety_stock",
        "unit_cost",
    ]

    if (df[nonnegative] < 0).any().any():
        raise ValueError("Negative input values detected")

    if (df["planning_lead_time_weeks"] <= 0).any():
        raise ValueError("Invalid planning lead time")

    print(f"Loaded {len(df):,} SKUs")
    return df


def calculate_scenarios(policy):
    section("CALCULATING SAFETY STOCK SCENARIOS")

    rows = []

    for _, item in policy.iterrows():
        sku = item["sku"]

        weekly_demand = float(
            item["planning_weekly_demand"]
        )

        demand_std = float(
            item["planning_weekly_demand_std"]
        )

        lead_time = float(
            item["planning_lead_time_weeks"]
        )

        lead_time_std = float(
            item["lead_time_std_weeks"]
        )

        policy_floor = float(
            item["policy_floor_safety_stock"]
        )

        unit_cost = float(item["unit_cost"])

        baseline_units = int(item["safety_stock"])

        baseline_level = float(
            item["service_level_target"]
        )

        baseline_value = baseline_units * unit_cost

        demand_variance = lead_time * demand_std ** 2

        lead_time_variance = (
            weekly_demand ** 2 * lead_time_std ** 2
        )

        combined_variance = (
            demand_variance + lead_time_variance
        )

        cap = weekly_demand * MAX_SAFETY_STOCK_WEEKS

        # Include the exact existing target so the baseline
        # can be reproduced even if it is not in the standard
        # scenario list.
        levels = sorted(
            set(SCENARIO_LEVELS + [baseline_level])
        )

        for level in levels:
            z_score = NormalDist().inv_cdf(
                float(level)
            )

            statistical_stock = max(
                0.0,
                z_score * np.sqrt(combined_variance)
            )

            raw_stock = max(
                statistical_stock,
                policy_floor
            )

            # Mirror the existing inventory policy:
            # cap only when weekly demand is positive.
            if weekly_demand > 0:
                capped_stock = min(raw_stock, cap)
            else:
                capped_stock = raw_stock

            scenario_units = int(
                np.ceil(capped_stock)
            )

            # Preserve the exact validated baseline.
            # We also check the independently calculated
            # result during validation.
            is_baseline = abs(
                level - baseline_level
            ) < TOL

            calculated_units = scenario_units

            if is_baseline:
                scenario_units = baseline_units

            scenario_value = (
                scenario_units * unit_cost
            )

            unit_change = (
                scenario_units - baseline_units
            )

            value_change = (
                scenario_value - baseline_value
            )

            if unit_change > 0:
                direction = "INCREASE"
            elif unit_change < 0:
                direction = "DECREASE"
            else:
                direction = "NO CHANGE"

            rows.append({
                "sku": sku,
                "final_demand_class": item[
                    "final_demand_class"
                ],
                "baseline_service_level": baseline_level,
                "scenario_service_level": level,
                "is_baseline": is_baseline,
                "baseline_safety_stock": baseline_units,
                "calculated_safety_stock": calculated_units,
                "scenario_safety_stock": scenario_units,
                "safety_stock_unit_change": unit_change,
                "baseline_safety_stock_value": baseline_value,
                "scenario_safety_stock_value": scenario_value,
                "safety_stock_value_change": value_change,
                "unit_cost": unit_cost,
                "planning_weekly_demand": weekly_demand,
                "planning_lead_time_weeks": lead_time,
                "policy_floor_safety_stock": policy_floor,
                "scenario_statistical_safety_stock": (
                    statistical_stock
                ),
                "scenario_direction": direction,
                "requires_stockout_validation": (
                    unit_change < 0
                ),
                "scenario_limitation": (
                    "Illustrative policy sensitivity only; "
                    "does not establish actual service levels, "
                    "supplier feasibility, or stockout savings."
                ),
            })

    result = pd.DataFrame(rows)

    print(
        f"Generated {len(result):,} SKU scenarios"
    )

    return result


def validate_results(policy, scenarios):
    section("VALIDATING SAFETY STOCK SCENARIOS")

    checks = []

    def check(condition, description):
        if not condition:
            raise AssertionError(description)

        checks.append(description)
        print(f"PASS - {description}")

    expected_rows = sum(
        len(
            set(
                SCENARIO_LEVELS
                + [float(level)]
            )
        )
        for level in policy["service_level_target"]
    )

    check(
        len(scenarios) == expected_rows,
        "Scenario row count reconciles"
    )

    check(
        scenarios["sku"].nunique() == len(policy),
        "All SKUs preserved"
    )

    check(
        not scenarios.duplicated(
            ["sku", "scenario_service_level"]
        ).any(),
        "No duplicate SKU and service-level combinations"
    )

    check(
        not scenarios.isna().any().any(),
        "No missing scenario values"
    )

    check(
        (scenarios["scenario_safety_stock"] >= 0).all(),
        "No negative safety stock"
    )

    check(
        (
            scenarios["scenario_safety_stock_value"]
            >= 0
        ).all(),
        "No negative safety stock values"
    )

    baseline = scenarios[
        scenarios["is_baseline"]
    ].copy()

    check(
        len(baseline) == len(policy),
        "Exactly one baseline per SKU"
    )

    check(
        (
            baseline["scenario_safety_stock"]
            == baseline["baseline_safety_stock"]
        ).all(),
        "Baseline safety stock preserved"
    )

    check(
        (
            baseline["calculated_safety_stock"]
            == baseline["baseline_safety_stock"]
        ).all(),
        "Existing safety stock formula reproduced"
    )

    check(
        np.allclose(
            baseline["safety_stock_value_change"],
            0.0,
            atol=TOL
        ),
        "Baseline inventory investment unchanged"
    )

    check(
        np.allclose(
            scenarios["safety_stock_unit_change"],
            scenarios["scenario_safety_stock"]
            - scenarios["baseline_safety_stock"]
        ),
        "Unit changes reconcile"
    )

    check(
        np.allclose(
            scenarios["safety_stock_value_change"],
            scenarios["scenario_safety_stock_value"]
            - scenarios["baseline_safety_stock_value"],
            atol=0.01
        ),
        "Value changes reconcile"
    )

    check(
        (
            scenarios["requires_stockout_validation"]
            == (
                scenarios["safety_stock_unit_change"] < 0
            )
        ).all(),
        "Reduced stock scenarios flagged for validation"
    )

    # For each SKU, a higher service-level target
    # must not produce less safety stock.
    monotonic = True

    for _, group in scenarios.groupby("sku"):
        ordered = group.sort_values(
            "scenario_service_level"
        )

        if (
            ordered["scenario_safety_stock"]
            .diff()
            .dropna()
            < 0
        ).any():
            monotonic = False
            break

    check(
        monotonic,
        "Safety stock is nondecreasing with service level"
    )

    print(
        f"\nVALIDATION COMPLETE: {len(checks)} PASS, 0 FAIL"
    )


def print_summary(policy, scenarios):
    section("SAFETY STOCK FINANCIAL BASELINE")

    baseline_units = policy[
        "safety_stock"
    ].sum()

    baseline_value = (
        policy["safety_stock"]
        * policy["unit_cost"]
    ).sum()

    print(
        f"Baseline SKUs: {len(policy):,}"
    )

    print(
        f"Baseline safety stock units: "
        f"{baseline_units:,.0f}"
    )

    print(
        f"Baseline safety stock value USD: "
        f"{baseline_value:,.2f}"
    )

    section("SCENARIO COMPARISON")

    # Each target is a portfolio-wide hypothetical
    # service-level setting for comparison.
    comparison = scenarios[
        scenarios["scenario_service_level"].isin(
            SCENARIO_LEVELS
        )
    ].groupby(
        "scenario_service_level"
    ).agg(
        skus=("sku", "nunique"),
        safety_stock_units=(
            "scenario_safety_stock", "sum"
        ),
        safety_stock_value=(
            "scenario_safety_stock_value", "sum"
        ),
        unit_change=(
            "safety_stock_unit_change", "sum"
        ),
        value_change=(
            "safety_stock_value_change", "sum"
        ),
        reduced_skus=(
            "requires_stockout_validation", "sum"
        ),
    ).reset_index()

    comparison["value_change_pct"] = (
        comparison["value_change"]
        / baseline_value * 100
        if baseline_value > 0
        else 0.0
    )

    print(
        comparison.round(2).to_string(
            index=False
        )
    )

    print(
        "\nIMPORTANT: These are hypothetical changes "
        "to safety stock targets, not validated savings."
    )

    print(
        "Changing safety stock alone does not "
        "retroactively change the existing "
        "inventory projection or purchase orders."
    )


def save_results(scenarios):
    section("SAVING RESULTS")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    scenarios.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Saved: {OUTPUT_FILE}")


def main():
    policy = load_data()

    scenarios = calculate_scenarios(
        policy
    )

    validate_results(
        policy,
        scenarios
    )

    print_summary(
        policy,
        scenarios
    )

    save_results(
        scenarios
    )

    section("ENHANCEMENT 1E COMPLETE")


if __name__ == "__main__":
    main()
