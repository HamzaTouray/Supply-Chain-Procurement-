
from pathlib import Path
import math

import numpy as np
import pandas as pd


# ==========================================================
# ENHANCEMENT 1A - MINIMUM BRIDGE QUANTITY ANALYSIS
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

ACTIONS_PATH = DATA / "feasible_supply_actions.csv"
PROJECTION_PATH = DATA / "feasible_inventory_projection.csv"
RECOVERY_PATH = DATA / "recovery_action_plan.csv"

OUTPUT_PATH = DATA / "bridge_quantity_analysis.csv"

TOL = 1e-7


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("PASS:", message)


def load_data():
    section("LOADING EXISTING PLANNING OUTPUTS")

    files = {
        "actions": ACTIONS_PATH,
        "projection": PROJECTION_PATH,
        "recovery": RECOVERY_PATH,
    }

    data = {}

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        data[name] = pd.read_csv(path)
        print(f"Loaded {path.name}: {len(data[name]):,} rows")

    return data


def prepare(data):
    actions = data["actions"].copy()
    projection = data["projection"].copy()
    recovery = data["recovery"].copy()

    required = {
        "actions": [
            "sku",
            "earliest_feasible_receipt",
            "recommended_order_qty",
        ],
        "projection": [
            "sku",
            "forecast_week",
            "beginning_inventory",
            "confirmed_existing_receipts",
            "feasible_new_receipts",
            "forecast_demand",
            "ending_inventory",
            "safety_stock",
        ],
        "recovery": [
            "sku",
            "supplier_id",
            "priority",
            "proposed_expedite_units",
        ],
    }

    for name, columns in required.items():
        missing = set(columns) - set(data[name].columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    actions["earliest_feasible_receipt"] = pd.to_datetime(
        actions["earliest_feasible_receipt"],
        errors="coerce",
    )

    projection["forecast_week"] = pd.to_datetime(
        projection["forecast_week"],
        errors="coerce",
    )

    if actions["earliest_feasible_receipt"].isna().any():
        raise ValueError("Invalid feasible receipt dates")

    if projection["forecast_week"].isna().any():
        raise ValueError("Invalid forecast week dates")

    numeric_projection = [
        "beginning_inventory",
        "confirmed_existing_receipts",
        "feasible_new_receipts",
        "forecast_demand",
        "ending_inventory",
        "safety_stock",
    ]

    for column in numeric_projection:
        projection[column] = pd.to_numeric(
            projection[column],
            errors="raise",
        )

    actions["recommended_order_qty"] = pd.to_numeric(
        actions["recommended_order_qty"],
        errors="raise",
    )

    if projection[numeric_projection].isna().any().any():
        raise ValueError("Missing projection numeric values")

    if not np.isfinite(
        projection[numeric_projection].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite projection values")

    if projection[["sku", "forecast_week"]].duplicated().any():
        raise ValueError("Duplicate SKU/week projection records")

    if recovery["sku"].duplicated().any():
        raise ValueError("Duplicate recovery SKU records")

    if (actions["recommended_order_qty"] < 0).any():
        raise ValueError("Negative recommended order quantity")

    if (projection["forecast_demand"] < 0).any():
        raise ValueError("Negative forecast demand")

    return actions, projection, recovery


def calculate_bridge(actions, projection, recovery):
    section("CALCULATING MINIMUM BRIDGE QUANTITIES")

    results = []

    first_receipts = (
        actions.groupby("sku")["earliest_feasible_receipt"]
        .min()
        .to_dict()
    )

    for sku, group in projection.groupby("sku", sort=True):
        group = group.sort_values("forecast_week").copy()

        first_normal_receipt = first_receipts.get(sku, pd.NaT)

        # This is the first week bucket that may contain
        # a newly planned normal replenishment.
        if pd.notna(first_normal_receipt):
            receipt_week = (
                first_normal_receipt
                - pd.Timedelta(
                    days=first_normal_receipt.weekday()
                )
            )

            # Analyze shortage exposure strictly before
            # the first feasible new replenishment week.
            bridge_period = group[
                group["forecast_week"] < receipt_week
            ].copy()

        else:
            receipt_week = pd.NaT

            # No new planned supply: measure exposure
            # over the available planning horizon.
            bridge_period = group.copy()

        if bridge_period.empty:
            peak_stockout_gap = 0.0
            peak_safety_gap = 0.0
            first_stockout = pd.NaT
            first_safety_breach = pd.NaT
            stockout_weeks = 0
            safety_breach_weeks = 0

        else:
            # The feasible projection already includes
            # confirmed existing PO receipts and demand.
            # No additional PO quantities are added here.
            stockout_gap = (
                -bridge_period["ending_inventory"]
            ).clip(lower=0)

            safety_gap = (
                bridge_period["safety_stock"]
                - bridge_period["ending_inventory"]
            ).clip(lower=0)

            peak_stockout_gap = float(stockout_gap.max())
            peak_safety_gap = float(safety_gap.max())

            stockout_mask = stockout_gap > TOL
            safety_mask = safety_gap > TOL

            stockout_weeks = int(stockout_mask.sum())
            safety_breach_weeks = int(safety_mask.sum())

            first_stockout = (
                bridge_period.loc[
                    stockout_mask, "forecast_week"
                ].min()
                if stockout_mask.any()
                else pd.NaT
            )

            first_safety_breach = (
                bridge_period.loc[
                    safety_mask, "forecast_week"
                ].min()
                if safety_mask.any()
                else pd.NaT
            )

        # Round up to whole units for planning.
        bridge_units = max(
            0,
            math.ceil(peak_stockout_gap - TOL),
        )

        safety_bridge_units = max(
            bridge_units,
            math.ceil(peak_safety_gap - TOL),
        )

        additional_buffer_units = (
            safety_bridge_units - bridge_units
        )

        if bridge_units > 0:
            bridge_status = "STOCKOUT BRIDGE REQUIRED"

        elif additional_buffer_units > 0:
            bridge_status = "SAFETY STOCK BRIDGE ONLY"

        else:
            bridge_status = "NO PRE-RECEIPT BRIDGE"

        results.append(
            {
                "sku": sku,
                "first_feasible_normal_receipt":
                    first_normal_receipt,
                "first_feasible_receipt_week":
                    receipt_week,
                "first_pre_receipt_stockout_week":
                    first_stockout,
                "first_pre_receipt_safety_breach_week":
                    first_safety_breach,
                "pre_receipt_stockout_weeks":
                    stockout_weeks,
                "pre_receipt_safety_breach_weeks":
                    safety_breach_weeks,
                "peak_pre_receipt_stockout_gap":
                    peak_stockout_gap,
                "peak_pre_receipt_safety_gap":
                    peak_safety_gap,
                "minimum_stockout_bridge_units":
                    bridge_units,
                "minimum_safety_bridge_units":
                    safety_bridge_units,
                "additional_safety_buffer_units":
                    additional_buffer_units,
                "bridge_status":
                    bridge_status,
            }
        )

    result = pd.DataFrame(results)

    result = result.merge(
        recovery[
            [
                "sku",
                "supplier_id",
                "priority",
                "proposed_expedite_units",
            ]
        ],
        on="sku",
        how="left",
        validate="one_to_one",
    )

    result["proposed_expedite_units"] = (
        pd.to_numeric(
            result["proposed_expedite_units"],
            errors="coerce",
        )
        .fillna(0)
    )

    result["bridge_vs_existing_expedite_units"] = (
        result["minimum_stockout_bridge_units"]
        - result["proposed_expedite_units"]
    )

    return result


def validate(actions, projection, recovery, result):
    section("BRIDGE QUANTITY VALIDATION")

    check(
        len(result) == projection["sku"].nunique(),
        "One bridge record per projected SKU",
    )

    check(
        result["sku"].is_unique,
        "No duplicate SKU records",
    )

    check(
        set(result["sku"]) == set(projection["sku"]),
        "All projected SKUs preserved",
    )

    check(
        (
            result["minimum_stockout_bridge_units"] >= 0
        ).all(),
        "Nonnegative stockout bridge quantities",
    )

    check(
        (
            result["minimum_safety_bridge_units"]
            >= result["minimum_stockout_bridge_units"]
        ).all(),
        "Safety bridge is at least stockout bridge",
    )

    check(
        (
            result["additional_safety_buffer_units"]
            == (
                result["minimum_safety_bridge_units"]
                - result["minimum_stockout_bridge_units"]
            )
        ).all(),
        "Additional safety buffer reconciles",
    )

    check(
        (
            result.loc[
                result["bridge_status"]
                == "NO PRE-RECEIPT BRIDGE",
                "minimum_safety_bridge_units",
            ] == 0
        ).all(),
        "No-bridge SKUs have zero bridge requirement",
    )

    check(
        len(result) == len(recovery),
        "Recovery SKU count preserved",
    )

    print("\nAll bridge quantity validations passed.")


def save(result):
    section("SAVING ENHANCEMENT OUTPUT")

    output = result.copy()

    date_columns = [
        "first_feasible_normal_receipt",
        "first_feasible_receipt_week",
        "first_pre_receipt_stockout_week",
        "first_pre_receipt_safety_breach_week",
    ]

    for column in date_columns:
        output[column] = pd.to_datetime(
            output[column],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    output.to_csv(OUTPUT_PATH, index=False)

    print("Saved:", OUTPUT_PATH)


def print_summary(result):
    section("BRIDGE QUANTITY RESULTS")

    print(f"SKUs analyzed: {len(result):,}")

    print(
        "SKUs requiring stockout bridge:",
        int(
            (
                result["minimum_stockout_bridge_units"] > 0
            ).sum()
        ),
    )

    print(
        "SKUs requiring safety-only bridge:",
        int(
            (
                result["bridge_status"]
                == "SAFETY STOCK BRIDGE ONLY"
            ).sum()
        ),
    )

    print(
        "Total stockout bridge units:",
        f"{result['minimum_stockout_bridge_units'].sum():,.0f}",
    )

    print(
        "Additional safety buffer units:",
        f"{result['additional_safety_buffer_units'].sum():,.0f}",
    )

    print("\nBridge status distribution:")
    print(result["bridge_status"].value_counts().to_string())

    print("\nTop 15 pre-receipt stockout exposures:")

    columns = [
        "sku",
        "supplier_id",
        "priority",
        "first_pre_receipt_stockout_week",
        "first_feasible_normal_receipt",
        "minimum_stockout_bridge_units",
        "additional_safety_buffer_units",
        "proposed_expedite_units",
        "bridge_status",
    ]

    top = result.sort_values(
        "minimum_stockout_bridge_units",
        ascending=False,
    ).head(15)

    print(top[columns].to_string(index=False))

    print(
        "\nNOTE: Bridge quantities are scenario estimates."
        "\nThey assume supplemental supply could arrive before"
        " the shortage."
        "\nThey are not approved purchase orders or"
        " confirmed expedited shipments."
    )


def main():
    data = load_data()

    actions, projection, recovery = prepare(data)

    result = calculate_bridge(
        actions,
        projection,
        recovery,
    )

    validate(
        actions,
        projection,
        recovery,
        result,
    )

    save(result)

    print_summary(result)

    section("ENHANCEMENT 1A COMPLETE")


if __name__ == "__main__":
    main()
