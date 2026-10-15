
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ENHANCEMENT 1D - RECOVERY SIMULATION
# ============================================================
# Independently simulates one existing PO line at a time.
#
# Confirmed PO:
#   Removes the original receipt and moves it earlier.
#
# Uncertain PO:
#   Adds a hypothetical receipt because the conservative
#   baseline does not currently count it.
#
# All scenarios are conditional, not supplier commitments.
# The existing planning engine and approval DB are untouched.
# ============================================================


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

PROJECTION_FILE = DATA / "feasible_inventory_projection.csv"
RECOVERY_FILE = DATA / "existing_po_recovery_lines.csv"
FREIGHT_FILE = DATA / "freight_recovery_scenarios.csv"

OUTPUT_FILE = DATA / "recovery_simulation_results.csv"

# Illustrative incremental transportation premium.
# Not a real carrier quote.
AIR_PREMIUM_PER_UNIT = 10.0

TOL = 0.000001


def check(condition, message):
    if not condition:
        raise ValueError(message)
    print(f"PASS: {message}")


def monday(date):
    date = pd.Timestamp(date)
    return date.normalize() - pd.Timedelta(
        days=date.weekday()
    )


def load_inputs():
    paths = {
        "projection": PROJECTION_FILE,
        "recovery": RECOVERY_FILE,
        "freight": FREIGHT_FILE,
    }

    data = {}

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)

        data[name] = pd.read_csv(path)
        print(f"Loaded {path.name}: {len(data[name]):,} rows")

    return data


def prepare_inputs(data):
    projection = data["projection"].copy()
    recovery = data["recovery"].copy()
    freight = data["freight"].copy()

    required_projection = {
        "sku", "forecast_week", "beginning_inventory",
        "confirmed_existing_receipts",
        "feasible_new_receipts", "forecast_demand",
        "ending_inventory", "safety_stock",
    }

    required_recovery = {
        "po_number", "sku", "open_qty",
        "expected_date", "uncertain_receipt",
        "included_as_confirmed",
        "first_pre_receipt_stockout_week",
        "priority", "recovery_action",
    }

    required_freight = {
        "sku", "snapshot_date",
    }

    for name, frame, columns in [
        ("Projection", projection, required_projection),
        ("Recovery", recovery, required_recovery),
        ("Freight", freight, required_freight),
    ]:
        missing = columns - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    for column in ["forecast_week"]:
        projection[column] = pd.to_datetime(
            projection[column], errors="coerce"
        )

    for column in [
        "expected_date",
        "first_pre_receipt_stockout_week",
    ]:
        recovery[column] = pd.to_datetime(
            recovery[column], errors="coerce"
        )

    freight["snapshot_date"] = pd.to_datetime(
        freight["snapshot_date"], errors="coerce"
    )

    for column in [
        "uncertain_receipt",
        "included_as_confirmed",
    ]:
        values = (
            recovery[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        check(
            values.isin(["true", "false"]).all(),
            f"Valid Boolean values in {column}"
        )

        recovery[column] = values.eq("true")

    numeric_columns = [
        "beginning_inventory",
        "confirmed_existing_receipts",
        "feasible_new_receipts",
        "forecast_demand",
        "ending_inventory",
        "safety_stock",
    ]

    for column in numeric_columns:
        projection[column] = pd.to_numeric(
            projection[column], errors="coerce"
        )

    recovery["open_qty"] = pd.to_numeric(
        recovery["open_qty"], errors="coerce"
    )

    check(
        projection[numeric_columns].notna().all().all(),
        "Projection numerical values complete"
    )

    check(
        recovery["open_qty"].notna().all()
        and (recovery["open_qty"] > 0).all(),
        "Recovery PO quantities valid"
    )

    check(
        projection["forecast_week"].notna().all()
        and recovery["expected_date"].notna().all()
        and freight["snapshot_date"].notna().all(),
        "Required dates valid"
    )

    check(
        freight["snapshot_date"].nunique() == 1,
        "Single planning snapshot"
    )

    check(
        not projection.duplicated(
            ["sku", "forecast_week"]
        ).any(),
        "Unique SKU/week projection records"
    )

    check(
        not freight["sku"].duplicated().any(),
        "Unique freight scenario SKUs"
    )

    check(
        set(recovery["sku"]).issubset(
            set(projection["sku"])
        ),
        "Recovery SKUs exist in projection"
    )

    check(
        (
            recovery["uncertain_receipt"]
            != recovery["included_as_confirmed"]
        ).all(),
        "Confirmed and uncertain classifications agree"
    )

    projection = projection.sort_values(
        ["sku", "forecast_week"]
    ).reset_index(drop=True)

    # Reproduce the original inventory projection before
    # testing any hypothetical recovery changes.
    for sku, group in projection.groupby("sku"):
        inventory = float(
            group.iloc[0]["beginning_inventory"]
        )

        for _, week in group.iterrows():
            check_value = (
                inventory
                + week["confirmed_existing_receipts"]
                + week["feasible_new_receipts"]
                - week["forecast_demand"]
            )

            if (
                abs(
                    check_value
                    - week["ending_inventory"]
                ) > TOL
            ):
                raise ValueError(
                    f"Baseline reproduction failed: {sku}"
                )

            inventory = check_value

    print(
        "PASS: All original SKU inventory "
        "projections reproduced"
    )

    return projection, recovery, freight


def calculate_metrics(ending_inventory, safety_stock):
    ending = np.asarray(
        ending_inventory, dtype=float
    )

    safety = np.asarray(
        safety_stock, dtype=float
    )

    # These are weekly ending-balance measurements.
    # They are not lost-sales or fill-rate estimates.
    negative_gap = np.maximum(-ending, 0)

    safety_gap = np.maximum(
        safety - ending, 0
    )

    return {
        "stockout_weeks": int(
            np.sum(ending < -TOL)
        ),
        "cumulative_negative_balance_units": float(
            negative_gap.sum()
        ),
        "peak_negative_balance_units": float(
            negative_gap.max()
        ),
        "safety_breach_weeks": int(
            np.sum(ending < safety - TOL)
        ),
        "cumulative_safety_gap_units": float(
            safety_gap.sum()
        ),
    }


def simulate_inventory(group, receipts):
    inventory = float(
        group.iloc[0]["beginning_inventory"]
    )

    endings = []

    for index, (_, week) in enumerate(group.iterrows()):
        inventory = (
            inventory
            + receipts[index]
            + float(week["feasible_new_receipts"])
            - float(week["forecast_demand"])
        )

        endings.append(inventory)

    return np.array(endings)


def simulate_recovery(projection, recovery, freight):
    snapshot = freight["snapshot_date"].iloc[0]

    projection_lookup = {
        sku: group.sort_values(
            "forecast_week"
        ).reset_index(drop=True)
        for sku, group in projection.groupby("sku")
    }

    results = []

    for _, po in recovery.iterrows():
        sku = po["sku"]
        group = projection_lookup[sku]

        weeks = pd.DatetimeIndex(
            group["forecast_week"]
        )

        first_week = weeks.min()
        last_week = weeks.max()

        quantity = float(po["open_qty"])

        stockout_week = (
            po["first_pre_receipt_stockout_week"]
        )

        original_week = monday(
            po["expected_date"]
        )

        # This is an analytical what-if:
        # receipt available in the bucket of the first
        # forecast stockout. It is not an achievable
        # transportation date or supplier commitment.
        target_week = (
            pd.NaT
            if pd.isna(stockout_week)
            else monday(stockout_week)
        )

        confirmed = bool(
            po["included_as_confirmed"]
        )

        baseline_receipts = (
            group["confirmed_existing_receipts"]
            .to_numpy(dtype=float)
            .copy()
        )

        baseline_ending = (
            group["ending_inventory"]
            .to_numpy(dtype=float)
        )

        safety = (
            group["safety_stock"]
            .to_numpy(dtype=float)
        )

        baseline_metrics = calculate_metrics(
            baseline_ending, safety
        )

        scenario_receipts = baseline_receipts.copy()

        scenario_status = "NOT SIMULATED"
        valid_scenario = False

        if pd.isna(target_week):
            scenario_status = "NO STOCKOUT DATE"

        elif target_week < monday(snapshot):
            scenario_status = "TARGET BEFORE SNAPSHOT WEEK"

        elif target_week < first_week:
            scenario_status = "TARGET BEFORE FORECAST"

        elif target_week > last_week:
            scenario_status = "TARGET AFTER FORECAST"

        elif confirmed:
            # A confirmed receipt is already in the
            # baseline. Move it rather than adding it.
            #
            # Existing engine buckets receipts by Monday.
            # An earlier actual receipt date in the same
            # week has no measurable weekly effect.

            effective_original = max(
                original_week, first_week
            )

            if effective_original > last_week:
                scenario_status = (
                    "ORIGINAL CONFIRMED RECEIPT "
                    "OUTSIDE FORECAST"
                )

            elif target_week >= effective_original:
                scenario_status = (
                    "NO EARLIER WEEKLY RECEIPT POSSIBLE"
                )

            else:
                old_positions = np.flatnonzero(
                    weeks == effective_original
                )

                new_positions = np.flatnonzero(
                    weeks == target_week
                )

                old_index = int(old_positions[0])
                new_index = int(new_positions[0])

                if (
                    scenario_receipts[old_index]
                    < quantity - TOL
                ):
                    raise ValueError(
                        f"Confirmed PO receipt cannot "
                        f"be reconciled: {po['po_number']}"
                    )

                scenario_receipts[old_index] -= quantity
                scenario_receipts[new_index] += quantity

                valid_scenario = True
                scenario_status = (
                    "CONFIRMED PO MOVED EARLIER "
                    "- HYPOTHETICAL"
                )

        else:
            # Uncertain POs are excluded from the
            # conservative baseline. Add a conditional
            # receipt without subtracting existing supply.
            target_positions = np.flatnonzero(
                weeks == target_week
            )

            new_index = int(target_positions[0])

            scenario_receipts[new_index] += quantity

            valid_scenario = True
            scenario_status = (
                "UNCERTAIN PO RECEIPT ADDED "
                "- CONDITIONAL"
            )

        if valid_scenario:
            simulated_ending = simulate_inventory(
                group, scenario_receipts
            )

            scenario_metrics = calculate_metrics(
                simulated_ending, safety
            )

        else:
            scenario_metrics = baseline_metrics.copy()

        stockout_weeks_avoided = (
            baseline_metrics["stockout_weeks"]
            - scenario_metrics["stockout_weeks"]
        )

        negative_balance_reduction = (
            baseline_metrics[
                "cumulative_negative_balance_units"
            ]
            - scenario_metrics[
                "cumulative_negative_balance_units"
            ]
        )

        peak_gap_reduction = (
            baseline_metrics[
                "peak_negative_balance_units"
            ]
            - scenario_metrics[
                "peak_negative_balance_units"
            ]
        )

        illustrative_premium = (
            quantity * AIR_PREMIUM_PER_UNIT
            if valid_scenario
            else 0.0
        )

        cost_per_negative_balance_unit_week = (
            illustrative_premium
            / negative_balance_reduction
            if negative_balance_reduction > TOL
            else np.nan
        )

        results.append({
            "po_number": po["po_number"],
            "sku": sku,
            "priority": po["priority"],
            "open_qty": quantity,
            "original_expected_date": po["expected_date"],
            "original_receipt_week": original_week,
            "hypothetical_target_week": target_week,
            "first_stockout_week": stockout_week,
            "confirmed_in_baseline": confirmed,
            "recovery_action": po["recovery_action"],
            "scenario_status": scenario_status,
            "scenario_simulated": valid_scenario,
            "baseline_stockout_weeks":
                baseline_metrics["stockout_weeks"],
            "scenario_stockout_weeks":
                scenario_metrics["stockout_weeks"],
            "stockout_weeks_avoided":
                stockout_weeks_avoided,
            "baseline_cumulative_negative_balance":
                baseline_metrics[
                    "cumulative_negative_balance_units"
                ],
            "scenario_cumulative_negative_balance":
                scenario_metrics[
                    "cumulative_negative_balance_units"
                ],
            "negative_balance_reduction":
                negative_balance_reduction,
            "peak_negative_balance_reduction":
                peak_gap_reduction,
            "baseline_safety_breach_weeks":
                baseline_metrics["safety_breach_weeks"],
            "scenario_safety_breach_weeks":
                scenario_metrics["safety_breach_weeks"],
            "illustrative_freight_premium":
                illustrative_premium,
            "cost_per_negative_balance_unit_week":
                cost_per_negative_balance_unit_week,
            "supplier_confirmation_required": True,
            "scenario_assumption": (
                "Entire open PO quantity arrives in "
                "first stockout week; supplier readiness "
                "and shipment feasibility unverified"
            ),
        })

    return pd.DataFrame(results)


def validate_results(result, recovery):
    check(
        len(result) == len(recovery),
        "One scenario per existing PO line"
    )

    check(
        np.isclose(
            result["open_qty"].sum(),
            recovery["open_qty"].sum()
        ),
        "All original PO quantities preserved"
    )

    check(
        (
            result["stockout_weeks_avoided"] >= 0
        ).all(),
        "No scenario increases stockout weeks"
    )

    check(
        (
            result["negative_balance_reduction"]
            >= -TOL
        ).all(),
        "No scenario increases negative balance exposure"
    )

    check(
        (
            result["illustrative_freight_premium"]
            >= 0
        ).all(),
        "No negative illustrative freight premiums"
    )

    check(
        result.loc[
            ~result["scenario_simulated"],
            "illustrative_freight_premium"
        ].eq(0).all(),
        "Unsimulated scenarios have zero premium"
    )

    check(
        result["scenario_status"].notna().all(),
        "All scenarios classified"
    )


def print_summary(result):
    print("\n" + "=" * 65)
    print("ENHANCEMENT 1D - RECOVERY SIMULATION RESULTS")
    print("=" * 65)

    print(f"PO lines evaluated: {len(result):,}")

    print(
        "Scenarios simulated:",
        int(result["scenario_simulated"].sum())
    )

    print(
        "Scenarios reducing negative inventory exposure:",
        int(
            (
                result["negative_balance_reduction"]
                > TOL
            ).sum()
        )
    )

    print("\nScenario classification:")
    print(
        result["scenario_status"]
        .value_counts()
        .to_string()
    )

    print("\nTop 10 modeled improvements:")

    columns = [
        "po_number",
        "sku",
        "open_qty",
        "scenario_status",
        "stockout_weeks_avoided",
        "negative_balance_reduction",
        "illustrative_freight_premium",
    ]

    top = result.sort_values(
        "negative_balance_reduction",
        ascending=False
    ).head(10)

    print(
        top[columns].to_string(index=False)
    )

    print(
        "\nIMPORTANT: Each PO scenario is evaluated "
        "independently against the same baseline."
    )

    print(
        "Do not add scenario benefits together; "
        "multiple POs for the same SKU may overlap."
    )

    print(
        "The target receipt week is a hypothetical "
        "what-if, not a feasible delivery commitment."
    )

    print(
        "Freight costs are illustrative, "
        "not supplier or carrier quotes."
    )


def main():
    print(
        "\nRunning Enhancement 1D: "
        "Recovery Simulation\n"
    )

    data = load_inputs()

    projection, recovery, freight = prepare_inputs(
        data
    )

    result = simulate_recovery(
        projection, recovery, freight
    )

    validate_results(result, recovery)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d"
    )

    print(f"\nSaved: {OUTPUT_FILE}")

    print_summary(result)


if __name__ == "__main__":
    main()
