
"""
Enhancement 1J - Portfolio Recovery Optimizer

Read-only, synthetic procurement recovery analysis.

Evaluates existing PO recovery actions together against the
validated 26-week feasible inventory projection.

Confirmed POs:
    Move receipt from original week to hypothetical target week.

Uncertain POs:
    Add receipt conditionally at hypothetical target week.

This is an illustrative scenario optimizer, not an executable
procurement or freight recommendation.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

PROJECTION_FILE = DATA / "feasible_inventory_projection.csv"
SIMULATION_FILE = DATA / "recovery_simulation_results.csv"

OUTPUT_ACTIONS = DATA / "portfolio_recovery_actions.csv"
OUTPUT_SUMMARY = DATA / "portfolio_recovery_summary.csv"
OUTPUT_PROJECTION = DATA / "portfolio_recovery_projection.csv"

TOLERANCE = 1e-6


def load_inputs():
    projection = pd.read_csv(PROJECTION_FILE)
    candidates = pd.read_csv(SIMULATION_FILE)

    projection["forecast_week"] = pd.to_datetime(
        projection["forecast_week"]
    )

    for column in [
        "original_receipt_week",
        "hypothetical_target_week",
    ]:
        candidates[column] = pd.to_datetime(
            candidates[column],
            errors="coerce",
        )

    return projection, candidates


def validate_inputs(projection, candidates):
    required_projection = {
        "sku",
        "forecast_week",
        "beginning_inventory",
        "confirmed_existing_receipts",
        "feasible_new_receipts",
        "forecast_demand",
        "ending_inventory",
        "safety_stock",
    }

    required_candidates = {
        "po_number",
        "sku",
        "open_qty",
        "confirmed_in_baseline",
        "scenario_simulated",
        "original_receipt_week",
        "hypothetical_target_week",
        "illustrative_freight_premium",
        "supplier_confirmation_required",
    }

    if not required_projection.issubset(projection.columns):
        raise ValueError("Projection is missing required columns")

    if not required_candidates.issubset(candidates.columns):
        raise ValueError("Recovery simulation is missing columns")

    if len(projection) != 5200:
        raise ValueError("Expected 5,200 projection records")

    if projection["sku"].nunique() != 200:
        raise ValueError("Expected 200 SKUs")

    if not projection.groupby("sku").size().eq(26).all():
        raise ValueError("Expected 26 weeks per SKU")

    if projection.duplicated(["sku", "forecast_week"]).any():
        raise ValueError("Duplicate SKU/week projection records")

    if candidates["po_number"].duplicated().any():
        raise ValueError("Duplicate PO numbers")

    if not set(candidates["sku"]).issubset(
        set(projection["sku"])
    ):
        raise ValueError("Recovery candidate has unknown SKU")

    if candidates["open_qty"].isna().any():
        raise ValueError("Missing PO quantities")

    if (candidates["open_qty"] <= 0).any():
        raise ValueError("Nonpositive PO quantity")

    if candidates[
        "illustrative_freight_premium"
    ].isna().any():
        raise ValueError("Missing freight premium")

    if (
        candidates["illustrative_freight_premium"] < 0
    ).any():
        raise ValueError("Negative freight premium")


def inventory_metrics(ending, safety):
    ending = np.asarray(ending, dtype=float)
    safety = np.asarray(safety, dtype=float)

    negative = np.maximum(-ending, 0)

    return {
        "stockout_weeks": int(np.sum(ending < -TOLERANCE)),
        "cumulative_negative_balance": float(
            negative.sum()
        ),
        "peak_negative_balance": float(
            negative.max()
        ),
        "safety_breach_weeks": int(
            np.sum(ending < safety - TOLERANCE)
        ),
    }


def simulate_sku(base, actions):
    """
    Simulate a single SKU using combined recovery actions.

    Every action is applied once to the same receipt schedule.
    """

    base = base.sort_values("forecast_week").copy()

    weeks = pd.DatetimeIndex(base["forecast_week"])

    existing = base[
        "confirmed_existing_receipts"
    ].to_numpy(dtype=float).copy()

    new = base[
        "feasible_new_receipts"
    ].to_numpy(dtype=float)

    demand = base[
        "forecast_demand"
    ].to_numpy(dtype=float)

    safety = base[
        "safety_stock"
    ].to_numpy(dtype=float)

    week_positions = {
        week: i for i, week in enumerate(weeks)
    }

    for action in actions:
        qty = float(action["open_qty"])

        target = pd.Timestamp(
            action["hypothetical_target_week"]
        )

        if target not in week_positions:
            raise ValueError(
                f"Target week outside horizon: {target}"
            )

        target_index = week_positions[target]

        if bool(action["confirmed_in_baseline"]):
            original = pd.Timestamp(
                action["original_receipt_week"]
            )

            if original not in week_positions:
                raise ValueError(
                    f"Original receipt outside horizon: {original}"
                )

            original_index = week_positions[original]

            existing[original_index] -= qty

            if existing[original_index] < -TOLERANCE:
                raise ValueError(
                    "Confirmed PO movement exceeds baseline "
                    f"receipts for {action['sku']}"
                )

            existing[target_index] += qty

        else:
            # Conditional supplier-confirmed receipt scenario.
            # This is not treated as actual confirmed supply.
            existing[target_index] += qty

    beginning = np.zeros(len(base))
    ending = np.zeros(len(base))

    beginning[0] = float(
        base.iloc[0]["beginning_inventory"]
    )

    for i in range(len(base)):
        if i > 0:
            beginning[i] = ending[i - 1]

        ending[i] = (
            beginning[i]
            + existing[i]
            + new[i]
            - demand[i]
        )

    result = base[
        ["sku", "forecast_week"]
    ].copy()

    result["scenario_beginning_inventory"] = beginning
    result["scenario_existing_receipts"] = existing
    result["scenario_new_receipts"] = new
    result["forecast_demand"] = demand
    result["scenario_ending_inventory"] = ending
    result["safety_stock"] = safety

    return result


def optimize_sku(base, candidates):
    """
    Greedy selection based on marginal shortage improvement
    per illustrative freight dollar.

    Selection is a scenario ranking, not proof of feasibility.
    """

    baseline = simulate_sku(base, [])

    baseline_metrics = inventory_metrics(
        baseline["scenario_ending_inventory"],
        baseline["safety_stock"],
    )

    selected = []
    remaining = candidates.to_dict("records")
    selected_records = []

    current = baseline
    current_metrics = baseline_metrics

    while remaining:
        evaluations = []

        for candidate in remaining:
            scenario = simulate_sku(
                base,
                selected + [candidate],
            )

            metrics = inventory_metrics(
                scenario["scenario_ending_inventory"],
                scenario["safety_stock"],
            )

            improvement = (
                current_metrics[
                    "cumulative_negative_balance"
                ]
                - metrics[
                    "cumulative_negative_balance"
                ]
            )

            improvement = max(0.0, improvement)

            premium = float(
                candidate["illustrative_freight_premium"]
            )

            efficiency = (
                improvement / premium
                if premium > TOLERANCE
                else (
                    float("inf")
                    if improvement > TOLERANCE
                    else 0.0
                )
            )

            evaluations.append(
                {
                    "candidate": candidate,
                    "scenario": scenario,
                    "metrics": metrics,
                    "improvement": improvement,
                    "efficiency": efficiency,
                    "premium": premium,
                }
            )

        evaluations.sort(
            key=lambda item: (
                item["efficiency"],
                item["improvement"],
                -item["premium"],
                str(item["candidate"]["po_number"]),
            ),
            reverse=True,
        )

        best = evaluations[0]

        if best["improvement"] <= TOLERANCE:
            break

        candidate = best["candidate"]

        selected.append(candidate)

        selected_records.append(
            {
                "po_number": candidate["po_number"],
                "sku": candidate["sku"],
                "open_qty": candidate["open_qty"],
                "confirmed_in_baseline": candidate[
                    "confirmed_in_baseline"
                ],
                "supplier_confirmation_required": candidate[
                    "supplier_confirmation_required"
                ],
                "original_receipt_week": candidate[
                    "original_receipt_week"
                ],
                "hypothetical_target_week": candidate[
                    "hypothetical_target_week"
                ],
                "selection_order": len(selected),
                "incremental_negative_balance_reduction": (
                    best["improvement"]
                ),
                "incremental_stockout_weeks_avoided": (
                    current_metrics["stockout_weeks"]
                    - best["metrics"]["stockout_weeks"]
                ),
                "illustrative_freight_premium": (
                    best["premium"]
                ),
                "incremental_benefit_per_dollar": (
                    best["efficiency"]
                ),
                "selection_status": (
                    "SELECTED - CONDITIONAL SCENARIO"
                ),
            }
        )

        current = best["scenario"]
        current_metrics = best["metrics"]

        remaining = [
            item
            for item in remaining
            if item["po_number"] != candidate["po_number"]
        ]

    return {
        "baseline": baseline,
        "optimized": current,
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": current_metrics,
        "selected_records": selected_records,
        "selected_actions": selected,
    }


def main():
    print("PORTFOLIO RECOVERY OPTIMIZER")
    print("Mode: Synthetic, read-only, conditional\n")

    projection, candidates = load_inputs()

    validate_inputs(projection, candidates)

    eligible = candidates[
        candidates["scenario_simulated"] == True
    ].copy()

    all_actions = []
    all_summaries = []
    all_projections = []

    for sku, base in projection.groupby("sku"):
        sku_candidates = eligible[
            eligible["sku"] == sku
        ]

        result = optimize_sku(
            base,
            sku_candidates,
        )

        all_actions.extend(
            result["selected_records"]
        )

        baseline_metrics = result["baseline_metrics"]
        optimized_metrics = result["optimized_metrics"]

        selected = result["selected_actions"]

        premium = sum(
            float(action["illustrative_freight_premium"])
            for action in selected
        )

        all_summaries.append(
            {
                "sku": sku,
                "eligible_po_actions": len(sku_candidates),
                "selected_po_actions": len(selected),
                "baseline_stockout_weeks": (
                    baseline_metrics["stockout_weeks"]
                ),
                "optimized_stockout_weeks": (
                    optimized_metrics["stockout_weeks"]
                ),
                "stockout_weeks_avoided": (
                    baseline_metrics["stockout_weeks"]
                    - optimized_metrics["stockout_weeks"]
                ),
                "baseline_cumulative_negative_balance": (
                    baseline_metrics[
                        "cumulative_negative_balance"
                    ]
                ),
                "optimized_cumulative_negative_balance": (
                    optimized_metrics[
                        "cumulative_negative_balance"
                    ]
                ),
                "negative_balance_reduction": (
                    baseline_metrics[
                        "cumulative_negative_balance"
                    ]
                    - optimized_metrics[
                        "cumulative_negative_balance"
                    ]
                ),
                "baseline_safety_breach_weeks": (
                    baseline_metrics["safety_breach_weeks"]
                ),
                "optimized_safety_breach_weeks": (
                    optimized_metrics["safety_breach_weeks"]
                ),
                "illustrative_total_freight_premium": (
                    premium
                ),
                "supplier_confirmation_required": any(
                    bool(action[
                        "supplier_confirmation_required"
                    ])
                    for action in selected
                ),
                "scenario_limitation": (
                    "Conditional synthetic recovery scenario. "
                    "Supplier readiness, shipment feasibility, "
                    "and actual freight rates are unverified."
                ),
            }
        )

        optimized = result["optimized"].copy()

        optimized["baseline_ending_inventory"] = (
            result["baseline"][
                "scenario_ending_inventory"
            ].to_numpy()
        )

        all_projections.append(optimized)

    actions = pd.DataFrame(all_actions)
    summary = pd.DataFrame(all_summaries)
    combined_projection = pd.concat(
        all_projections,
        ignore_index=True,
    )

    # Output validations
    checks = []

    def check(label, condition):
        passed = bool(condition)
        checks.append(passed)
        print(
            f"[{'PASS' if passed else 'FAIL'}] {label}"
        )

    check("200 SKU summaries", len(summary) == 200)
    check("Unique summary SKUs", summary["sku"].is_unique)

    check(
        "5,200 projection records",
        len(combined_projection) == 5200,
    )

    check(
        "91 input PO lines",
        len(candidates) == 91,
    )

    check(
        "80 eligible PO scenarios",
        len(eligible) == 80,
    )

    check(
        "No duplicate selected POs",
        actions.empty
        or actions["po_number"].is_unique,
    )

    check(
        "Selected actions do not exceed eligibility",
        len(actions) <= len(eligible),
    )

    check(
        "No negative selected improvements",
        actions.empty
        or (
            actions[
                "incremental_negative_balance_reduction"
            ] >= -TOLERANCE
        ).all(),
    )

    check(
        "No negative scenario receipt quantities",
        (
            combined_projection[
                "scenario_existing_receipts"
            ] >= -TOLERANCE
        ).all(),
    )

    check(
        "No worse cumulative negative balance",
        (
            summary[
                "negative_balance_reduction"
            ] >= -TOLERANCE
        ).all(),
    )

    check(
        "No increased total stockout weeks",
        summary["stockout_weeks_avoided"].sum() >= 0,
    )

    check(
        "Nonnegative freight premium",
        (
            summary[
                "illustrative_total_freight_premium"
            ] >= 0
        ).all(),
    )

    passed = sum(checks)
    failed = len(checks) - passed

    print("\nVALIDATION SUMMARY")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")

    if failed:
        raise AssertionError(
            "Optimizer validation failed. No outputs saved."
        )

    actions.to_csv(OUTPUT_ACTIONS, index=False)
    summary.to_csv(OUTPUT_SUMMARY, index=False)
    combined_projection.to_csv(
        OUTPUT_PROJECTION,
        index=False,
    )

    print("\nOUTPUTS")
    print(f"Actions: {OUTPUT_ACTIONS}")
    print(f"Summary: {OUTPUT_SUMMARY}")
    print(f"Projection: {OUTPUT_PROJECTION}")

    print("\nPORTFOLIO RESULTS")
    print(f"Selected actions: {len(actions)}")

    print(
        "Stockout weeks avoided:",
        int(summary["stockout_weeks_avoided"].sum()),
    )

    print(
        "Negative balance reduction:",
        round(
            summary[
                "negative_balance_reduction"
            ].sum(),
            2,
        ),
    )

    print(
        "Illustrative freight premium: $",
        round(
            summary[
                "illustrative_total_freight_premium"
            ].sum(),
            2,
        ),
    )


if __name__ == "__main__":
    main()
