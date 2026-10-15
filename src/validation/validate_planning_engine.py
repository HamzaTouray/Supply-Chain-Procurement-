
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PHASE 9 - PLANNING ENGINE VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

results = []


def report(name, status, details):
    results.append({
        "Test": name,
        "Status": status,
        "Details": str(details),
    })

    print(f"\n[{status}] {name}")
    print(f"  {details}")


def check(name, condition, details):
    report(
        name,
        "PASS" if condition else "FAIL",
        details,
    )


def review(name, details):
    report(name, "REVIEW", details)


def load(filename):
    path = DATA / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {path}"
        )

    df = pd.read_csv(path)

    print(f"Loaded {filename}: {len(df):,} rows")

    return df


def numeric(df, columns):
    for col in columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )


def boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def close_enough(actual, expected, tolerance=0.01):
    return np.isclose(
        actual,
        expected,
        rtol=0,
        atol=tolerance,
        equal_nan=False,
    )


def main():
    print("=" * 65)
    print("PHASE 9 - PROCUREMENT PLANNING ENGINE VALIDATION")
    print("=" * 65)

    master = load("planning_master.csv")
    policy = load("inventory_policy.csv")
    timing = load("order_timing_recommendations.csv")
    actions = load("feasible_supply_actions.csv")
    projection = load("feasible_inventory_projection.csv")

    # --------------------------------------------------------
    # 1. DATA INTEGRITY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SECTION 1 - DATA INTEGRITY")
    print("=" * 65)

    for name, df in [
        ("Planning Master", master),
        ("Inventory Policy", policy),
        ("Order Timing", timing),
        ("Feasible Actions", actions),
        ("Inventory Projection", projection),
    ]:
        check(
            f"{name}: no duplicate rows",
            not df.duplicated().any(),
            f"Duplicate rows: {df.duplicated().sum()}",
        )

    for name, df in [
        ("Planning Master", master),
        ("Inventory Policy", policy),
        ("Order Timing", timing),
    ]:
        check(
            f"{name}: one row per SKU",
            df["sku"].is_unique,
            f"Rows: {len(df)}, SKUs: {df['sku'].nunique()}",
        )

    master_skus = set(master["sku"])

    for name, df in [
        ("Inventory Policy", policy),
        ("Order Timing", timing),
        ("Inventory Projection", projection),
    ]:
        check(
            f"{name}: SKU coverage",
            set(df["sku"]) == master_skus,
            f"Unique SKUs: {df['sku'].nunique()}",
        )

    check(
        "Feasible Actions: valid SKU references",
        set(actions["sku"]).issubset(master_skus),
        f"Action SKUs: {actions['sku'].nunique()}",
    )

    check(
        "Inventory Projection: unique SKU/week",
        not projection.duplicated(
            ["sku", "forecast_week"]
        ).any(),
        "Each SKU/week must appear only once.",
    )

    weeks_per_sku = projection.groupby(
        "sku"
    )["forecast_week"].nunique()

    check(
        "Inventory Projection: 26 weeks per SKU",
        weeks_per_sku.eq(26).all(),
        f"Minimum: {weeks_per_sku.min()}, "
        f"Maximum: {weeks_per_sku.max()}",
    )

    # --------------------------------------------------------
    # 2. INVENTORY RECONCILIATION
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SECTION 2 - INVENTORY RECONCILIATION")
    print("=" * 65)

    numeric(
        master,
        [
            "on_hand",
            "allocated",
            "available",
            "backordered",
            "net_available",
            "total_open_po_qty",
            "inventory_position",
            "unit_cost",
            "inventory_value",
        ],
    )

    check(
        "Available inventory = On hand - Allocated",
        close_enough(
            master["available"],
            master["on_hand"] - master["allocated"],
        ).all(),
        "Checking all 200 SKUs.",
    )

    # Net available may follow a business-specific
    # backorder convention, so identify the actual rule.

    net_available_rule_1 = close_enough(
        master["net_available"],
        master["available"],
    )

    net_available_rule_2 = close_enough(
        master["net_available"],
        master["available"] - master["backordered"],
    )

    if net_available_rule_1.all():
        report(
            "Net available reconciliation",
            "PASS",
            "Net available = Available",
        )
    elif net_available_rule_2.all():
        report(
            "Net available reconciliation",
            "PASS",
            "Net available = Available - Backordered",
        )
    else:
        review(
            "Net available reconciliation",
            "Neither common definition matches every SKU. "
            "Review the source formula.",
        )

    inventory_position_expected = (
        master["net_available"]
        + master["total_open_po_qty"]
    )

    check(
        "Inventory position reconciliation",
        close_enough(
            master["inventory_position"],
            inventory_position_expected,
        ).all(),
        "Inventory position = Net available + Open PO units.",
    )

    # Inventory valuation conventions can differ.
    valuation_on_hand = close_enough(
        master["inventory_value"],
        master["on_hand"] * master["unit_cost"],
    )

    valuation_available = close_enough(
        master["inventory_value"],
        master["available"] * master["unit_cost"],
    )

    if valuation_on_hand.all():
        report(
            "Inventory valuation",
            "PASS",
            "Inventory value = On hand x Unit cost",
        )
    elif valuation_available.all():
        report(
            "Inventory valuation",
            "PASS",
            "Inventory value = Available x Unit cost",
        )
    else:
        review(
            "Inventory valuation",
            "Neither common valuation rule matches all SKUs.",
        )

    # --------------------------------------------------------
    # 3. INVENTORY POLICY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SECTION 3 - INVENTORY POLICY")
    print("=" * 65)

    numeric(
        policy,
        [
            "safety_stock",
            "reorder_point",
            "forecast_lead_time_demand",
            "planning_weekly_demand",
            "planning_lead_time_weeks",
            "moq",
            "case_pack",
            "net_available",
            "total_open_po_qty",
            "policy_inventory_position",
        ],
    )

    check(
        "Safety stock is nonnegative",
        policy["safety_stock"].ge(0).all(),
        f"Negative values: "
        f"{policy['safety_stock'].lt(0).sum()}",
    )

    check(
        "Reorder point is nonnegative",
        policy["reorder_point"].ge(0).all(),
        f"Negative values: "
        f"{policy['reorder_point'].lt(0).sum()}",
    )

    check(
        "Reorder point covers safety stock",
        policy["reorder_point"]
        .ge(policy["safety_stock"]).all(),
        "Checking all SKU policies.",
    )

    check(
        "Planning lead time is positive",
        policy["planning_lead_time_weeks"].gt(0).all(),
        "Checking all supplier lead times.",
    )

    check(
        "MOQ is positive",
        policy["moq"].gt(0).all(),
        "Checking all minimum order quantities.",
    )

    check(
        "Case pack is positive",
        policy["case_pack"].gt(0).all(),
        "Checking all case pack quantities.",
    )

    check(
        "Policy inventory position reconciliation",
        close_enough(
            policy["policy_inventory_position"],
            policy["net_available"]
            + policy["total_open_po_qty"],
        ).all(),
        "Policy position = Net available + Open POs.",
    )

    policy_rop_expected = (
        policy["forecast_lead_time_demand"]
        + policy["safety_stock"]
    )

    rop_matches = close_enough(
        policy["reorder_point"],
        policy_rop_expected,
        tolerance=1.01,
    )

    if rop_matches.all():
        report(
            "Reorder point formula",
            "PASS",
            "Reorder point approximately equals "
            "lead-time demand + safety stock.",
        )
    else:
        review(
            "Reorder point formula",
            f"{(~rop_matches).sum()} SKUs differ from "
            "lead-time demand + safety stock. "
            "Check rounding and policy floors.",
        )

    # --------------------------------------------------------
    # 4. PURCHASE ORDER RECOMMENDATIONS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SECTION 4 - PURCHASE ORDER RECOMMENDATIONS")
    print("=" * 65)

    numeric(
        actions,
        [
            "recommended_order_qty",
            "recommended_order_value",
            "unit_cost",
            "moq",
            "case_pack",
            "receipt_delay_days",
            "planning_lead_time_days",
        ],
    )

    check(
        "Recommended order quantities are positive",
        actions["recommended_order_qty"].gt(0).all(),
        "Checking all purchasing proposal lines.",
    )

    check(
        "Recommended purchasing values are nonnegative",
        actions["recommended_order_value"].ge(0).all(),
        "Checking all purchasing proposal lines.",
    )

    expected_values = (
        actions["recommended_order_qty"]
        * actions["unit_cost"]
    )

    check(
        "Order value = Quantity x Unit cost",
        close_enough(
            actions["recommended_order_value"],
            expected_values,
        ).all(),
        "Checking all 993 purchasing proposals.",
    )

    check(
        "Recommended quantity meets MOQ",
        actions["recommended_order_qty"]
        .ge(actions["moq"]).all(),
        "Checking all purchasing proposals.",
    )

    case_pack = actions["case_pack"]

    pack_remainder = (
        actions["recommended_order_qty"] % case_pack
    )

    pack_valid = (
        case_pack.gt(0)
        & np.isclose(pack_remainder, 0, atol=0.000001)
    )

    check(
        "Recommended quantity respects case pack",
        pack_valid.all(),
        f"Noncompliant lines: {(~pack_valid).sum()}",
    )

    # --------------------------------------------------------
    # 5. DELIVERY FEASIBILITY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SECTION 5 - DELIVERY FEASIBILITY")
    print("=" * 65)

    date_columns = [
        "required_receipt_week",
        "planned_release_date",
        "original_release_date",
        "executable_release_date",
        "earliest_feasible_receipt",
    ]

    for col in date_columns:
        actions[col] = pd.to_datetime(
            actions[col],
            errors="coerce",
        )

    check(
        "Purchasing dates are valid",
        actions[date_columns].notna().all().all(),
        "Checking required and executable dates.",
    )

    check(
        "Feasible receipt is not before release",
        (
            actions["earliest_feasible_receipt"]
            >= actions["executable_release_date"]
        ).all(),
        "Checking release-to-receipt chronology.",
    )

    check(
        "Planning lead-time days are nonnegative",
        actions["planning_lead_time_days"].ge(0).all(),
        "Checking all purchasing proposals.",
    )

    computed_delay = (
        actions["earliest_feasible_receipt"]
        - actions["required_receipt_week"]
    ).dt.days.clip(lower=0)

    delay_matches = close_enough(
        actions["receipt_delay_days"],
        computed_delay,
        tolerance=1,
    )

    if delay_matches.all():
        report(
            "Receipt delay calculation",
            "PASS",
            "Receipt delay agrees with required "
            "receipt and feasible receipt dates.",
        )
    else:
        review(
            "Receipt delay calculation",
            f"{(~delay_matches).sum()} lines differ. "
            "Review weekly date conventions.",
        )

    feasible_flag = boolean(
        actions["receipt_feasible_on_time"]
    )

    expected_feasible = (
        actions["earliest_feasible_receipt"]
        <= actions["required_receipt_week"]
    )

    check(
        "On-time feasibility flag matches dates",
        feasible_flag.eq(expected_feasible).all(),
        "Checking all purchasing proposals.",
    )

    # --------------------------------------------------------
    # 6. PROJECTED INVENTORY BALANCES
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SECTION 6 - PROJECTED INVENTORY BALANCES")
    print("=" * 65)

    numeric(
        projection,
        [
            "beginning_inventory",
            "confirmed_existing_receipts",
            "feasible_new_receipts",
            "forecast_demand",
            "ending_inventory",
            "safety_stock",
        ],
    )

    expected_ending = (
        projection["beginning_inventory"]
        + projection["confirmed_existing_receipts"]
        + projection["feasible_new_receipts"]
        - projection["forecast_demand"]
    )

    balance_matches = close_enough(
        projection["ending_inventory"],
        expected_ending,
        tolerance=0.02,
    )

    check(
        "Projected inventory balance equation",
        balance_matches.all(),
        f"Rows checked: {len(projection):,}. "
        f"Mismatches: {(~balance_matches).sum():,}",
    )

    projection["forecast_week"] = pd.to_datetime(
        projection["forecast_week"],
        errors="coerce",
    )

    ordered = projection.sort_values(
        ["sku", "forecast_week"]
    ).copy()

    previous_ending = ordered.groupby(
        "sku"
    )["ending_inventory"].shift(1)

    continuation = previous_ending.notna()

    continuity_matches = close_enough(
        ordered.loc[continuation, "beginning_inventory"],
        previous_ending.loc[continuation],
        tolerance=0.02,
    )

    check(
        "Weekly inventory continuity",
        continuity_matches.all(),
        f"Transitions checked: {continuation.sum():,}. "
        f"Mismatches: {(~continuity_matches).sum():,}",
    )

    expected_stockout = (
        projection["ending_inventory"] < 0
    )

    actual_stockout = boolean(
        projection["stockout"]
    )

    if actual_stockout.eq(expected_stockout).all():
        report(
            "Stockout flag calculation",
            "PASS",
            "Stockout = Ending inventory below zero.",
        )
    else:
        review(
            "Stockout flag calculation",
            "Stockout flag differs from ending inventory "
            "below zero. Check whether zero inventory "
            "also counts as a stockout.",
        )

    expected_safety_breach = (
        projection["ending_inventory"]
        < projection["safety_stock"]
    )

    actual_safety_breach = boolean(
        projection["below_safety_stock"]
    )

    check(
        "Safety stock breach flag",
        actual_safety_breach.eq(
            expected_safety_breach
        ).all(),
        "Checking all projected SKU/weeks.",
    )

    # --------------------------------------------------------
    # 7. REVIEW FLAGS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("SECTION 7 - MANUAL REVIEW")
    print("=" * 65)

    no_action_skus = (
        master_skus - set(actions["sku"])
    )

    review(
        "SKUs without purchasing proposals",
        f"{len(no_action_skus)} SKUs: "
        f"{', '.join(map(str, sorted(no_action_skus)))}",
    )

    missing_wape = policy[
        policy["selected_wape"].isna()
    ]["sku"].tolist()

    if missing_wape:
        review(
            "Missing forecast accuracy",
            f"SKUs: {missing_wape}",
        )
    else:
        report(
            "Forecast accuracy completeness",
            "PASS",
            "All SKUs have a selected WAPE.",
        )

    missing_timing_dates = timing[
        timing["required_receipt_date"].isna()
    ]

    review(
        "Missing order timing dates",
        f"{len(missing_timing_dates)} SKUs. "
        "Verify that these are no-order or "
        "non-actionable recommendations.",
    )

    # --------------------------------------------------------
    # 8. FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 65)

    summary = pd.DataFrame(results)

    counts = summary["Status"].value_counts()

    passed = int(counts.get("PASS", 0))
    failed = int(counts.get("FAIL", 0))
    reviews = int(counts.get("REVIEW", 0))

    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"REVIEW: {reviews}")
    print(f"TOTAL TESTS: {len(summary)}")

    print("\nPROCUREMENT TOTALS")
    print(
        "Recommended Order Lines:",
        f"{len(actions):,}",
    )
    print(
        "Recommended Units:",
        f"{actions['recommended_order_qty'].sum():,.0f}",
    )
    print(
        "Recommended Value:",
        f"${actions['recommended_order_value'].sum():,.2f}",
    )

    output_dir = ROOT / "data" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir / "planning_validation_results.csv"
    )

    summary.to_csv(output_file, index=False)

    print("\nValidation report saved:")
    print(output_file)

    if failed:
        print(
            "\nRESULT: VALIDATION FAILED. "
            "Investigate failed checks before proceeding."
        )
    elif reviews:
        print(
            "\nRESULT: STRUCTURAL CHECKS PASSED, "
            "BUT MANUAL REVIEWS REMAIN."
        )
    else:
        print(
            "\nRESULT: ALL IMPLEMENTED CHECKS PASSED."
        )


if __name__ == "__main__":
    main()
