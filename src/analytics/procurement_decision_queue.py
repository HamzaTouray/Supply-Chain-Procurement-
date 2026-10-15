
from pathlib import Path
import numpy as np
import pandas as pd

# Enhancement 1I: Read-only procurement decision prioritization
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
SNAPSHOT = pd.Timestamp("2026-09-30")

INPUTS = {
    "bridge": "bridge_quantity_analysis.csv",
    "recovery": "existing_po_recovery_summary.csv",
    "freight": "freight_recovery_scenarios.csv",
    "forecast": "forecast_risk_analysis.csv",
}

OUTPUT = DATA / "procurement_decision_queue.csv"

RECOVERY_STATUSES = {
    "SUPPLIER CONFIRMATION REQUIRED",
    "NO EXISTING PO - OTHER RECOVERY NEEDED",
    "CONFIRMED PO TIMING REVIEW",
    "MONITOR CONFIRMED EXISTING PO",
}

BRIDGE_STATUSES = {
    "STOCKOUT BRIDGE REQUIRED",
    "SAFETY STOCK BRIDGE ONLY",
    "NO PRE-RECEIPT BRIDGE",
}


def load_data():
    frames = {}

    for key, filename in INPUTS.items():
        path = DATA / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

        df = pd.read_csv(path)

        if "sku" not in df.columns:
            raise ValueError(f"{filename} has no SKU column")

        if df["sku"].isna().any():
            raise ValueError(f"{filename} contains missing SKUs")

        if df["sku"].duplicated().any():
            raise ValueError(f"{filename} contains duplicate SKUs")

        frames[key] = df

    return frames


def require_columns(df, name, columns):
    missing = sorted(set(columns) - set(df.columns))

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}"
        )


def classify_decision(row):
    bridge = row["bridge_status"]
    recovery = row["recovery_status"]

    if bridge == "NO PRE-RECEIPT BRIDGE":
        return (
            "P4",
            "MONITOR",
            "Monitor inventory and forecast",
            "No pre-receipt stockout or safety-stock bridge."
        )

    if bridge == "SAFETY STOCK BRIDGE ONLY":
        return (
            "P3",
            "PREVENTIVE",
            "Review safety stock and replenishment coverage",
            "Safety stock is exposed, but no pre-receipt "
            "stockout is projected."
        )

    if recovery == "SUPPLIER CONFIRMATION REQUIRED":
        return (
            "P1",
            "SUPPLIER CONFIRMATION",
            "Confirm existing PO quantity and delivery date",
            "Existing PO supply is uncertain and requires "
            "supplier confirmation."
        )

    if recovery == "CONFIRMED PO TIMING REVIEW":
        return (
            "P1",
            "EARLIER DELIVERY REVIEW",
            "Request earlier delivery of confirmed PO",
            "Confirmed supply exists, but its timing needs review."
        )

    if recovery == "NO EXISTING PO - OTHER RECOVERY NEEDED":
        if row["new_order_air_before_stockout"]:
            return (
                "P2",
                "CONDITIONAL AIR FREIGHT",
                "Verify supplier readiness and compare freight",
                "A hypothetical new air order may arrive before "
                "the first projected stockout."
            )

        return (
            "P1",
            "ALTERNATIVE SUPPLY",
            "Investigate alternate supply or feasible recovery",
            "No existing PO is available and a new air order "
            "is not projected to arrive before the shortage."
        )

    if recovery == "MONITOR CONFIRMED EXISTING PO":
        return (
            "P2",
            "CONFIRMED PO COVERAGE REVIEW",
            "Verify confirmed PO coverage against demand",
            "Confirmed supply is present, but the projected "
            "stockout still requires attention."
        )

    raise ValueError(
        f"Unexpected recovery classification for {row['sku']}: "
        f"{recovery}"
    )


def build_queue(frames):
    bridge = frames["bridge"].copy()
    recovery = frames["recovery"].copy()
    freight = frames["freight"].copy()
    forecast = frames["forecast"].copy()

    require_columns(
        bridge, "bridge",
        [
            "sku", "bridge_status", "supplier_id", "priority",
            "minimum_stockout_bridge_units",
            "minimum_safety_bridge_units",
            "first_pre_receipt_stockout_week",
        ]
    )

    require_columns(
        recovery, "recovery",
        [
            "sku", "recovery_status", "open_po_lines",
            "open_po_units", "confirmed_po_units",
            "uncertain_po_units",
            "lines_needing_confirmation",
            "lines_for_earlier_delivery_review",
        ]
    )

    require_columns(
        freight, "freight",
        [
            "sku", "snapshot_date",
            "new_order_air_before_stockout",
            "scenario_status",
            "incremental_air_freight_cost",
        ]
    )

    require_columns(
        forecast, "forecast",
        [
            "sku", "risk_tier", "risk_score",
            "selected_wape", "material_underforecast",
            "currently_stocked_out",
            "unit_cost", "planning_weekly_demand",
        ]
    )

    if len(bridge) != 200:
        raise ValueError("Expected 200 SKUs in bridge analysis")

    if len(freight) != 200 or len(forecast) != 200:
        raise ValueError("Expected 200 SKUs in freight and forecast")

    if not set(recovery["sku"]).issubset(set(bridge["sku"])):
        raise ValueError("Recovery contains unknown SKUs")

    if set(bridge["sku"]) != set(freight["sku"]):
        raise ValueError("Bridge/freight SKU mismatch")

    if set(bridge["sku"]) != set(forecast["sku"]):
        raise ValueError("Bridge/forecast SKU mismatch")

    if not set(bridge["bridge_status"]).issubset(BRIDGE_STATUSES):
        raise ValueError("Unexpected bridge status")

    if not set(recovery["recovery_status"]).issubset(
        RECOVERY_STATUSES
    ):
        raise ValueError("Unexpected recovery status")

    bridge = bridge.rename(
        columns={"priority": "planning_priority"}
    )

    queue = bridge.merge(
        recovery[
            [
                "sku", "recovery_status", "open_po_lines",
                "open_po_units", "confirmed_po_units",
                "uncertain_po_units",
                "lines_needing_confirmation",
                "lines_for_earlier_delivery_review",
            ]
        ],
        on="sku",
        how="left",
        validate="one_to_one",
    )

    queue = queue.merge(
        freight[
            [
                "sku", "snapshot_date",
                "new_order_air_before_stockout",
                "scenario_status",
                "incremental_air_freight_cost",
            ]
        ],
        on="sku",
        how="left",
        validate="one_to_one",
    )

    queue = queue.merge(
        forecast[
            [
                "sku", "risk_tier", "risk_score",
                "selected_wape", "material_underforecast",
                "currently_stocked_out", "unit_cost",
                "planning_weekly_demand",
            ]
        ],
        on="sku",
        how="left",
        validate="one_to_one",
    )

    for col in [
        "open_po_lines", "open_po_units",
        "confirmed_po_units", "uncertain_po_units",
        "lines_needing_confirmation",
        "lines_for_earlier_delivery_review",
    ]:
        queue[col] = queue[col].fillna(0).astype(int)

    queue["recovery_status"] = queue[
        "recovery_status"
    ].fillna("NOT REQUIRED")

    queue["new_order_air_before_stockout"] = queue[
        "new_order_air_before_stockout"
    ].fillna(False).astype(bool)

    queue["first_stockout_date"] = pd.to_datetime(
        queue["first_pre_receipt_stockout_week"],
        errors="coerce",
    )

    queue["snapshot_date"] = pd.to_datetime(
        queue["snapshot_date"],
        errors="coerce",
    )

    if queue["snapshot_date"].isna().any():
        raise ValueError("Missing or invalid snapshot dates")

    if not (queue["snapshot_date"] == SNAPSHOT).all():
        raise ValueError("Unexpected planning snapshot date")

    queue["weeks_to_first_stockout"] = (
        (queue["first_stockout_date"] - SNAPSHOT).dt.days / 7
    )

    queue["weeks_to_first_stockout"] = (
        queue["weeks_to_first_stockout"].round(2)
    )

    decisions = queue.apply(classify_decision, axis=1)

    queue[
        [
            "decision_priority", "decision_category",
            "recommended_action", "decision_reason",
        ]
    ] = pd.DataFrame(
        decisions.tolist(),
        index=queue.index,
    )

    queue["forecast_warning"] = np.select(
        [
            queue["risk_tier"].eq("CRITICAL"),
            queue["risk_tier"].eq("HIGH"),
            queue["material_underforecast"].eq(True),
        ],
        [
            "CRITICAL FORECAST RISK",
            "HIGH FORECAST RISK",
            "UNDERFORECAST WARNING",
        ],
        default="NONE",
    )

    queue["supplier_confirmation_required"] = queue[
        "recovery_status"
    ].eq("SUPPLIER CONFIRMATION REQUIRED")

    queue["conditional_air_option"] = (
        queue["bridge_status"].eq("STOCKOUT BRIDGE REQUIRED")
        & queue["new_order_air_before_stockout"]
    )

    queue["stockout_exposure_value"] = (
        queue["minimum_stockout_bridge_units"]
        * queue["unit_cost"]
    ).round(2)

    # Ranking is heuristic, not a validated stockout probability.
    priority_points = {
        "P1": 400,
        "P2": 300,
        "P3": 200,
        "P4": 100,
    }

    queue["decision_score"] = queue[
        "decision_priority"
    ].map(priority_points).astype(float)

    stockout = queue["bridge_status"].eq(
        "STOCKOUT BRIDGE REQUIRED"
    )

    urgency = (
        20 - queue["weeks_to_first_stockout"].fillna(20)
    ).clip(lower=0, upper=20)

    queue["decision_score"] += np.where(
        stockout, urgency, 0
    )

    queue["decision_score"] += (
        np.log1p(
            queue["minimum_stockout_bridge_units"].clip(lower=0)
        ).clip(upper=10)
    )

    queue["decision_score"] += queue[
        "risk_tier"
    ].map(
        {
            "CRITICAL": 8,
            "HIGH": 5,
            "MODERATE": 2,
            "LOW": 0,
        }
    ).fillna(0)

    queue["decision_score"] = queue[
        "decision_score"
    ].round(2)

    queue["analysis_limitation"] = (
        "Synthetic scenario using the 2026-09-30 snapshot. "
        "Dates and rankings are relative to that snapshot, "
        "not live inventory conditions. Existing PO recovery "
        "and air freight are conditional scenarios; supplier "
        "readiness, receipt timing and actual shortage "
        "prevention are not verified. "
        "Do not interpret the decision score as a probability."
    )

    queue = queue.sort_values(
        [
            "decision_score",
            "weeks_to_first_stockout",
            "minimum_stockout_bridge_units",
            "sku",
        ],
        ascending=[False, True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    queue.insert(0, "decision_rank", np.arange(1, len(queue) + 1))

    return queue


def validate_queue(queue, frames):
    checks = []

    def check(name, passed):
        checks.append((name, bool(passed)))
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")

    stockout = queue["bridge_status"].eq(
        "STOCKOUT BRIDGE REQUIRED"
    )

    safety = queue["bridge_status"].eq(
        "SAFETY STOCK BRIDGE ONLY"
    )

    no_bridge = queue["bridge_status"].eq(
        "NO PRE-RECEIPT BRIDGE"
    )

    check("Exactly 200 SKUs", len(queue) == 200)
    check("Unique SKUs", queue["sku"].is_unique)
    check("126 stockout SKUs", stockout.sum() == 126)
    check("33 safety-only SKUs", safety.sum() == 33)
    check("41 no-bridge SKUs", no_bridge.sum() == 41)

    check(
        "126 stockout dates",
        queue["first_stockout_date"].notna().sum() == 126,
    )

    check(
        "Stockout dates only for stockout SKUs",
        queue.loc[~stockout, "first_stockout_date"].isna().all(),
    )

    check(
        "Recovery rows reconcile",
        queue.loc[stockout, "recovery_status"].ne(
            "NOT REQUIRED"
        ).all()
        and queue.loc[~stockout, "recovery_status"].eq(
            "NOT REQUIRED"
        ).all(),
    )

    check(
        "53 supplier confirmations",
        queue["recovery_status"].eq(
            "SUPPLIER CONFIRMATION REQUIRED"
        ).sum() == 53,
    )

    check(
        "46 no-existing-PO cases",
        queue["recovery_status"].eq(
            "NO EXISTING PO - OTHER RECOVERY NEEDED"
        ).sum() == 46,
    )

    check(
        "20 confirmed PO timing reviews",
        queue["recovery_status"].eq(
            "CONFIRMED PO TIMING REVIEW"
        ).sum() == 20,
    )

    check(
        "7 confirmed PO monitoring cases",
        queue["recovery_status"].eq(
            "MONITOR CONFIRMED EXISTING PO"
        ).sum() == 7,
    )

    check(
        "Four conditional air opportunities",
        queue["conditional_air_option"].sum() == 4,
    )

    check(
        "P3 assigned to safety-only cases",
        queue.loc[safety, "decision_priority"].eq("P3").all(),
    )

    check(
        "P4 assigned to no-bridge cases",
        queue.loc[no_bridge, "decision_priority"].eq("P4").all(),
    )

    check(
        "Stockout cases are P1 or P2",
        queue.loc[stockout, "decision_priority"].isin(
            ["P1", "P2"]
        ).all(),
    )

    check(
        "No missing recommendations",
        queue["recommended_action"].notna().all(),
    )

    check(
        "All snapshot dates match",
        queue["snapshot_date"].eq(SNAPSHOT).all(),
    )

    check(
        "Ranks are sequential",
        queue["decision_rank"].tolist() == list(
            range(1, 201)
        ),
    )

    check(
        "Decision scores sorted descending",
        queue["decision_score"].is_monotonic_decreasing,
    )

    check(
        "Bridge units reconcile",
        int(queue["minimum_stockout_bridge_units"].sum())
        == int(
            frames["bridge"][
                "minimum_stockout_bridge_units"
            ].sum()
        ),
    )

    passed = sum(result for _, result in checks)
    failed = len(checks) - passed

    print("\nVALIDATION SUMMARY")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")

    if failed:
        raise AssertionError(
            "Decision queue validation failed; output not saved."
        )


def main():
    print("PROCUREMENT DECISION QUEUE")
    print(f"Planning snapshot: {SNAPSHOT.date()}")
    print("Mode: Synthetic, read-only decision support\n")

    frames = load_data()
    queue = build_queue(frames)

    validate_queue(queue, frames)

    queue.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print(f"SKUs: {len(queue)}")

    print("\nDECISION PRIORITIES")
    print(
        queue["decision_priority"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDECISION CATEGORIES")
    print(
        queue["decision_category"]
        .value_counts()
        .to_string()
    )

    print("\nTOP 10 PRIORITIES")
    print(
        queue[
            [
                "decision_rank", "sku",
                "decision_priority", "decision_category",
                "minimum_stockout_bridge_units",
                "weeks_to_first_stockout", "risk_tier",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
