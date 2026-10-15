
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ENHANCEMENT 1C - EXISTING PO RECOVERY INTELLIGENCE
# ============================================================
# Read-only analysis of existing purchase orders.
#
# Outputs:
#   existing_po_recovery_lines.csv
#   existing_po_recovery_summary.csv
#
# No changes to the planning engine, forecasts, proposals,
# approval workflow, or database.
# ============================================================


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

PO_FILE = DATA_DIR / "open_purchase_orders_clean.csv"
FREIGHT_FILE = DATA_DIR / "freight_recovery_scenarios.csv"
PROJECTION_FILE = DATA_DIR / "feasible_inventory_projection.csv"

LINE_OUTPUT = DATA_DIR / "existing_po_recovery_lines.csv"
SUMMARY_OUTPUT = DATA_DIR / "existing_po_recovery_summary.csv"


def check(condition, message):
    if not condition:
        raise ValueError(message)
    print(f"PASS: {message}")


def load_data():
    files = {
        "open_pos": PO_FILE,
        "freight": FREIGHT_FILE,
        "projection": PROJECTION_FILE,
    }

    data = {}

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        data[name] = pd.read_csv(path)
        print(f"Loaded {path.name}: {len(data[name]):,} rows")

    return data


def as_bool(series):
    values = series.astype(str).str.strip().str.lower()

    valid = {"true", "false", "1", "0"}

    if not values.isin(valid).all():
        raise ValueError(
            f"Invalid Boolean values: "
            f"{values.loc[~values.isin(valid)].unique().tolist()}"
        )

    return values.isin({"true", "1"})


def monday(series):
    return (
        series
        - pd.to_timedelta(series.dt.weekday, unit="D")
    ).dt.normalize()


def prepare_data(data):
    po = data["open_pos"].copy()
    freight = data["freight"].copy()
    projection = data["projection"].copy()

    required_po = {
        "po_number",
        "sku",
        "open_qty",
        "expected_date",
        "open_po_scenario",
        "overdue_flag",
        "partial_receipt_flag",
        "expected_after_promise_flag",
    }

    required_freight = {
        "sku",
        "snapshot_date",
        "priority",
        "minimum_stockout_bridge_units",
        "first_pre_receipt_stockout_week",
    }

    required_projection = {
        "sku",
        "forecast_week",
        "confirmed_existing_receipts",
    }

    for name, frame, required in [
        ("Open POs", po, required_po),
        ("Freight scenarios", freight, required_freight),
        ("Inventory projection", projection, required_projection),
    ]:
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    check(
        not freight["sku"].duplicated().any(),
        "One freight scenario per SKU"
    )

    check(
        not projection.duplicated(
            ["sku", "forecast_week"]
        ).any(),
        "Unique SKU/week inventory projections"
    )

    check(
        po["po_number"].notna().all()
        and po["sku"].notna().all(),
        "PO identifiers and SKUs present"
    )

    check(
        set(po["sku"]).issubset(set(freight["sku"])),
        "All PO SKUs exist in freight scenarios"
    )

    po["open_qty"] = pd.to_numeric(
        po["open_qty"], errors="coerce"
    )

    check(
        po["open_qty"].notna().all()
        and (po["open_qty"] > 0).all(),
        "All open PO quantities positive"
    )

    for column in [
        "overdue_flag",
        "partial_receipt_flag",
        "expected_after_promise_flag",
    ]:
        po[column] = as_bool(po[column])

    po["expected_date"] = pd.to_datetime(
        po["expected_date"], errors="coerce"
    )

    freight["snapshot_date"] = pd.to_datetime(
        freight["snapshot_date"], errors="coerce"
    )

    freight["first_pre_receipt_stockout_week"] = (
        pd.to_datetime(
            freight["first_pre_receipt_stockout_week"],
            errors="coerce"
        )
    )

    freight["minimum_stockout_bridge_units"] = (
        pd.to_numeric(
            freight["minimum_stockout_bridge_units"],
            errors="coerce"
        )
    )

    projection["forecast_week"] = pd.to_datetime(
        projection["forecast_week"], errors="coerce"
    )

    projection["confirmed_existing_receipts"] = (
        pd.to_numeric(
            projection["confirmed_existing_receipts"],
            errors="coerce"
        )
    )

    check(
        po["expected_date"].notna().all()
        and freight["snapshot_date"].notna().all()
        and projection["forecast_week"].notna().all(),
        "Required dates valid"
    )

    check(
        freight["snapshot_date"].nunique() == 1,
        "Single planning snapshot"
    )

    check(
        freight["minimum_stockout_bridge_units"].notna().all()
        and (
            freight["minimum_stockout_bridge_units"] >= 0
        ).all(),
        "Stockout bridge quantities valid"
    )

    check(
        projection["confirmed_existing_receipts"].notna().all()
        and (
            projection["confirmed_existing_receipts"] >= 0
        ).all(),
        "Confirmed projected receipts valid"
    )

    return po, freight, projection


def build_line_analysis(po, freight, projection):
    snapshot = freight["snapshot_date"].iloc[0]

    exposed = freight.loc[
        freight["minimum_stockout_bridge_units"] > 0,
        [
            "sku",
            "priority",
            "minimum_stockout_bridge_units",
            "first_pre_receipt_stockout_week",
        ]
    ].copy()

    lines = po.merge(
        exposed,
        on="sku",
        how="inner",
        validate="many_to_one"
    )

    lines["historical_expected_date"] = (
        lines["expected_date"] < snapshot
    )

    # Match the feasible supply engine's exclusion logic.
    lines["uncertain_receipt"] = (
        lines["overdue_flag"]
        | lines["partial_receipt_flag"]
        | lines["expected_after_promise_flag"]
        | lines["historical_expected_date"]
    )

    lines["included_as_confirmed"] = (
        ~lines["uncertain_receipt"]
    )

    lines["expected_receipt_week"] = monday(
        lines["expected_date"]
    )

    lines["expected_after_stockout_start"] = (
        lines["expected_receipt_week"]
        >= lines["first_pre_receipt_stockout_week"]
    )

    lines["days_from_expected_to_stockout"] = (
        lines["first_pre_receipt_stockout_week"]
        - lines["expected_date"]
    ).dt.days

    def classify(row):
        if row["uncertain_receipt"]:
            if row["expected_after_stockout_start"]:
                return "CONFIRM AND INVESTIGATE EARLIER DELIVERY"

            return "CONFIRM AT-RISK PO RECEIPT"

        if row["expected_after_stockout_start"]:
            return "REVIEW CONFIRMED PO DELIVERY TIMING"

        return "MONITOR CONFIRMED PO"

    lines["recovery_action"] = lines.apply(
        classify, axis=1
    )

    action_rank = {
        "CONFIRM AND INVESTIGATE EARLIER DELIVERY": 1,
        "CONFIRM AT-RISK PO RECEIPT": 2,
        "REVIEW CONFIRMED PO DELIVERY TIMING": 3,
        "MONITOR CONFIRMED PO": 4,
    }

    lines["action_rank"] = (
        lines["recovery_action"].map(action_rank)
    )

    priority_rank = {
        "CRITICAL": 1,
        "HIGH": 2,
        "MEDIUM": 3,
        "LOW": 4,
    }

    lines["priority_rank"] = (
        lines["priority"]
        .astype(str)
        .str.upper()
        .map(priority_rank)
        .fillna(5)
        .astype(int)
    )

    lines["supplier_confirmation_required"] = (
        lines["uncertain_receipt"]
    )

    lines["recovery_units_under_review"] = (
        np.where(
            lines["uncertain_receipt"]
            | lines["expected_after_stockout_start"],
            lines["open_qty"],
            0
        )
    )

    lines["recovery_note"] = np.select(
        [
            lines["uncertain_receipt"]
            & lines["expected_after_stockout_start"],

            lines["uncertain_receipt"],

            lines["expected_after_stockout_start"],
        ],
        [
            "Receipt excluded from confirmed projection; "
            "ask supplier for readiness and earlier delivery.",

            "Receipt excluded from confirmed projection; "
            "verify quantity and delivery commitment.",

            "Receipt already counted in projection; "
            "investigate earlier arrival without double-counting.",
        ],
        default=(
            "Receipt counted in projection; "
            "monitor delivery commitment."
        )
    )

    lines = lines.sort_values(
        [
            "action_rank",
            "priority_rank",
            "first_pre_receipt_stockout_week",
            "sku",
            "expected_date",
        ]
    ).reset_index(drop=True)

    # Reconcile confirmed PO quantities against the
    # conservative inventory projection.
    #
    # The engine maps dates to Monday week buckets.
    # Receipts before the first forecast week are moved
    # to the first week and remain flagged uncertain.

    first_week = projection["forecast_week"].min()

    confirmed = lines.loc[
        lines["included_as_confirmed"]
    ].copy()

    if not confirmed.empty:
        confirmed["receipt_week"] = (
            confirmed["expected_receipt_week"]
        )

        confirmed.loc[
            confirmed["receipt_week"] < first_week,
            "receipt_week"
        ] = first_week

        last_week = projection["forecast_week"].max()

        confirmed = confirmed.loc[
            confirmed["receipt_week"] <= last_week
        ]

        expected_total = confirmed["open_qty"].sum()

        projected_total = projection.loc[
            projection["sku"].isin(exposed["sku"]),
            "confirmed_existing_receipts"
        ].sum()

        check(
            np.isclose(
                expected_total,
                projected_total,
                atol=0.000001
            ),
            "Confirmed existing PO units reconcile "
            "with inventory projection"
        )

        expected_by_week = confirmed.groupby(
            ["sku", "receipt_week"]
        )["open_qty"].sum()

        actual_by_week = projection.loc[
            projection["sku"].isin(exposed["sku"])
        ].groupby(
            ["sku", "forecast_week"]
        )["confirmed_existing_receipts"].sum()

        all_keys = expected_by_week.index.union(
            actual_by_week.index
        )

        differences = (
            expected_by_week.reindex(
                all_keys, fill_value=0
            )
            -
            actual_by_week.reindex(
                all_keys, fill_value=0
            )
        )

        check(
            (differences.abs() < 0.000001).all(),
            "Confirmed receipts reconcile by SKU and week"
        )

    return lines


def build_summary(lines, freight):
    exposed = freight.loc[
        freight["minimum_stockout_bridge_units"] > 0,
        [
            "sku",
            "priority",
            "minimum_stockout_bridge_units",
            "first_pre_receipt_stockout_week",
        ]
    ].copy()

    if lines.empty:
        summary = exposed.copy()

        for column in [
            "open_po_lines",
            "open_po_units",
            "confirmed_po_units",
            "uncertain_po_units",
            "lines_needing_confirmation",
            "lines_for_earlier_delivery_review",
            "recovery_units_under_review",
        ]:
            summary[column] = 0

        summary["recovery_status"] = (
            "NO EXISTING PO - OTHER RECOVERY NEEDED"
        )

        return summary

    temp = lines.copy()

    temp["confirmed_units"] = np.where(
        temp["included_as_confirmed"],
        temp["open_qty"], 0
    )

    temp["uncertain_units"] = np.where(
        temp["uncertain_receipt"],
        temp["open_qty"], 0
    )

    temp["confirmation_line"] = (
        temp["supplier_confirmation_required"].astype(int)
    )

    temp["timing_review_line"] = (
        temp["expected_after_stockout_start"].astype(int)
    )

    grouped = temp.groupby(
        "sku", as_index=False
    ).agg(
        open_po_lines=("po_number", "size"),
        open_po_units=("open_qty", "sum"),
        confirmed_po_units=("confirmed_units", "sum"),
        uncertain_po_units=("uncertain_units", "sum"),
        lines_needing_confirmation=(
            "confirmation_line", "sum"
        ),
        lines_for_earlier_delivery_review=(
            "timing_review_line", "sum"
        ),
        recovery_units_under_review=(
            "recovery_units_under_review", "sum"
        ),
    )

    summary = exposed.merge(
        grouped,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    numeric_columns = [
        "open_po_lines",
        "open_po_units",
        "confirmed_po_units",
        "uncertain_po_units",
        "lines_needing_confirmation",
        "lines_for_earlier_delivery_review",
        "recovery_units_under_review",
    ]

    summary[numeric_columns] = (
        summary[numeric_columns].fillna(0)
    )

    summary["recovery_status"] = np.select(
        [
            summary["open_po_lines"] == 0,

            summary["uncertain_po_units"] > 0,

            summary[
                "lines_for_earlier_delivery_review"
            ] > 0,
        ],
        [
            "NO EXISTING PO - OTHER RECOVERY NEEDED",
            "SUPPLIER CONFIRMATION REQUIRED",
            "CONFIRMED PO TIMING REVIEW",
        ],
        default="MONITOR CONFIRMED EXISTING PO"
    )

    summary = summary.sort_values(
        [
            "uncertain_po_units",
            "first_pre_receipt_stockout_week",
        ],
        ascending=[False, True]
    ).reset_index(drop=True)

    return summary


def validate_results(lines, summary, data):
    po = data["open_pos"]
    freight = data["freight"]

    exposed_skus = set(
        freight.loc[
            freight["minimum_stockout_bridge_units"] > 0,
            "sku"
        ]
    )

    expected_lines = po.loc[
        po["sku"].isin(exposed_skus)
    ]

    check(
        len(lines) == len(expected_lines),
        "All stockout-exposed PO lines preserved"
    )

    check(
        np.isclose(
            lines["open_qty"].sum(),
            expected_lines["open_qty"].sum()
        ),
        "All stockout-exposed PO units preserved"
    )

    check(
        len(summary) == len(exposed_skus),
        "One summary per stockout-exposed SKU"
    )

    check(
        set(summary["sku"]) == exposed_skus,
        "All stockout-exposed SKUs preserved"
    )

    check(
        np.isclose(
            lines["open_qty"].sum(),
            summary["open_po_units"].sum()
        ),
        "Line and summary PO units reconcile"
    )

    check(
        np.isclose(
            lines.loc[
                lines["uncertain_receipt"], "open_qty"
            ].sum(),
            summary["uncertain_po_units"].sum()
        ),
        "Uncertain PO units reconcile"
    )

    check(
        np.isclose(
            lines.loc[
                lines["included_as_confirmed"], "open_qty"
            ].sum(),
            summary["confirmed_po_units"].sum()
        ),
        "Confirmed PO units reconcile"
    )

    check(
        (
            summary["confirmed_po_units"]
            + summary["uncertain_po_units"]
            == summary["open_po_units"]
        ).all(),
        "Confirmed and uncertain supply partition correctly"
    )

    check(
        lines["recovery_action"].notna().all()
        and summary["recovery_status"].notna().all(),
        "All recovery actions classified"
    )


def print_summary(lines, summary):
    print("\n" + "=" * 65)
    print("EXISTING PO RECOVERY INTELLIGENCE")
    print("=" * 65)

    print(f"Stockout-exposed SKUs: {len(summary):,}")
    print(f"Existing PO lines: {len(lines):,}")

    print(
        f"Total existing PO units: "
        f"{lines['open_qty'].sum():,.0f}"
    )

    print(
        f"Confirmed PO units: "
        f"{summary['confirmed_po_units'].sum():,.0f}"
    )

    print(
        f"Uncertain PO units: "
        f"{summary['uncertain_po_units'].sum():,.0f}"
    )

    print("\nPO recovery actions:")
    print(
        lines["recovery_action"]
        .value_counts()
        .to_string()
    )

    print("\nSKU recovery statuses:")
    print(
        summary["recovery_status"]
        .value_counts()
        .to_string()
    )

    print("\nTop 10 supplier follow-up lines:")

    columns = [
        "po_number",
        "sku",
        "priority",
        "open_qty",
        "expected_date",
        "first_pre_receipt_stockout_week",
        "recovery_action",
    ]

    print(
        lines[columns].head(10).to_string(index=False)
    )

    print(
        "\nNOTE: These are supplier follow-up priorities, "
        "not confirmed recoverable quantities."
    )


def main():
    print(
        "\nRunning Enhancement 1C: "
        "Existing PO Recovery Intelligence\n"
    )

    data = load_data()

    po, freight, projection = prepare_data(data)

    lines = build_line_analysis(
        po, freight, projection
    )

    summary = build_summary(lines, freight)

    validate_results(lines, summary, data)

    DATA_DIR.mkdir(
        parents=True, exist_ok=True
    )

    lines.to_csv(
        LINE_OUTPUT,
        index=False,
        date_format="%Y-%m-%d"
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
        date_format="%Y-%m-%d"
    )

    print(f"\nSaved: {LINE_OUTPUT}")
    print(f"Saved: {SUMMARY_OUTPUT}")

    print_summary(lines, summary)


if __name__ == "__main__":
    main()
