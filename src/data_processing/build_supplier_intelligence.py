
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PHASE 8C - SUPPLIER INTELLIGENCE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_data():
    suppliers = pd.read_csv(RAW / "suppliers.csv")
    orders = pd.read_csv(RAW / "purchase_orders.csv")
    receipts = pd.read_csv(RAW / "po_receipts.csv")
    master = pd.read_csv(PROCESSED / "planning_master.csv")
    recovery = pd.read_csv(
        PROCESSED / "supplier_recovery_priorities.csv"
    )

    return suppliers, orders, receipts, master, recovery


def build_supplier_intelligence():
    print("\nPHASE 8C - SUPPLIER INTELLIGENCE")
    print("=" * 65)

    suppliers, orders, receipts, master, recovery = load_data()

    print(f"Suppliers loaded: {len(suppliers):,}")
    print(f"PO lines loaded: {len(orders):,}")
    print(f"Receipt transactions loaded: {len(receipts):,}")
    print(f"Planning SKUs loaded: {len(master):,}")

    # --------------------------------------------------------
    # 1. VALIDATE SOURCE KEYS
    # --------------------------------------------------------

    po_keys = ["po_number", "po_line"]

    if suppliers["supplier_id"].duplicated().any():
        raise ValueError("Duplicate supplier IDs.")

    if orders.duplicated(po_keys).any():
        raise ValueError("Duplicate PO line keys.")

    if receipts["receipt_id"].duplicated().any():
        raise ValueError("Duplicate receipt IDs.")

    if recovery["supplier_id"].duplicated().any():
        raise ValueError("Duplicate supplier recovery records.")

    if not set(orders["supplier_id"]).issubset(
        set(suppliers["supplier_id"])
    ):
        raise ValueError("Unknown suppliers in purchase orders.")

    if set(recovery["supplier_id"]) != set(
        suppliers["supplier_id"]
    ):
        raise ValueError("Recovery supplier IDs do not reconcile.")

    # --------------------------------------------------------
    # 2. PREPARE DATES AND NUMERIC VALUES
    # --------------------------------------------------------

    date_columns = [
        "order_date",
        "promised_date",
        "first_receipt_date",
        "final_receipt_date",
    ]

    for col in date_columns:
        orders[col] = pd.to_datetime(
            orders[col], errors="coerce"
        )

    receipts["receipt_date"] = pd.to_datetime(
        receipts["receipt_date"], errors="coerce"
    )

    numeric_columns = [
        "ordered_qty",
        "received_qty",
        "open_qty",
        "po_value",
        "shipment_count",
    ]

    for col in numeric_columns:
        orders[col] = pd.to_numeric(
            orders[col], errors="raise"
        )

    receipts["receipt_qty"] = pd.to_numeric(
        receipts["receipt_qty"], errors="raise"
    )

    if orders[["order_date", "promised_date"]].isna().any().any():
        raise ValueError("Invalid PO order or promise dates.")

    if receipts["receipt_date"].isna().any():
        raise ValueError("Invalid receipt dates.")

    # --------------------------------------------------------
    # 3. RECONCILE RECEIPTS TO PURCHASE ORDERS
    # --------------------------------------------------------

    receipt_summary = (
        receipts.groupby(po_keys, as_index=False)
        .agg(
            actual_received_qty=("receipt_qty", "sum"),
            actual_first_receipt=("receipt_date", "min"),
            actual_last_receipt=("receipt_date", "max"),
            actual_shipments=("receipt_id", "count"),
        )
    )

    po = orders.merge(
        receipt_summary,
        on=po_keys,
        how="left",
        validate="one_to_one",
    )

    po["actual_received_qty"] = (
        po["actual_received_qty"].fillna(0)
    )

    po["actual_shipments"] = (
        po["actual_shipments"].fillna(0)
    )

    if not np.allclose(
        po["received_qty"],
        po["actual_received_qty"],
    ):
        raise ValueError(
            "Receipt quantities do not reconcile with PO history."
        )

    if not np.allclose(
        po["ordered_qty"],
        po["received_qty"] + po["open_qty"],
    ):
        raise ValueError("PO ordered quantities do not reconcile.")

    if not np.allclose(
        po["shipment_count"],
        po["actual_shipments"],
    ):
        raise ValueError("PO shipment counts do not reconcile.")

    # --------------------------------------------------------
    # 4. DELIVERY PERFORMANCE
    # --------------------------------------------------------

    # Only fully received lines have a final delivery outcome.
    completed = po[
        (po["open_qty"] == 0)
        & (po["received_qty"] == po["ordered_qty"])
        & po["actual_last_receipt"].notna()
    ].copy()

    completed["first_delay_days"] = (
        completed["actual_first_receipt"]
        - completed["promised_date"]
    ).dt.days

    completed["full_delay_days"] = (
        completed["actual_last_receipt"]
        - completed["promised_date"]
    ).dt.days

    completed["actual_first_lead_days"] = (
        completed["actual_first_receipt"]
        - completed["order_date"]
    ).dt.days

    completed["actual_full_lead_days"] = (
        completed["actual_last_receipt"]
        - completed["order_date"]
    ).dt.days

    completed["promised_lead_days"] = (
        completed["promised_date"]
        - completed["order_date"]
    ).dt.days

    completed["first_on_time"] = (
        completed["first_delay_days"] <= 0
    ).astype(int)

    completed["full_on_time"] = (
        completed["full_delay_days"] <= 0
    ).astype(int)

    completed["partial_shipment"] = (
        completed["actual_shipments"] > 1
    ).astype(int)

    performance = (
        completed.groupby("supplier_id", as_index=False)
        .agg(
            completed_po_lines=("po_number", "count"),
            first_on_time_rate=("first_on_time", "mean"),
            full_on_time_rate=("full_on_time", "mean"),
            avg_first_delay_days=("first_delay_days", "mean"),
            avg_full_delay_days=("full_delay_days", "mean"),
            delay_std_days=("full_delay_days", "std"),
            avg_promised_lead_days=("promised_lead_days", "mean"),
            avg_actual_first_lead_days=(
                "actual_first_lead_days", "mean"
            ),
            avg_actual_full_lead_days=(
                "actual_full_lead_days", "mean"
            ),
            partial_shipment_rate=("partial_shipment", "mean"),
            avg_shipments_per_po=("actual_shipments", "mean"),
        )
    )

    historical = (
        po.groupby("supplier_id", as_index=False)
        .agg(
            historical_po_lines=("po_number", "count"),
            historical_po_value=("po_value", "sum"),
            historical_ordered_units=("ordered_qty", "sum"),
            historical_received_units=("received_qty", "sum"),
            historical_open_units=("open_qty", "sum"),
        )
    )

    # --------------------------------------------------------
    # 5. CURRENT SKU AND INVENTORY EXPOSURE
    # --------------------------------------------------------

    sku_summary = (
        master.groupby("supplier_id", as_index=False)
        .agg(
            sku_count=("sku", "nunique"),
            inventory_value=("inventory_value", "sum"),
            net_available_units=("net_available", "sum"),
            open_po_units=("total_open_po_qty", "sum"),
        )
    )

    # --------------------------------------------------------
    # 6. COMBINE SUPPLIER METRICS
    # --------------------------------------------------------

    intelligence = (
        suppliers
        .merge(
            historical,
            on="supplier_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            performance,
            on="supplier_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            sku_summary,
            on="supplier_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            recovery,
            on="supplier_id",
            how="left",
            validate="one_to_one",
        )
    )

    # --------------------------------------------------------
    # 7. RISK CLASSIFICATION
    # --------------------------------------------------------

    # Heuristic operational classification, not a trained ML score.
    # Missing performance is treated as insufficient history.

    def classify_supplier(row):
        if pd.isna(row["full_on_time_rate"]):
            return "INSUFFICIENT HISTORY"

        on_time = row["full_on_time_rate"]
        stockout_weeks = row["total_stockout_sku_weeks"]
        critical = row["critical_skus"]

        if (
            on_time < 0.60
            or stockout_weeks >= 150
            or critical >= 15
        ):
            return "HIGH"

        if (
            on_time < 0.80
            or stockout_weeks >= 75
            or critical >= 7
        ):
            return "MEDIUM"

        return "LOW"

    intelligence["supplier_risk_level"] = (
        intelligence.apply(classify_supplier, axis=1)
    )

    # Transparent, descriptive performance score.
    # Score is based only on observed on-time delivery.
    intelligence["delivery_score"] = (
        intelligence["full_on_time_rate"] * 100
    ).round(1)

    # --------------------------------------------------------
    # 8. VALIDATION
    # --------------------------------------------------------

    if len(intelligence) != len(suppliers):
        raise ValueError("Supplier row count mismatch.")

    if intelligence["supplier_id"].duplicated().any():
        raise ValueError("Duplicate supplier intelligence rows.")

    if intelligence["supplier_id"].isna().any():
        raise ValueError("Missing supplier IDs.")

    if intelligence["sku_count"].sum() != len(master):
        raise ValueError("Supplier SKU totals do not reconcile.")

    if intelligence["historical_po_lines"].sum() != len(orders):
        raise ValueError("Historical PO totals do not reconcile.")

    if intelligence["completed_po_lines"].sum() != len(completed):
        raise ValueError("Completed PO totals do not reconcile.")

    if not np.isclose(
        intelligence["historical_po_value"].sum(),
        orders["po_value"].sum(),
    ):
        raise ValueError("Historical PO value mismatch.")

    if not np.isclose(
        intelligence["total_stockout_sku_weeks"].sum(),
        recovery["total_stockout_sku_weeks"].sum(),
    ):
        raise ValueError("Supplier stockout totals mismatch.")

    for col in [
        "first_on_time_rate",
        "full_on_time_rate",
        "partial_shipment_rate",
    ]:
        valid = intelligence[col].dropna()

        if not valid.between(0, 1).all():
            raise ValueError(f"Invalid rate: {col}")

    # --------------------------------------------------------
    # 9. EXPORT
    # --------------------------------------------------------

    PROCESSED.mkdir(parents=True, exist_ok=True)

    output = PROCESSED / "supplier_intelligence.csv"

    intelligence.to_csv(output, index=False)

    print("\nVALIDATION PASSED")
    print("-" * 65)
    print(f"Suppliers: {len(intelligence):,}")
    print(f"Historical PO lines: {len(orders):,}")
    print(f"Completed PO lines: {len(completed):,}")
    print(f"Receipt transactions: {len(receipts):,}")
    print(f"Planning SKUs: {len(master):,}")

    print("\nSUPPLIER RISK DISTRIBUTION")
    print(
        intelligence["supplier_risk_level"]
        .value_counts()
        .to_string()
    )

    print("\nSUPPLIER PERFORMANCE SUMMARY")
    print(
        intelligence[
            [
                "supplier_id",
                "supplier_name",
                "completed_po_lines",
                "full_on_time_rate",
                "avg_full_delay_days",
                "total_stockout_sku_weeks",
                "supplier_risk_level",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {output}")
    print("\nPHASE 8C DATASET BUILD COMPLETE")


if __name__ == "__main__":
    build_supplier_intelligence()
