from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# FILE PATHS
# =========================================================

RAW_FILES = {
    "suppliers": RAW_DATA_DIR / "suppliers.csv",
    "products": RAW_DATA_DIR / "products.csv",
    "sales": RAW_DATA_DIR / "sales_history.csv",
    "purchase_orders": RAW_DATA_DIR / "purchase_orders.csv",
    "receipts": RAW_DATA_DIR / "po_receipts.csv",
    "inventory": RAW_DATA_DIR / "inventory.csv",
    "open_pos": RAW_DATA_DIR / "open_purchase_orders.csv",
}


PROCESSED_FILES = {
    "suppliers": PROCESSED_DATA_DIR / "suppliers_clean.csv",
    "products": PROCESSED_DATA_DIR / "products_clean.csv",
    "sales": PROCESSED_DATA_DIR / "sales_clean.csv",
    "purchase_orders": PROCESSED_DATA_DIR / "purchase_orders_clean.csv",
    "receipts": PROCESSED_DATA_DIR / "receipts_clean.csv",
    "inventory": PROCESSED_DATA_DIR / "inventory_clean.csv",
    "open_pos": PROCESSED_DATA_DIR / "open_purchase_orders_clean.csv",
}


# =========================================================
# LOAD RAW DATA
# =========================================================

def load_raw_data():

    print("\nLoading raw datasets...")

    data = {}

    for name, file_path in RAW_FILES.items():

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing raw file: {file_path}"
            )

        data[name] = pd.read_csv(
            file_path
        )

        print(
            f"✓ {name:<20} "
            f"{len(data[name]):>7,} rows"
        )

    return data


# =========================================================
# CLEAN SUPPLIERS
# =========================================================

def clean_suppliers(df):

    df = df.copy()

    text_columns = [
        "supplier_id",
        "supplier_name",
        "country",
        "payment_terms",
        "currency",
        "supplier_status",
        "behavior_type",
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    integer_columns = [
        "default_lead_time_weeks",
        "default_transit_weeks",
    ]

    for column in integer_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        ).astype("int64")

    return df


# =========================================================
# CLEAN PRODUCTS
# =========================================================

def clean_products(df):

    df = df.copy()

    text_columns = [
        "sku",
        "product_name",
        "category",
        "supplier_id",
        "demand_profile",
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    integer_columns = [
        "moq",
        "case_pack",
        "quoted_lead_time_weeks",
        "transit_time_weeks",
    ]

    for column in integer_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        ).astype("int64")

    numeric_columns = [
        "unit_cost",
        "unit_price",
        "service_level_target",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        )

    if "active" in df.columns:

        df["active"] = (
            df["active"]
            .astype("boolean")
        )

    # Planned total replenishment time
    df["planned_replenishment_weeks"] = (
        df["quoted_lead_time_weeks"]
        +
        df["transit_time_weeks"]
    )

    # Gross margin dollars
    df["unit_margin"] = (
        df["unit_price"]
        -
        df["unit_cost"]
    ).round(2)

    # Gross margin %
    df["margin_pct"] = (
        df["unit_margin"]
        /
        df["unit_price"]
    ).round(4)

    return df


# =========================================================
# CLEAN SALES
# =========================================================

def clean_sales(df):

    df = df.copy()

    df["week_date"] = pd.to_datetime(
        df["week_date"],
        errors="raise"
    )

    df["sku"] = (
        df["sku"]
        .astype("string")
        .str.strip()
    )

    nullable_integer_columns = [
        "units_demanded",
        "units_sold",
        "lost_sales",
    ]

    for column in nullable_integer_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        ).astype("Int64")

    df["revenue"] = pd.to_numeric(
        df["revenue"],
        errors="coerce"
    )

    if "promotion_flag" in df.columns:

        df["promotion_flag"] = (
            df["promotion_flag"]
            .astype("boolean")
        )

    # Do NOT fill pre-launch nulls with zero.
    df["has_demand_history"] = (
        df["units_demanded"].notna()
    )

    return df


# =========================================================
# CLEAN HISTORICAL PURCHASE ORDERS
# =========================================================

def clean_purchase_orders(df):

    df = df.copy()

    text_columns = [
        "po_number",
        "supplier_id",
        "sku",
        "behavior_type",
        "status",
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    date_columns = [
        "order_date",
        "promised_date",
        "first_receipt_date",
        "final_receipt_date",
    ]

    for column in date_columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="raise"
        )

    integer_columns = [
        "po_line",
        "ordered_qty",
        "received_qty",
        "open_qty",
        "shipment_count",
    ]

    for column in integer_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="raise"
            ).astype("int64")

    numeric_columns = [
        "unit_cost",
        "po_value",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="raise"
            )

    # First receipt delay relative to promise.
    df["first_delay_days"] = (
        df["first_receipt_date"]
        -
        df["promised_date"]
    ).dt.days

    # Complete fulfillment delay.
    df["full_delay_days"] = (
        df["final_receipt_date"]
        -
        df["promised_date"]
    ).dt.days

    # Was the PO split?
    df["partial_shipment_flag"] = (
        df["shipment_count"] > 1
    )

    # On-time metrics.
    df["first_receipt_on_time"] = (
        df["first_receipt_date"]
        <=
        df["promised_date"]
    )

    df["full_po_on_time"] = (
        df["final_receipt_date"]
        <=
        df["promised_date"]
    )

    # Actual elapsed time from order
    # until first and final receipt.
    df["actual_first_lead_days"] = (
        df["first_receipt_date"]
        -
        df["order_date"]
    ).dt.days

    df["actual_full_lead_days"] = (
        df["final_receipt_date"]
        -
        df["order_date"]
    ).dt.days

    return df


# =========================================================
# CLEAN RECEIPTS
# =========================================================

def clean_receipts(df):

    df = df.copy()

    text_columns = [
        "receipt_id",
        "po_number",
        "supplier_id",
        "sku",
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    df["receipt_date"] = pd.to_datetime(
        df["receipt_date"],
        errors="raise"
    )

    integer_columns = [
        "po_line",
        "receipt_number",
        "receipt_qty",
    ]

    for column in integer_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        ).astype("int64")

    return df


# =========================================================
# CLEAN INVENTORY
# =========================================================

def clean_inventory(df):

    df = df.copy()

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"],
        errors="raise"
    )

    text_columns = [
        "sku",
        "warehouse",
        "scenario_label",
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    integer_columns = [
        "on_hand",
        "allocated",
        "available",
        "backordered",
    ]

    for column in integer_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        ).astype("int64")

    df["inventory_value"] = pd.to_numeric(
        df["inventory_value"],
        errors="raise"
    )

    # Net immediately usable inventory
    # after existing backorders.
    df["net_available"] = (
        df["available"]
        -
        df["backordered"]
    )

    return df


# =========================================================
# CLEAN CURRENT OPEN POS
# =========================================================

def clean_open_pos(df):

    df = df.copy()

    text_columns = [
        "po_number",
        "supplier_id",
        "sku",
        "status",
        "open_po_scenario",
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    date_columns = [
        "order_date",
        "promised_date",
        "expected_date",
    ]

    for column in date_columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="raise"
        )

    integer_columns = [
        "po_line",
        "ordered_qty",
        "received_qty",
        "open_qty",
        "days_late",
    ]

    for column in integer_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="raise"
        ).astype("int64")

    df["partial_receipt_flag"] = (
        df["received_qty"] > 0
    )

    df["overdue_flag"] = (
        df["promised_date"]
        <
        pd.Timestamp("2026-09-30")
    )

    df["expected_after_promise_flag"] = (
        df["expected_date"]
        >
        df["promised_date"]
    )

    return df


# =========================================================
# VALIDATE PROCESSED DATA
# =========================================================

def validate_processed_data(data):

    print("\n" + "=" * 70)
    print("PROCESSED DATA VALIDATION")
    print("=" * 70)

    # -----------------------------------------
    # ROW COUNTS PRESERVED
    # -----------------------------------------

    expected_rows = {
        "suppliers": 10,
        "products": 200,
        "sales": 20800,
        "purchase_orders": 2000,
        "receipts": 3105,
        "inventory": 200,
        "open_pos": 124,
    }

    for name, expected in (
        expected_rows.items()
    ):

        actual = len(
            data[name]
        )

        assert actual == expected, (
            f"{name}: expected "
            f"{expected} rows, "
            f"found {actual}"
        )

        print(
            f"✓ {name:<20} "
            f"{actual:>7,} rows"
        )

    # -----------------------------------------
    # PRODUCT FEATURES
    # -----------------------------------------

    assert (
        data["products"][
            "planned_replenishment_weeks"
        ] > 0
    ).all()

    assert (
        data["products"][
            "unit_margin"
        ] > 0
    ).all()

    # -----------------------------------------
    # SALES NULL LOGIC
    # -----------------------------------------

    sales = data["sales"]

    assert (
        sales[
            "has_demand_history"
        ]
        ==
        sales[
            "units_demanded"
        ].notna()
    ).all()

    # -----------------------------------------
    # INVENTORY
    # -----------------------------------------

    inventory = data[
        "inventory"
    ]

    assert (
        inventory[
            "net_available"
        ]
        ==
        inventory[
            "available"
        ]
        -
        inventory[
            "backordered"
        ]
    ).all()

    # -----------------------------------------
    # PO DELAYS
    # -----------------------------------------

    historical_pos = data[
        "purchase_orders"
    ]

    assert (
        historical_pos[
            "full_delay_days"
        ]
        >=
        historical_pos[
            "first_delay_days"
        ]
    ).all()

    print(
        "✓ Engineered product features valid"
    )

    print(
        "✓ Pre-launch demand nulls preserved"
    )

    print(
        "✓ Inventory net availability valid"
    )

    print(
        "✓ Historical PO delay metrics valid"
    )

    print(
        "\nAll processed-data checks passed."
    )


# =========================================================
# SAVE PROCESSED DATA
# =========================================================

def save_processed_data(data):

    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)

    for name, df in data.items():

        output_path = (
            PROCESSED_FILES[name]
        )

        df.to_csv(
            output_path,
            index=False
        )

        print(
            f"✓ {output_path.name}"
        )


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(data):

    print("\n" + "=" * 70)
    print("PROCESSED DATA SUMMARY")
    print("=" * 70)

    products = data["products"]
    inventory = data["inventory"]
    open_pos = data["open_pos"]

    print(
        f"\nActive SKUs: "
        f"{len(products):,}"
    )

    print(
        f"Current inventory value: "
        f"${inventory['inventory_value'].sum():,.2f}"
    )

    print(
        f"Net available units: "
        f"{inventory['net_available'].sum():,.0f}"
    )

    print(
        f"Current open PO units: "
        f"{open_pos['open_qty'].sum():,.0f}"
    )

    print(
        f"Overdue open POs: "
        f"{open_pos['overdue_flag'].sum():,.0f}"
    )

    print(
        f"Partially received open POs: "
        f"{open_pos['partial_receipt_flag'].sum():,.0f}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print(
        "PROCUREMENT AI - "
        "PROCESSED DATA BUILDER"
    )
    print("=" * 70)

    raw = load_raw_data()

    print("\nCleaning datasets...")

    processed = {
        "suppliers": clean_suppliers(
            raw["suppliers"]
        ),

        "products": clean_products(
            raw["products"]
        ),

        "sales": clean_sales(
            raw["sales"]
        ),

        "purchase_orders": (
            clean_purchase_orders(
                raw["purchase_orders"]
            )
        ),

        "receipts": clean_receipts(
            raw["receipts"]
        ),

        "inventory": clean_inventory(
            raw["inventory"]
        ),

        "open_pos": clean_open_pos(
            raw["open_pos"]
        ),
    }

    validate_processed_data(
        processed
    )

    save_processed_data(
        processed
    )

    print_summary(
        processed
    )

    print("\n" + "=" * 70)
    print("PHASE 3B COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()