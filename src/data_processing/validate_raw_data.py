from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


FILES = {
    "suppliers": RAW_DATA_DIR / "suppliers.csv",
    "products": RAW_DATA_DIR / "products.csv",
    "sales": RAW_DATA_DIR / "sales_history.csv",
    "purchase_orders": RAW_DATA_DIR / "purchase_orders.csv",
    "receipts": RAW_DATA_DIR / "po_receipts.csv",
    "inventory": RAW_DATA_DIR / "inventory.csv",
    "open_pos": RAW_DATA_DIR / "open_purchase_orders.csv",
}


# =========================================================
# AUDIT TRACKER
# =========================================================

audit_results = []


def record_check(
    category,
    check_name,
    passed,
    details=""
):

    audit_results.append(
        {
            "category": category,
            "check": check_name,
            "status": (
                "PASS"
                if passed
                else "FAIL"
            ),
            "details": details,
        }
    )


# =========================================================
# FILE VALIDATION
# =========================================================

def validate_files():

    print("=" * 70)
    print("1. FILE VALIDATION")
    print("=" * 70)

    all_exist = True

    for name, file_path in FILES.items():

        exists = file_path.exists()

        record_check(
            "Files",
            f"{name} exists",
            exists,
            str(file_path)
        )

        if exists:

            print(
                f"✓ {name}: "
                f"{file_path.name}"
            )

        else:

            print(
                f"✗ Missing: "
                f"{file_path}"
            )

            all_exist = False

    if not all_exist:

        raise FileNotFoundError(
            "One or more required raw "
            "data files are missing."
        )


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    data = {}

    for name, file_path in FILES.items():

        data[name] = pd.read_csv(
            file_path
        )

    return data


# =========================================================
# DATASET SHAPE
# =========================================================

def inspect_shapes(data):

    print("\n" + "=" * 70)
    print("2. DATASET SHAPES")
    print("=" * 70)

    for name, df in data.items():

        print(
            f"{name:<20} "
            f"Rows: {len(df):>7,} | "
            f"Columns: {len(df.columns):>2}"
        )

        record_check(
            "Shape",
            f"{name} is not empty",
            len(df) > 0,
            f"{len(df):,} rows"
        )


# =========================================================
# REQUIRED COLUMNS
# =========================================================

def validate_columns(data):

    print("\n" + "=" * 70)
    print("3. REQUIRED COLUMN VALIDATION")
    print("=" * 70)

    required_columns = {

        "suppliers": [
            "supplier_id",
            "supplier_name",
            "country",
            "default_lead_time_weeks",
            "default_transit_weeks",
            "behavior_type",
        ],

        "products": [
            "sku",
            "product_name",
            "category",
            "supplier_id",
            "unit_cost",
            "unit_price",
            "moq",
            "case_pack",
            "quoted_lead_time_weeks",
            "transit_time_weeks",
            "service_level_target",
            "demand_profile",
        ],

        "sales": [
            "week_date",
            "sku",
            "units_demanded",
            "units_sold",
            "lost_sales",
            "revenue",
        ],

        "purchase_orders": [
            "po_number",
            "supplier_id",
            "sku",
            "order_date",
            "ordered_qty",
            "promised_date",
            "received_qty",
            "open_qty",
            "status",
        ],

        "receipts": [
            "receipt_id",
            "po_number",
            "supplier_id",
            "sku",
            "receipt_date",
            "receipt_qty",
        ],

        "inventory": [
            "snapshot_date",
            "sku",
            "on_hand",
            "allocated",
            "available",
            "backordered",
            "inventory_value",
        ],

        "open_pos": [
            "po_number",
            "supplier_id",
            "sku",
            "order_date",
            "ordered_qty",
            "received_qty",
            "open_qty",
            "promised_date",
            "expected_date",
            "status",
            "days_late",
        ],
    }

    for name, columns in required_columns.items():

        missing = [
            column
            for column in columns
            if column not in data[name].columns
        ]

        passed = len(missing) == 0

        record_check(
            "Schema",
            f"{name} required columns",
            passed,
            (
                "All required columns present"
                if passed
                else f"Missing: {missing}"
            )
        )

        if passed:

            print(
                f"✓ {name}"
            )

        else:

            print(
                f"✗ {name}: "
                f"Missing {missing}"
            )


# =========================================================
# PRIMARY KEY CHECKS
# =========================================================

def validate_primary_keys(data):

    print("\n" + "=" * 70)
    print("4. PRIMARY KEY VALIDATION")
    print("=" * 70)

    checks = [
        (
            "suppliers",
            "supplier_id"
        ),
        (
            "products",
            "sku"
        ),
        (
            "receipts",
            "receipt_id"
        ),
        (
            "inventory",
            "sku"
        ),
        (
            "open_pos",
            "po_number"
        ),
    ]

    for dataset, column in checks:

        duplicates = (
            data[dataset][column]
            .duplicated()
            .sum()
        )

        passed = duplicates == 0

        record_check(
            "Primary Key",
            f"{dataset}.{column} unique",
            passed,
            f"{duplicates} duplicates"
        )

        symbol = "✓" if passed else "✗"

        print(
            f"{symbol} "
            f"{dataset}.{column} "
            f"duplicates: {duplicates}"
        )


# =========================================================
# FOREIGN KEY CHECKS
# =========================================================

def foreign_key_check(
    child_df,
    child_column,
    parent_df,
    parent_column
):

    child_values = set(
        child_df[
            child_column
        ].dropna()
    )

    parent_values = set(
        parent_df[
            parent_column
        ].dropna()
    )

    invalid = (
        child_values
        - parent_values
    )

    return invalid


def validate_relationships(data):

    print("\n" + "=" * 70)
    print("5. RELATIONSHIP VALIDATION")
    print("=" * 70)

    relationships = [

        (
            "products",
            "supplier_id",
            "suppliers",
            "supplier_id",
        ),

        (
            "sales",
            "sku",
            "products",
            "sku",
        ),

        (
            "purchase_orders",
            "sku",
            "products",
            "sku",
        ),

        (
            "purchase_orders",
            "supplier_id",
            "suppliers",
            "supplier_id",
        ),

        (
            "receipts",
            "po_number",
            "purchase_orders",
            "po_number",
        ),

        (
            "receipts",
            "sku",
            "products",
            "sku",
        ),

        (
            "inventory",
            "sku",
            "products",
            "sku",
        ),

        (
            "open_pos",
            "sku",
            "products",
            "sku",
        ),

        (
            "open_pos",
            "supplier_id",
            "suppliers",
            "supplier_id",
        ),
    ]

    for (
        child_name,
        child_column,
        parent_name,
        parent_column
    ) in relationships:

        invalid = foreign_key_check(
            data[child_name],
            child_column,
            data[parent_name],
            parent_column,
        )

        passed = len(invalid) == 0

        record_check(
            "Relationship",
            (
                f"{child_name}.{child_column} "
                f"-> "
                f"{parent_name}.{parent_column}"
            ),
            passed,
            (
                "Valid"
                if passed
                else (
                    f"{len(invalid)} "
                    f"invalid values"
                )
            )
        )

        symbol = "✓" if passed else "✗"

        print(
            f"{symbol} "
            f"{child_name}.{child_column} "
            f"→ "
            f"{parent_name}.{parent_column}"
        )

        if invalid:

            print(
                f"   Invalid values: "
                f"{list(invalid)[:5]}"
            )


# =========================================================
# NULL AUDIT
# =========================================================

def audit_nulls(data):

    print("\n" + "=" * 70)
    print("6. NULL VALUE AUDIT")
    print("=" * 70)

    for name, df in data.items():

        null_counts = (
            df.isna().sum()
        )

        null_counts = (
            null_counts[
                null_counts > 0
            ]
        )

        print(
            f"\n{name}:"
        )

        if null_counts.empty:

            print(
                "  No null values"
            )

        else:

            for column, count in (
                null_counts.items()
            ):

                percentage = (
                    count
                    / len(df)
                    * 100
                )

                print(
                    f"  {column}: "
                    f"{count:,} "
                    f"({percentage:.2f}%)"
                )


# =========================================================
# BUSINESS RULE VALIDATION
# =========================================================

def validate_business_rules(data):

    print("\n" + "=" * 70)
    print("7. BUSINESS RULE VALIDATION")
    print("=" * 70)

    checks = []

    # -----------------------------------------
    # PRODUCT RULES
    # -----------------------------------------

    checks.append(
        (
            "Product costs > 0",
            (
                data["products"][
                    "unit_cost"
                ] > 0
            ).all()
        )
    )

    checks.append(
        (
            "Product prices > costs",
            (
                data["products"][
                    "unit_price"
                ]
                >
                data["products"][
                    "unit_cost"
                ]
            ).all()
        )
    )

    checks.append(
        (
            "MOQ > 0",
            (
                data["products"][
                    "moq"
                ] > 0
            ).all()
        )
    )

    checks.append(
        (
            "Case pack > 0",
            (
                data["products"][
                    "case_pack"
                ] > 0
            ).all()
        )
    )

    # -----------------------------------------
    # SALES RULES
    # -----------------------------------------

    sales_valid = data[
        "sales"
    ].dropna(
        subset=[
            "units_demanded",
            "units_sold",
            "lost_sales",
        ]
    )

    checks.append(
        (
            "Sales never exceed demand",
            (
                sales_valid[
                    "units_sold"
                ]
                <=
                sales_valid[
                    "units_demanded"
                ]
            ).all()
        )
    )

    checks.append(
        (
            "Demand = sold + lost sales",
            (
                sales_valid[
                    "units_demanded"
                ]
                ==
                sales_valid[
                    "units_sold"
                ]
                +
                sales_valid[
                    "lost_sales"
                ]
            ).all()
        )
    )

    # -----------------------------------------
    # INVENTORY RULES
    # -----------------------------------------

    checks.append(
        (
            (
                "Inventory available = "
                "on hand - allocated"
            ),
            (
                data["inventory"][
                    "available"
                ]
                ==
                data["inventory"][
                    "on_hand"
                ]
                -
                data["inventory"][
                    "allocated"
                ]
            ).all()
        )
    )

    checks.append(
        (
            "Inventory available >= 0",
            (
                data["inventory"][
                    "available"
                ] >= 0
            ).all()
        )
    )

    # -----------------------------------------
    # OPEN PO RULES
    # -----------------------------------------

    checks.append(
        (
            (
                "Open PO received + open "
                "= ordered"
            ),
            (
                data["open_pos"][
                    "received_qty"
                ]
                +
                data["open_pos"][
                    "open_qty"
                ]
                ==
                data["open_pos"][
                    "ordered_qty"
                ]
            ).all()
        )
    )

    checks.append(
        (
            "Open PO quantity > 0",
            (
                data["open_pos"][
                    "open_qty"
                ] > 0
            ).all()
        )
    )

    # -----------------------------------------
    # RUN CHECKS
    # -----------------------------------------

    for check_name, passed in checks:

        record_check(
            "Business Rule",
            check_name,
            bool(passed),
            ""
        )

        symbol = (
            "✓"
            if passed
            else "✗"
        )

        print(
            f"{symbol} {check_name}"
        )


# =========================================================
# DATE VALIDATION
# =========================================================

def validate_dates(data):

    print("\n" + "=" * 70)
    print("8. DATE VALIDATION")
    print("=" * 70)

    date_columns = {

        "sales": [
            "week_date"
        ],

        "purchase_orders": [
            "order_date",
            "promised_date",
            "first_receipt_date",
            "final_receipt_date",
        ],

        "receipts": [
            "receipt_date"
        ],

        "inventory": [
            "snapshot_date"
        ],

        "open_pos": [
            "order_date",
            "promised_date",
            "expected_date",
        ],
    }

    for dataset, columns in (
        date_columns.items()
    ):

        for column in columns:

            parsed = pd.to_datetime(
                data[dataset][column],
                errors="coerce"
            )

            invalid_count = (
                parsed.isna().sum()
                -
                data[dataset][
                    column
                ].isna().sum()
            )

            passed = (
                invalid_count == 0
            )

            record_check(
                "Date",
                (
                    f"{dataset}."
                    f"{column} parseable"
                ),
                passed,
                (
                    f"{invalid_count} "
                    f"invalid dates"
                )
            )

            symbol = (
                "✓"
                if passed
                else "✗"
            )

            print(
                f"{symbol} "
                f"{dataset}.{column}"
            )


# =========================================================
# SPECIAL NULL VALIDATION
# =========================================================

def validate_expected_nulls(data):

    print("\n" + "=" * 70)
    print("9. EXPECTED NULL VALIDATION")
    print("=" * 70)

    sales = data[
        "sales"
    ].copy()

    products = data[
        "products"
    ][
        [
            "sku",
            "demand_profile"
        ]
    ]

    sales = sales.merge(
        products,
        on="sku",
        how="left"
    )

    demand_nulls = sales[
        sales[
            "units_demanded"
        ].isna()
    ]

    if demand_nulls.empty:

        passed = True

    else:

        passed = (
            demand_nulls[
                "demand_profile"
            ]
            ==
            "new_product"
        ).all()

    record_check(
        "Null Logic",
        (
            "Demand nulls only belong "
            "to pre-launch new products"
        ),
        bool(passed),
        (
            f"{len(demand_nulls):,} "
            f"demand-null rows"
        )
    )

    symbol = (
        "✓"
        if passed
        else "✗"
    )

    print(
        f"{symbol} Demand nulls only "
        f"belong to new products"
    )

    print(
        f"  Demand-null rows: "
        f"{len(demand_nulls):,}"
    )


# =========================================================
# AUDIT REPORT
# =========================================================

def print_audit_report():

    audit_df = pd.DataFrame(
        audit_results
    )

    print("\n" + "=" * 70)
    print("10. FINAL DATA QUALITY REPORT")
    print("=" * 70)

    pass_count = (
        audit_df[
            "status"
        ] == "PASS"
    ).sum()

    fail_count = (
        audit_df[
            "status"
        ] == "FAIL"
    ).sum()

    total_count = len(
        audit_df
    )

    score = (
        pass_count
        / total_count
        * 100
        if total_count > 0
        else 0
    )

    print(
        f"\nChecks performed: "
        f"{total_count}"
    )

    print(
        f"Passed: {pass_count}"
    )

    print(
        f"Failed: {fail_count}"
    )

    print(
        f"Data Quality Score: "
        f"{score:.1f}%"
    )

    if fail_count == 0:

        print(
            "\n✓ RAW DATA AUDIT PASSED"
        )

    else:

        print(
            "\n✗ RAW DATA AUDIT "
            "HAS FAILURES"
        )

        print(
            "\nFailed checks:"
        )

        failed = audit_df[
            audit_df[
                "status"
            ] == "FAIL"
        ]

        print(
            failed[
                [
                    "category",
                    "check",
                    "details"
                ]
            ].to_string(
                index=False
            )
        )

    return audit_df


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print(
        "PROCUREMENT AI - "
        "RAW DATA AUDIT"
    )
    print("=" * 70)

    validate_files()

    data = load_data()

    inspect_shapes(
        data
    )

    validate_columns(
        data
    )

    validate_primary_keys(
        data
    )

    validate_relationships(
        data
    )

    audit_nulls(
        data
    )

    validate_business_rules(
        data
    )

    validate_dates(
        data
    )

    validate_expected_nulls(
        data
    )

    audit_df = (
        print_audit_report()
    )

    print("\n" + "=" * 70)
    print("PHASE 3A COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()