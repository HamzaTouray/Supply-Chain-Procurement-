from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

SNAPSHOT_DATE = pd.Timestamp("2026-09-30")


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

PRODUCT_FILE = (
    PROCESSED_DATA_DIR
    / "products_clean.csv"
)

SUPPLIER_FILE = (
    PROCESSED_DATA_DIR
    / "suppliers_clean.csv"
)

SALES_FILE = (
    PROCESSED_DATA_DIR
    / "sales_clean.csv"
)

INVENTORY_FILE = (
    PROCESSED_DATA_DIR
    / "inventory_clean.csv"
)

HISTORICAL_PO_FILE = (
    PROCESSED_DATA_DIR
    / "purchase_orders_clean.csv"
)

OPEN_PO_FILE = (
    PROCESSED_DATA_DIR
    / "open_purchase_orders_clean.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "planning_master.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\nLoading processed datasets...")

    files = {
        "products": PRODUCT_FILE,
        "suppliers": SUPPLIER_FILE,
        "sales": SALES_FILE,
        "inventory": INVENTORY_FILE,
        "historical_pos": HISTORICAL_PO_FILE,
        "open_pos": OPEN_PO_FILE,
    }

    data = {}

    for name, file_path in files.items():

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing processed file: "
                f"{file_path}"
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
# PREPARE DATES
# =========================================================

def prepare_dates(data):

    sales = data["sales"]

    sales["week_date"] = pd.to_datetime(
        sales["week_date"]
    )

    inventory = data["inventory"]

    inventory["snapshot_date"] = pd.to_datetime(
        inventory["snapshot_date"]
    )

    historical_pos = data[
        "historical_pos"
    ]

    historical_date_columns = [
        "order_date",
        "promised_date",
        "first_receipt_date",
        "final_receipt_date",
    ]

    for column in historical_date_columns:

        historical_pos[column] = pd.to_datetime(
            historical_pos[column]
        )

    open_pos = data["open_pos"]

    open_date_columns = [
        "order_date",
        "promised_date",
        "expected_date",
    ]

    for column in open_date_columns:

        open_pos[column] = pd.to_datetime(
            open_pos[column]
        )

    return data


# =========================================================
# DEMAND FEATURES
# =========================================================

def calculate_demand_features(
    sales_df
):

    print("\nBuilding demand features...")

    sales = sales_df.copy()

    # -----------------------------------------------------
    # IMPORTANT:
    # Pre-launch NaN demand stays NaN.
    #
    # We only use rows where actual demand history exists.
    # -----------------------------------------------------

    valid_sales = sales[
        sales["units_demanded"].notna()
    ].copy()

    valid_sales[
        "units_demanded"
    ] = pd.to_numeric(
        valid_sales[
            "units_demanded"
        ]
    )

    valid_sales[
        "lost_sales"
    ] = pd.to_numeric(
        valid_sales[
            "lost_sales"
        ]
    )

    valid_sales[
        "units_sold"
    ] = pd.to_numeric(
        valid_sales[
            "units_sold"
        ]
    )

    # -----------------------------------------------------
    # FULL HISTORY
    # -----------------------------------------------------

    full_history = (
        valid_sales
        .groupby("sku")
        .agg(
            history_weeks=(
                "units_demanded",
                "count"
            ),

            lifetime_avg_weekly_demand=(
                "units_demanded",
                "mean"
            ),

            lifetime_demand_std=(
                "units_demanded",
                "std"
            ),

            lifetime_demand_units=(
                "units_demanded",
                "sum"
            ),

            lifetime_units_sold=(
                "units_sold",
                "sum"
            ),

            lifetime_lost_sales=(
                "lost_sales",
                "sum"
            ),

            first_demand_date=(
                "week_date",
                "min"
            ),

            last_demand_date=(
                "week_date",
                "max"
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # RECENT WINDOW FUNCTION
    # -----------------------------------------------------

    def window_summary(
        weeks,
        prefix
    ):

        start_date = (
            SNAPSHOT_DATE
            - pd.Timedelta(
                weeks=weeks - 1
            )
        )

        window = valid_sales[
            (
                valid_sales[
                    "week_date"
                ] >= start_date
            )
            &
            (
                valid_sales[
                    "week_date"
                ] <= SNAPSHOT_DATE
            )
        ].copy()

        summary = (
            window
            .groupby("sku")
            .agg(
                **{
                    f"{prefix}_avg_demand": (
                        "units_demanded",
                        "mean"
                    ),

                    f"{prefix}_demand_std": (
                        "units_demanded",
                        "std"
                    ),

                    f"{prefix}_total_demand": (
                        "units_demanded",
                        "sum"
                    ),

                    f"{prefix}_lost_sales": (
                        "lost_sales",
                        "sum"
                    ),
                }
            )
            .reset_index()
        )

        return summary

    demand_4w = window_summary(
        4,
        "4w"
    )

    demand_13w = window_summary(
        13,
        "13w"
    )

    demand_26w = window_summary(
        26,
        "26w"
    )

    demand_52w = window_summary(
        52,
        "52w"
    )

    # -----------------------------------------------------
    # COMBINE
    # -----------------------------------------------------

    demand_features = full_history.copy()

    for feature_df in [
        demand_4w,
        demand_13w,
        demand_26w,
        demand_52w,
    ]:

        demand_features = (
            demand_features.merge(
                feature_df,
                on="sku",
                how="left"
            )
        )

    # -----------------------------------------------------
    # DEMAND VARIABILITY
    # -----------------------------------------------------

    demand_features[
        "13w_cv"
    ] = np.where(
        demand_features[
            "13w_avg_demand"
        ] > 0,

        demand_features[
            "13w_demand_std"
        ]
        /
        demand_features[
            "13w_avg_demand"
        ],

        np.nan
    )

    # -----------------------------------------------------
    # SHORT-TERM TREND
    #
    # > 1 means recent demand is stronger
    # < 1 means recent demand is weaker
    # -----------------------------------------------------

    demand_features[
        "demand_trend_ratio"
    ] = np.where(
        demand_features[
            "13w_avg_demand"
        ] > 0,

        demand_features[
            "4w_avg_demand"
        ]
        /
        demand_features[
            "13w_avg_demand"
        ],

        np.nan
    )

    # -----------------------------------------------------
    # LOST SALES RATE
    # -----------------------------------------------------

    demand_features[
        "13w_lost_sales_rate"
    ] = np.where(
        demand_features[
            "13w_total_demand"
        ] > 0,

        demand_features[
            "13w_lost_sales"
        ]
        /
        demand_features[
            "13w_total_demand"
        ],

        0
    )

    return demand_features


# =========================================================
# HISTORICAL SUPPLIER PERFORMANCE
# =========================================================

def calculate_supplier_features(
    historical_po_df
):

    print(
        "Building supplier performance features..."
    )

    po = historical_po_df.copy()

    supplier_features = (
        po
        .groupby("supplier_id")
        .agg(
            historical_po_count=(
                "po_number",
                "count"
            ),

            supplier_avg_first_delay_days=(
                "first_delay_days",
                "mean"
            ),

            supplier_avg_full_delay_days=(
                "full_delay_days",
                "mean"
            ),

            supplier_delay_std_days=(
                "full_delay_days",
                "std"
            ),

            supplier_avg_actual_first_lead_days=(
                "actual_first_lead_days",
                "mean"
            ),

            supplier_avg_actual_full_lead_days=(
                "actual_full_lead_days",
                "mean"
            ),

            supplier_first_on_time_rate=(
                "first_receipt_on_time",
                "mean"
            ),

            supplier_full_on_time_rate=(
                "full_po_on_time",
                "mean"
            ),

            supplier_partial_shipment_rate=(
                "partial_shipment_flag",
                "mean"
            ),

            supplier_avg_shipments_per_po=(
                "shipment_count",
                "mean"
            ),
        )
        .reset_index()
    )

    return supplier_features


# =========================================================
# SKU PROCUREMENT HISTORY
# =========================================================

def calculate_sku_procurement_features(
    historical_po_df
):

    print(
        "Building SKU procurement features..."
    )

    po = historical_po_df.copy()

    sku_features = (
        po
        .groupby("sku")
        .agg(
            sku_historical_po_count=(
                "po_number",
                "count"
            ),

            sku_avg_first_delay_days=(
                "first_delay_days",
                "mean"
            ),

            sku_avg_full_delay_days=(
                "full_delay_days",
                "mean"
            ),

            sku_avg_actual_full_lead_days=(
                "actual_full_lead_days",
                "mean"
            ),

            sku_full_on_time_rate=(
                "full_po_on_time",
                "mean"
            ),

            sku_partial_shipment_rate=(
                "partial_shipment_flag",
                "mean"
            ),
        )
        .reset_index()
    )

    return sku_features


# =========================================================
# CURRENT OPEN PO FEATURES
# =========================================================

def calculate_open_po_features(
    open_po_df
):

    print(
        "Building current open-PO features..."
    )

    open_pos = open_po_df.copy()

    # -----------------------------------------------------
    # CURRENT FLAGS
    # -----------------------------------------------------

    open_pos[
        "overdue_now"
    ] = (
        open_pos[
            "promised_date"
        ] < SNAPSHOT_DATE
    )

    open_pos[
        "partial_now"
    ] = (
        open_pos[
            "received_qty"
        ] > 0
    )

    open_pos[
        "expected_late"
    ] = (
        open_pos[
            "expected_date"
        ]
        >
        open_pos[
            "promised_date"
        ]
    )

    open_pos[
        "overdue_open_qty"
    ] = np.where(
        open_pos[
            "overdue_now"
        ],

        open_pos[
            "open_qty"
        ],

        0
    )

    open_pos[
        "partial_open_qty"
    ] = np.where(
        open_pos[
            "partial_now"
        ],

        open_pos[
            "open_qty"
        ],

        0
    )

    # -----------------------------------------------------
    # DAYS UNTIL EXPECTED RECEIPT
    # -----------------------------------------------------

    open_pos[
        "days_until_expected"
    ] = (
        open_pos[
            "expected_date"
        ]
        -
        SNAPSHOT_DATE
    ).dt.days

    # -----------------------------------------------------
    # SUMMARY BY SKU
    # -----------------------------------------------------

    summary = (
        open_pos
        .groupby("sku")
        .agg(
            open_po_count=(
                "po_number",
                "count"
            ),

            total_open_po_qty=(
                "open_qty",
                "sum"
            ),

            total_overdue_po_qty=(
                "overdue_open_qty",
                "sum"
            ),

            total_partial_po_qty=(
                "partial_open_qty",
                "sum"
            ),

            next_expected_receipt_date=(
                "expected_date",
                "min"
            ),

            latest_expected_receipt_date=(
                "expected_date",
                "max"
            ),

            max_days_late=(
                "days_late",
                "max"
            ),

            overdue_po_count=(
                "overdue_now",
                "sum"
            ),

            partial_po_count=(
                "partial_now",
                "sum"
            ),

            expected_late_po_count=(
                "expected_late",
                "sum"
            ),
        )
        .reset_index()
    )

    return summary


# =========================================================
# BUILD PLANNING MASTER
# =========================================================

def build_planning_master(
    data,
    demand_features,
    supplier_features,
    sku_procurement_features,
    open_po_features
):

    print(
        "\nBuilding planning master table..."
    )

    products = data[
        "products"
    ].copy()

    suppliers = data[
        "suppliers"
    ].copy()

    inventory = data[
        "inventory"
    ].copy()

    # -----------------------------------------------------
    # PRODUCT + SUPPLIER MASTER
    # -----------------------------------------------------

    supplier_columns = [
        "supplier_id",
        "supplier_name",
        "country",
        "default_lead_time_weeks",
        "default_transit_weeks",
        "behavior_type",
    ]

    master = products.merge(
        suppliers[
            supplier_columns
        ],
        on="supplier_id",
        how="left",
        validate="many_to_one"
    )

    # -----------------------------------------------------
    # INVENTORY
    # -----------------------------------------------------

    inventory_columns = [
        "sku",
        "snapshot_date",
        "warehouse",
        "on_hand",
        "allocated",
        "available",
        "backordered",
        "net_available",
        "inventory_value",
        "scenario_label",
    ]

    master = master.merge(
        inventory[
            inventory_columns
        ],
        on="sku",
        how="left",
        validate="one_to_one"
    )

    # -----------------------------------------------------
    # DEMAND
    # -----------------------------------------------------

    master = master.merge(
        demand_features,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    # -----------------------------------------------------
    # SUPPLIER PERFORMANCE
    # -----------------------------------------------------

    master = master.merge(
        supplier_features,
        on="supplier_id",
        how="left",
        validate="many_to_one"
    )

    # -----------------------------------------------------
    # SKU PROCUREMENT HISTORY
    # -----------------------------------------------------

    master = master.merge(
        sku_procurement_features,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    # -----------------------------------------------------
    # CURRENT OPEN POS
    # -----------------------------------------------------

    master = master.merge(
        open_po_features,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    return master


# =========================================================
# FINAL DERIVED FEATURES
# =========================================================

def add_master_features(
    master
):

    print(
        "Adding planning-level features..."
    )

    df = master.copy()

    # -----------------------------------------------------
    # OPEN PO NULLS
    #
    # No open PO row means zero open supply.
    # -----------------------------------------------------

    zero_fill_columns = [
        "open_po_count",
        "total_open_po_qty",
        "total_overdue_po_qty",
        "total_partial_po_qty",
        "max_days_late",
        "overdue_po_count",
        "partial_po_count",
        "expected_late_po_count",
    ]

    for column in zero_fill_columns:

        df[column] = (
            df[column]
            .fillna(0)
        )

    integer_columns = [
        "open_po_count",
        "total_open_po_qty",
        "total_overdue_po_qty",
        "total_partial_po_qty",
        "max_days_late",
        "overdue_po_count",
        "partial_po_count",
        "expected_late_po_count",
    ]

    for column in integer_columns:

        df[column] = (
            df[column]
            .astype(int)
        )

    # -----------------------------------------------------
    # INVENTORY POSITION
    #
    # This is NOT a time-phased forecast.
    # It is simply today's net inventory
    # plus all currently open supply.
    # -----------------------------------------------------

    df[
        "inventory_position"
    ] = (
        df[
            "net_available"
        ]
        +
        df[
            "total_open_po_qty"
        ]
    )

    # -----------------------------------------------------
    # CURRENT WEEKS OF SUPPLY
    #
    # Uses 13-week average demand.
    # Open POs are NOT included here.
    # -----------------------------------------------------

    df[
        "current_weeks_of_supply"
    ] = np.where(
        df[
            "13w_avg_demand"
        ] > 0,

        np.maximum(
            df[
                "net_available"
            ],
            0
        )
        /
        df[
            "13w_avg_demand"
        ],

        np.nan
    )

    # -----------------------------------------------------
    # WEEKS OF SUPPLY INCLUDING OPEN POS
    #
    # Diagnostic only.
    #
    # Later planning logic will NOT assume
    # all inbound supply is available today.
    # -----------------------------------------------------

    df[
        "position_weeks_of_supply"
    ] = np.where(
        df[
            "13w_avg_demand"
        ] > 0,

        np.maximum(
            df[
                "inventory_position"
            ],
            0
        )
        /
        df[
            "13w_avg_demand"
        ],

        np.nan
    )

    # -----------------------------------------------------
    # CURRENT STOCKOUT FLAG
    # -----------------------------------------------------

    df[
        "currently_stocked_out"
    ] = (
        df[
            "net_available"
        ] <= 0
    )

    # -----------------------------------------------------
    # HAS OPEN SUPPLY
    # -----------------------------------------------------

    df[
        "has_open_po"
    ] = (
        df[
            "total_open_po_qty"
        ] > 0
    )

    # -----------------------------------------------------
    # HAS OVERDUE SUPPLY
    # -----------------------------------------------------

    df[
        "has_overdue_po"
    ] = (
        df[
            "overdue_po_count"
        ] > 0
    )

    # -----------------------------------------------------
    # HAS PARTIAL SUPPLY
    # -----------------------------------------------------

    df[
        "has_partial_po"
    ] = (
        df[
            "partial_po_count"
        ] > 0
    )

    return df


# =========================================================
# VALIDATION
# =========================================================

def validate_master(
    master,
    data
):

    print("\n" + "=" * 70)
    print("PLANNING MASTER VALIDATION")
    print("=" * 70)

    # -----------------------------------------------------
    # ROW COUNT
    # -----------------------------------------------------

    assert len(master) == 200, (
        f"Expected 200 SKUs, "
        f"found {len(master)}"
    )

    print(
        "✓ Exactly 200 SKU rows"
    )

    # -----------------------------------------------------
    # UNIQUE SKU
    # -----------------------------------------------------

    assert (
        master[
            "sku"
        ].is_unique
    )

    print(
        "✓ One row per SKU"
    )

    # -----------------------------------------------------
    # ALL PRODUCTS PRESERVED
    # -----------------------------------------------------

    assert set(
        master["sku"]
    ) == set(
        data["products"]["sku"]
    )

    print(
        "✓ All products preserved"
    )

    # -----------------------------------------------------
    # SUPPLIERS PRESENT
    # -----------------------------------------------------

    assert (
        master[
            "supplier_name"
        ].notna()
    ).all()

    print(
        "✓ Supplier data present "
        "for every SKU"
    )

    # -----------------------------------------------------
    # INVENTORY PRESENT
    # -----------------------------------------------------

    assert (
        master[
            "on_hand"
        ].notna()
    ).all()

    print(
        "✓ Inventory data present "
        "for every SKU"
    )

    # -----------------------------------------------------
    # DEMAND HISTORY
    # -----------------------------------------------------

    assert (
        master[
            "history_weeks"
        ] > 0
    ).all()

    print(
        "✓ Demand history present "
        "for every SKU"
    )

    # -----------------------------------------------------
    # OPEN PO RECONCILIATION
    # -----------------------------------------------------

    master_open_qty = (
        master[
            "total_open_po_qty"
        ].sum()
    )

    source_open_qty = (
        data[
            "open_pos"
        ][
            "open_qty"
        ].sum()
    )

    assert (
        master_open_qty
        ==
        source_open_qty
    ), (
        "Open PO quantities do not "
        "reconcile."
    )

    print(
        f"✓ Open PO units reconcile: "
        f"{master_open_qty:,.0f}"
    )

    # -----------------------------------------------------
    # INVENTORY VALUE RECONCILIATION
    # -----------------------------------------------------

    master_inventory_value = (
        master[
            "inventory_value"
        ].sum()
    )

    source_inventory_value = (
        data[
            "inventory"
        ][
            "inventory_value"
        ].sum()
    )

    assert np.isclose(
        master_inventory_value,
        source_inventory_value
    )

    print(
        "✓ Inventory value reconciles: "
        f"${master_inventory_value:,.2f}"
    )

    # -----------------------------------------------------
    # NEGATIVE OPEN SUPPLY
    # -----------------------------------------------------

    assert (
        master[
            "total_open_po_qty"
        ] >= 0
    ).all()

    print(
        "✓ No negative open supply"
    )

    # -----------------------------------------------------
    # INVENTORY POSITION
    # -----------------------------------------------------

    assert (
        master[
            "inventory_position"
        ]
        ==
        master[
            "net_available"
        ]
        +
        master[
            "total_open_po_qty"
        ]
    ).all()

    print(
        "✓ Inventory position calculation valid"
    )

    # -----------------------------------------------------
    # CURRENT WOS
    # -----------------------------------------------------

    valid_wos = master[
        master[
            "13w_avg_demand"
        ] > 0
    ]

    assert (
        valid_wos[
            "current_weeks_of_supply"
        ] >= 0
    ).all()

    print(
        "✓ Current weeks-of-supply "
        "calculation valid"
    )

    print(
        "\nAll planning-master "
        "validation checks passed."
    )


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    master
):

    print("\n" + "=" * 80)
    print("PLANNING MASTER SUMMARY")
    print("=" * 80)

    print(
        f"\nSKUs: "
        f"{len(master):,}"
    )

    print(
        f"Suppliers: "
        f"{master['supplier_id'].nunique():,}"
    )

    print(
        f"Categories: "
        f"{master['category'].nunique():,}"
    )

    print(
        f"Total inventory value: "
        f"${master['inventory_value'].sum():,.2f}"
    )

    print(
        f"Net available units: "
        f"{master['net_available'].sum():,.0f}"
    )

    print(
        f"Open PO units: "
        f"{master['total_open_po_qty'].sum():,.0f}"
    )

    print(
        f"SKUs with open POs: "
        f"{master['has_open_po'].sum():,.0f}"
    )

    print(
        f"SKUs with overdue POs: "
        f"{master['has_overdue_po'].sum():,.0f}"
    )

    print(
        f"SKUs with partial POs: "
        f"{master['has_partial_po'].sum():,.0f}"
    )

    print(
        f"Currently stocked-out SKUs: "
        f"{master['currently_stocked_out'].sum():,.0f}"
    )

    print(
        "\nDemand profile distribution:"
    )

    print(
        master[
            "demand_profile"
        ].value_counts()
    )


# =========================================================
# PRINT SAMPLE RISKS
# =========================================================

def print_examples(
    master
):

    print("\n" + "=" * 100)
    print("EXAMPLE INTEGRATED SKU RECORDS")
    print("=" * 100)

    example_columns = [
        "sku",
        "supplier_id",
        "demand_profile",
        "13w_avg_demand",
        "net_available",
        "current_weeks_of_supply",
        "total_open_po_qty",
        "next_expected_receipt_date",
        "has_overdue_po",
        "supplier_avg_full_delay_days",
        "supplier_partial_shipment_rate",
    ]

    examples = (
        master
        .sort_values(
            "current_weeks_of_supply",
            ascending=True,
            na_position="last"
        )
        .head(10)
    )

    print(
        examples[
            example_columns
        ].round(2).to_string(
            index=False
        )
    )


# =========================================================
# SAVE
# =========================================================

def save_master(
    master
):

    master.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nCreated: "
        f"{OUTPUT_FILE}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)

    print(
        "PROCUREMENT AI - "
        "PLANNING MASTER BUILDER"
    )

    print("=" * 70)

    data = load_data()

    data = prepare_dates(
        data
    )

    demand_features = (
        calculate_demand_features(
            data["sales"]
        )
    )

    supplier_features = (
        calculate_supplier_features(
            data["historical_pos"]
        )
    )

    sku_procurement_features = (
        calculate_sku_procurement_features(
            data["historical_pos"]
        )
    )

    open_po_features = (
        calculate_open_po_features(
            data["open_pos"]
        )
    )

    master = (
        build_planning_master(
            data,
            demand_features,
            supplier_features,
            sku_procurement_features,
            open_po_features,
        )
    )

    master = (
        add_master_features(
            master
        )
    )

    validate_master(
        master,
        data
    )

    save_master(
        master
    )

    print_summary(
        master
    )

    print_examples(
        master
    )

    print("\n" + "=" * 70)

    print(
        "PHASE 3C COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()