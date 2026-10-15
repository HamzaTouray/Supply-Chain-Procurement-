from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

DAYS_PER_WEEK = 7


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

SUPPLIERS_FILE = (
    PROCESSED_DATA_DIR
    / "suppliers_clean.csv"
)

PURCHASE_ORDERS_FILE = (
    PROCESSED_DATA_DIR
    / "purchase_orders_clean.csv"
)

RECEIPTS_FILE = (
    PROCESSED_DATA_DIR
    / "receipts_clean.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "supplier_performance.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\nLoading Phase 5A inputs...")

    required_files = {
        "suppliers": SUPPLIERS_FILE,
        "purchase_orders": PURCHASE_ORDERS_FILE,
        "receipts": RECEIPTS_FILE,
    }

    data = {}

    for name, file_path in required_files.items():

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing required file: {file_path}"
            )

        data[name] = pd.read_csv(
            file_path
        )

        print(
            f"✓ {name:<22}"
            f"{len(data[name]):>7,} rows"
        )

    # -----------------------------------------------------
    # Convert date columns
    # -----------------------------------------------------

    po_date_columns = [
        "order_date",
        "promised_date",
        "first_receipt_date",
        "final_receipt_date",
    ]

    for column in po_date_columns:

        if column in data["purchase_orders"].columns:

            data["purchase_orders"][column] = (
                pd.to_datetime(
                    data["purchase_orders"][column]
                )
            )

    if "receipt_date" in data["receipts"].columns:

        data["receipts"]["receipt_date"] = (
            pd.to_datetime(
                data["receipts"]["receipt_date"]
            )
        )

    return data


# =========================================================
# PREPARE HISTORICAL PO FEATURES
# =========================================================

def prepare_po_features(
    purchase_orders
):

    print(
        "\nBuilding PO-level performance features..."
    )

    po = purchase_orders.copy()

    # -----------------------------------------------------
    # Lead times
    # -----------------------------------------------------

    po[
        "quoted_lead_time_days"
    ] = (
        po[
            "promised_date"
        ]
        -
        po[
            "order_date"
        ]
    ).dt.days

    po[
        "actual_first_lead_time_days"
    ] = (
        po[
            "first_receipt_date"
        ]
        -
        po[
            "order_date"
        ]
    ).dt.days

    po[
        "actual_full_lead_time_days"
    ] = (
        po[
            "final_receipt_date"
        ]
        -
        po[
            "order_date"
        ]
    ).dt.days

    # -----------------------------------------------------
    # Delays relative to promise
    # -----------------------------------------------------

    po[
        "first_receipt_delay_days"
    ] = (
        po[
            "first_receipt_date"
        ]
        -
        po[
            "promised_date"
        ]
    ).dt.days

    po[
        "full_receipt_delay_days"
    ] = (
        po[
            "final_receipt_date"
        ]
        -
        po[
            "promised_date"
        ]
    ).dt.days

    # -----------------------------------------------------
    # On-time indicators
    #
    # Early delivery counts as on-time.
    # -----------------------------------------------------

    po[
        "first_receipt_on_time"
    ] = (
        po[
            "first_receipt_delay_days"
        ] <= 0
    )

    po[
        "full_receipt_on_time"
    ] = (
        po[
            "full_receipt_delay_days"
        ] <= 0
    )

    # -----------------------------------------------------
    # Late indicators
    # -----------------------------------------------------

    po[
        "first_receipt_late"
    ] = (
        po[
            "first_receipt_delay_days"
        ] > 0
    )

    po[
        "full_receipt_late"
    ] = (
        po[
            "full_receipt_delay_days"
        ] > 0
    )

    # -----------------------------------------------------
    # Partial shipment indicator
    # -----------------------------------------------------

    po[
        "partial_shipment"
    ] = (
        po[
            "shipment_count"
        ] > 1
    )

    # -----------------------------------------------------
    # Fulfillment span
    #
    # How long from the first received quantity until the
    # PO was fully received?
    # -----------------------------------------------------

    po[
        "fulfillment_span_days"
    ] = (
        po[
            "final_receipt_date"
        ]
        -
        po[
            "first_receipt_date"
        ]
    ).dt.days

    # -----------------------------------------------------
    # Extra time beyond quoted lead time
    # -----------------------------------------------------

    po[
        "full_lead_time_overrun_days"
    ] = (
        po[
            "actual_full_lead_time_days"
        ]
        -
        po[
            "quoted_lead_time_days"
        ]
    )

    # -----------------------------------------------------
    # Lead-time ratio
    #
    # 1.00 = exactly quoted lead time
    # 1.25 = took 25% longer than quoted
    # -----------------------------------------------------

    po[
        "full_lead_time_ratio"
    ] = np.where(
        po[
            "quoted_lead_time_days"
        ] > 0,

        po[
            "actual_full_lead_time_days"
        ]
        /
        po[
            "quoted_lead_time_days"
        ],

        np.nan
    )

    print(
        f"✓ PO performance records: "
        f"{len(po):,}"
    )

    return po


# =========================================================
# RECEIPT-LEVEL FEATURES
# =========================================================

def build_receipt_features(
    purchase_orders,
    receipts
):

    print(
        "Building receipt-level fulfillment features..."
    )

    po_columns = [
        "po_number",
        "supplier_id",
        "ordered_qty",
        "order_date",
        "promised_date",
    ]

    receipt_detail = (
        receipts
        .merge(
            purchase_orders[
                po_columns
            ],
            on="po_number",
            how="left",
            validate="many_to_one",
            suffixes=("", "_po")
        )
    )

    # -----------------------------------------------------
    # If receipt data already contains supplier_id,
    # prefer the PO's supplier_id as the authoritative
    # relationship.
    # -----------------------------------------------------

    if "supplier_id_po" in receipt_detail.columns:

        receipt_detail[
            "supplier_id"
        ] = receipt_detail[
            "supplier_id_po"
        ]

    # -----------------------------------------------------
    # Receipt timing relative to PO date
    # -----------------------------------------------------

    receipt_detail[
        "receipt_lead_time_days"
    ] = (
        receipt_detail[
            "receipt_date"
        ]
        -
        receipt_detail[
            "order_date"
        ]
    ).dt.days

    # -----------------------------------------------------
    # Receipt timing relative to promised date
    # -----------------------------------------------------

    receipt_detail[
        "receipt_delay_days"
    ] = (
        receipt_detail[
            "receipt_date"
        ]
        -
        receipt_detail[
            "promised_date"
        ]
    ).dt.days

    # -----------------------------------------------------
    # Quantity share of PO
    # -----------------------------------------------------

    receipt_detail[
        "receipt_qty_share"
    ] = np.where(
        receipt_detail[
            "ordered_qty"
        ] > 0,

        receipt_detail[
            "receipt_qty"
        ]
        /
        receipt_detail[
            "ordered_qty"
        ],

        0
    )

    return receipt_detail


# =========================================================
# QUANTITY-WEIGHTED RECEIPT TIMING
# =========================================================

def calculate_weighted_receipt_timing(
    receipt_detail
):

    print(
        "Calculating quantity-weighted receipt timing..."
    )

    records = []

    for po_number, group in (
        receipt_detail.groupby(
            "po_number"
        )
    ):

        total_qty = (
            group[
                "receipt_qty"
            ].sum()
        )

        if total_qty <= 0:

            weighted_lead = np.nan
            weighted_delay = np.nan

        else:

            weighted_lead = (
                (
                    group[
                        "receipt_lead_time_days"
                    ]
                    *
                    group[
                        "receipt_qty"
                    ]
                ).sum()
                /
                total_qty
            )

            weighted_delay = (
                (
                    group[
                        "receipt_delay_days"
                    ]
                    *
                    group[
                        "receipt_qty"
                    ]
                ).sum()
                /
                total_qty
            )

        records.append(
            {
                "po_number": po_number,

                "qty_weighted_receipt_lead_days": (
                    weighted_lead
                ),

                "qty_weighted_receipt_delay_days": (
                    weighted_delay
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# FIRST RECEIPT QUANTITY
# =========================================================

def calculate_first_receipt_features(
    receipt_detail
):

    print(
        "Calculating first-receipt quantity features..."
    )

    ordered = (
        receipt_detail
        .sort_values(
            [
                "po_number",
                "receipt_date",
                "receipt_number",
            ]
        )
    )

    first_receipts = (
        ordered
        .groupby(
            "po_number",
            as_index=False
        )
        .first()
    )

    first_receipts[
        "first_receipt_fill_rate"
    ] = np.where(
        first_receipts[
            "ordered_qty"
        ] > 0,

        first_receipts[
            "receipt_qty"
        ]
        /
        first_receipts[
            "ordered_qty"
        ],

        np.nan
    )

    return first_receipts[
        [
            "po_number",
            "receipt_qty",
            "first_receipt_fill_rate",
        ]
    ].rename(
        columns={
            "receipt_qty":
                "first_receipt_qty"
        }
    )


# =========================================================
# ADD RECEIPT FEATURES TO PO TABLE
# =========================================================

def enrich_po_table(
    po,
    weighted_timing,
    first_receipts
):

    print(
        "Combining PO and receipt intelligence..."
    )

    result = (
        po
        .merge(
            weighted_timing,
            on="po_number",
            how="left",
            validate="one_to_one"
        )
        .merge(
            first_receipts,
            on="po_number",
            how="left",
            validate="one_to_one"
        )
    )

    return result


# =========================================================
# SUPPLIER AGGREGATION
# =========================================================

def build_supplier_metrics(
    suppliers,
    po
):

    print(
        "Aggregating supplier performance..."
    )

    metrics = (
        po
        .groupby(
            "supplier_id"
        )
        .agg(

            po_count=(
                "po_number",
                "count"
            ),

            total_units_ordered=(
                "ordered_qty",
                "sum"
            ),

            avg_quoted_lead_time_days=(
                "quoted_lead_time_days",
                "mean"
            ),

            avg_first_lead_time_days=(
                "actual_first_lead_time_days",
                "mean"
            ),

            avg_full_lead_time_days=(
                "actual_full_lead_time_days",
                "mean"
            ),

            median_full_lead_time_days=(
                "actual_full_lead_time_days",
                "median"
            ),

            std_full_lead_time_days=(
                "actual_full_lead_time_days",
                "std"
            ),

            avg_first_delay_days=(
                "first_receipt_delay_days",
                "mean"
            ),

            avg_full_delay_days=(
                "full_receipt_delay_days",
                "mean"
            ),

            median_full_delay_days=(
                "full_receipt_delay_days",
                "median"
            ),

            std_full_delay_days=(
                "full_receipt_delay_days",
                "std"
            ),

            first_receipt_on_time_rate=(
                "first_receipt_on_time",
                "mean"
            ),

            full_po_on_time_rate=(
                "full_receipt_on_time",
                "mean"
            ),

            first_receipt_late_rate=(
                "first_receipt_late",
                "mean"
            ),

            full_po_late_rate=(
                "full_receipt_late",
                "mean"
            ),

            partial_shipment_rate=(
                "partial_shipment",
                "mean"
            ),

            avg_shipments_per_po=(
                "shipment_count",
                "mean"
            ),

            max_shipments_per_po=(
                "shipment_count",
                "max"
            ),

            avg_fulfillment_span_days=(
                "fulfillment_span_days",
                "mean"
            ),

            max_fulfillment_span_days=(
                "fulfillment_span_days",
                "max"
            ),

            avg_first_receipt_fill_rate=(
                "first_receipt_fill_rate",
                "mean"
            ),

            avg_qty_weighted_receipt_lead_days=(
                "qty_weighted_receipt_lead_days",
                "mean"
            ),

            avg_qty_weighted_receipt_delay_days=(
                "qty_weighted_receipt_delay_days",
                "mean"
            ),

            avg_full_lead_time_ratio=(
                "full_lead_time_ratio",
                "mean"
            ),

            avg_full_lead_time_overrun_days=(
                "full_lead_time_overrun_days",
                "mean"
            ),
        )
        .reset_index()
    )

    # -----------------------------------------------------
    # Percentile metrics
    #
    # These are extremely useful later for planning.
    # Average lead time alone does not protect us against
    # suppliers with long-tail delays.
    # -----------------------------------------------------

    percentile_metrics = (
        po
        .groupby(
            "supplier_id"
        )
        .agg(

            p75_full_lead_time_days=(
                "actual_full_lead_time_days",
                lambda x: x.quantile(0.75)
            ),

            p90_full_lead_time_days=(
                "actual_full_lead_time_days",
                lambda x: x.quantile(0.90)
            ),

            p95_full_lead_time_days=(
                "actual_full_lead_time_days",
                lambda x: x.quantile(0.95)
            ),

            p90_full_delay_days=(
                "full_receipt_delay_days",
                lambda x: x.quantile(0.90)
            ),

            p95_full_delay_days=(
                "full_receipt_delay_days",
                lambda x: x.quantile(0.95)
            ),

            p90_fulfillment_span_days=(
                "fulfillment_span_days",
                lambda x: x.quantile(0.90)
            ),
        )
        .reset_index()
    )

    metrics = (
        metrics
        .merge(
            percentile_metrics,
            on="supplier_id",
            how="left",
            validate="one_to_one"
        )
    )

    # -----------------------------------------------------
    # Lead-time variability
    # -----------------------------------------------------

    metrics[
        "full_lead_time_cv"
    ] = np.where(
        metrics[
            "avg_full_lead_time_days"
        ] > 0,

        metrics[
            "std_full_lead_time_days"
        ]
        /
        metrics[
            "avg_full_lead_time_days"
        ],

        np.nan
    )

    # -----------------------------------------------------
    # Merge supplier master data
    # -----------------------------------------------------

    result = (
        suppliers
        .merge(
            metrics,
            on="supplier_id",
            how="left",
            validate="one_to_one"
        )
    )

    return result


# =========================================================
# ADD BASIC PERFORMANCE FLAGS
# =========================================================

def add_performance_flags(
    supplier_metrics
):

    print(
        "Adding supplier performance flags..."
    )

    result = supplier_metrics.copy()

    # These are descriptive flags only.
    # Phase 5C will build the actual risk classification.

    result[
        "frequent_late_flag"
    ] = (
        result[
            "full_po_late_rate"
        ] >= 0.50
    )

    result[
        "frequent_partial_flag"
    ] = (
        result[
            "partial_shipment_rate"
        ] >= 0.40
    )

    result[
        "high_lead_time_variability_flag"
    ] = (
        result[
            "full_lead_time_cv"
        ] >= 0.20
    )

    result[
        "long_fulfillment_tail_flag"
    ] = (
        result[
            "p90_full_lead_time_days"
        ]
        >
        (
            result[
                "avg_quoted_lead_time_days"
            ]
            + 14
        )
    )

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate_output(
    result,
    suppliers,
    po
):

    print("\n" + "=" * 85)
    print("SUPPLIER PERFORMANCE VALIDATION")
    print("=" * 85)

    assert len(
        result
    ) == len(
        suppliers
    )

    print(
        f"✓ Exactly "
        f"{len(result)} supplier rows"
    )

    assert (
        result[
            "supplier_id"
        ].is_unique
    )

    print(
        "✓ One row per supplier"
    )

    assert set(
        result[
            "supplier_id"
        ]
    ) == set(
        suppliers[
            "supplier_id"
        ]
    )

    print(
        "✓ All suppliers preserved"
    )

    assert (
        result[
            "po_count"
        ].sum()
        ==
        len(po)
    )

    print(
        f"✓ Historical PO count reconciles: "
        f"{len(po):,}"
    )

    assert (
        (
            result[
                "first_receipt_on_time_rate"
            ] >= 0
        )
        &
        (
            result[
                "first_receipt_on_time_rate"
            ] <= 1
        )
    ).all()

    print(
        "✓ First-receipt on-time rates valid"
    )

    assert (
        (
            result[
                "full_po_on_time_rate"
            ] >= 0
        )
        &
        (
            result[
                "full_po_on_time_rate"
            ] <= 1
        )
    ).all()

    print(
        "✓ Full-PO on-time rates valid"
    )

    assert (
        (
            result[
                "partial_shipment_rate"
            ] >= 0
        )
        &
        (
            result[
                "partial_shipment_rate"
            ] <= 1
        )
    ).all()

    print(
        "✓ Partial-shipment rates valid"
    )

    assert (
        (
            result[
                "avg_first_receipt_fill_rate"
            ] >= 0
        )
        &
        (
            result[
                "avg_first_receipt_fill_rate"
            ] <= 1
        )
    ).all()

    print(
        "✓ First-receipt fill rates valid"
    )

    assert (
        result[
            "avg_full_lead_time_days"
        ] >= 0
    ).all()

    print(
        "✓ Full lead times non-negative"
    )

    assert (
        result[
            "avg_fulfillment_span_days"
        ] >= 0
    ).all()

    print(
        "✓ Fulfillment spans non-negative"
    )

    assert (
        result[
            "p90_full_lead_time_days"
        ]
        >=
        result[
            "median_full_lead_time_days"
        ]
    ).all()

    print(
        "✓ P90 lead times >= median lead times"
    )

    # -----------------------------------------------------
    # Expected synthetic behavior validation
    # -----------------------------------------------------

    partial_suppliers = result[
        result[
            "behavior_type"
        ].isin(
            [
                "partial_shipments",
                "partial_and_late",
            ]
        )
    ]

    other_suppliers = result[
        ~result[
            "behavior_type"
        ].isin(
            [
                "partial_shipments",
                "partial_and_late",
            ]
        )
    ]

    assert (
        partial_suppliers[
            "partial_shipment_rate"
        ].mean()
        >
        other_suppliers[
            "partial_shipment_rate"
        ].mean()
    )

    print(
        "✓ Partial-shipment suppliers show "
        "higher fragmentation"
    )

    late_suppliers = result[
        result[
            "behavior_type"
        ].isin(
            [
                "frequently_late",
                "partial_and_late",
            ]
        )
    ]

    reliable_suppliers = result[
        result[
            "behavior_type"
        ].isin(
            [
                "excellent",
                "reliable",
            ]
        )
    ]

    assert (
        late_suppliers[
            "avg_full_delay_days"
        ].mean()
        >
        reliable_suppliers[
            "avg_full_delay_days"
        ].mean()
    )

    print(
        "✓ Late suppliers show greater "
        "full-delivery delays"
    )

    print(
        "\nAll Phase 5A validation "
        "checks passed."
    )


# =========================================================
# PRINT PERFORMANCE SUMMARY
# =========================================================

def print_performance_summary(
    result
):

    print("\n" + "=" * 145)
    print("SUPPLIER PERFORMANCE SUMMARY")
    print("=" * 145)

    columns = [
        "supplier_id",
        "supplier_name",
        "behavior_type",
        "po_count",
        "avg_quoted_lead_time_days",
        "avg_first_lead_time_days",
        "avg_full_lead_time_days",
        "p90_full_lead_time_days",
        "avg_first_delay_days",
        "avg_full_delay_days",
        "full_po_on_time_rate",
        "partial_shipment_rate",
        "avg_shipments_per_po",
        "avg_first_receipt_fill_rate",
        "avg_fulfillment_span_days",
    ]

    print(
        result[
            columns
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


# =========================================================
# PRINT FIRST VS FULL RECEIPT
# =========================================================

def print_first_vs_full_summary(
    result
):

    print("\n" + "=" * 110)
    print("FIRST RECEIPT VS FULL FULFILLMENT")
    print("=" * 110)

    summary = result[
        [
            "supplier_id",
            "behavior_type",
            "avg_first_lead_time_days",
            "avg_full_lead_time_days",
            "avg_fulfillment_span_days",
            "avg_first_receipt_fill_rate",
            "partial_shipment_rate",
        ]
    ].copy()

    summary[
        "first_to_full_gap_days"
    ] = (
        summary[
            "avg_full_lead_time_days"
        ]
        -
        summary[
            "avg_first_lead_time_days"
        ]
    )

    print(
        summary
        .round(3)
        .to_string(
            index=False
        )
    )


# =========================================================
# PRINT LONG-TAIL SUMMARY
# =========================================================

def print_tail_summary(
    result
):

    print("\n" + "=" * 115)
    print("SUPPLIER LEAD-TIME TAIL RISK")
    print("=" * 115)

    columns = [
        "supplier_id",
        "behavior_type",
        "avg_quoted_lead_time_days",
        "avg_full_lead_time_days",
        "p75_full_lead_time_days",
        "p90_full_lead_time_days",
        "p95_full_lead_time_days",
        "full_lead_time_cv",
        "long_fulfillment_tail_flag",
    ]

    print(
        result[
            columns
        ]
        .sort_values(
            "p90_full_lead_time_days",
            ascending=False
        )
        .round(3)
        .to_string(
            index=False
        )
    )


# =========================================================
# SAVE
# =========================================================

def save_output(
    result
):

    result.to_csv(
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

    print("=" * 85)

    print(
        "PROCUREMENT AI - "
        "SUPPLIER PERFORMANCE INTELLIGENCE"
    )

    print("=" * 85)

    data = load_data()

    po = prepare_po_features(
        data[
            "purchase_orders"
        ]
    )

    receipt_detail = (
        build_receipt_features(
            data[
                "purchase_orders"
            ],
            data[
                "receipts"
            ]
        )
    )

    weighted_timing = (
        calculate_weighted_receipt_timing(
            receipt_detail
        )
    )

    first_receipts = (
        calculate_first_receipt_features(
            receipt_detail
        )
    )

    po = enrich_po_table(
        po,
        weighted_timing,
        first_receipts
    )

    supplier_metrics = (
        build_supplier_metrics(
            data[
                "suppliers"
            ],
            po
        )
    )

    result = (
        add_performance_flags(
            supplier_metrics
        )
    )

    validate_output(
        result,
        data[
            "suppliers"
        ],
        po
    )

    save_output(
        result
    )

    print_performance_summary(
        result
    )

    print_first_vs_full_summary(
        result
    )

    print_tail_summary(
        result
    )

    print("\n" + "=" * 85)

    print(
        "PHASE 5A COMPLETE"
    )

    print("=" * 85)


if __name__ == "__main__":
    main()