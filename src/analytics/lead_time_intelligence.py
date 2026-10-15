from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

DAYS_PER_WEEK = 7

# Minimum historical PO volume used for confidence.
HIGH_CONFIDENCE_PO_COUNT = 100
MEDIUM_CONFIDENCE_PO_COUNT = 50

# Fulfillment behavior thresholds.
HIGH_PARTIAL_RATE = 0.50
MODERATE_PARTIAL_RATE = 0.20

LOW_FIRST_FILL_RATE = 0.70
MODERATE_FIRST_FILL_RATE = 0.90

HIGH_VARIABILITY_CV = 0.25
MODERATE_VARIABILITY_CV = 0.15


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

SUPPLIER_PERFORMANCE_FILE = (
    PROCESSED_DATA_DIR
    / "supplier_performance.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "supplier_lead_time_intelligence.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\nLoading Phase 5B inputs...")

    if not SUPPLIER_PERFORMANCE_FILE.exists():

        raise FileNotFoundError(
            f"Missing required file: "
            f"{SUPPLIER_PERFORMANCE_FILE}"
        )

    df = pd.read_csv(
        SUPPLIER_PERFORMANCE_FILE
    )

    print(
        f"✓ supplier_performance"
        f"{len(df):>12,} rows"
    )

    return df


# =========================================================
# VALIDATE INPUT SCHEMA
# =========================================================

def validate_input_schema(df):

    required_columns = [
        "supplier_id",
        "supplier_name",
        "behavior_type",
        "po_count",
        "avg_quoted_lead_time_days",
        "avg_first_lead_time_days",
        "avg_full_lead_time_days",
        "median_full_lead_time_days",
        "std_full_lead_time_days",
        "p75_full_lead_time_days",
        "p90_full_lead_time_days",
        "p95_full_lead_time_days",
        "avg_qty_weighted_receipt_lead_days",
        "avg_first_delay_days",
        "avg_full_delay_days",
        "full_po_on_time_rate",
        "partial_shipment_rate",
        "avg_shipments_per_po",
        "avg_first_receipt_fill_rate",
        "avg_fulfillment_span_days",
        "p90_fulfillment_span_days",
        "full_lead_time_cv",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "supplier_performance.csv is missing "
            f"required columns: {missing}"
        )

    print(
        "✓ Required supplier-performance "
        "columns present"
    )


# =========================================================
# BASIC LEAD-TIME GAPS
# =========================================================

def add_lead_time_gaps(df):

    print(
        "Calculating lead-time gaps..."
    )

    result = df.copy()

    result[
        "quoted_lead_time_weeks"
    ] = (
        result[
            "avg_quoted_lead_time_days"
        ]
        / DAYS_PER_WEEK
    )

    result[
        "avg_first_lead_time_weeks"
    ] = (
        result[
            "avg_first_lead_time_days"
        ]
        / DAYS_PER_WEEK
    )

    result[
        "avg_full_lead_time_weeks"
    ] = (
        result[
            "avg_full_lead_time_days"
        ]
        / DAYS_PER_WEEK
    )

    result[
        "p90_full_lead_time_weeks"
    ] = (
        result[
            "p90_full_lead_time_days"
        ]
        / DAYS_PER_WEEK
    )

    result[
        "p95_full_lead_time_weeks"
    ] = (
        result[
            "p95_full_lead_time_days"
        ]
        / DAYS_PER_WEEK
    )

    # Difference between first supply and full completion.
    result[
        "first_to_full_gap_days"
    ] = (
        result[
            "avg_full_lead_time_days"
        ]
        -
        result[
            "avg_first_lead_time_days"
        ]
    )

    # How much longer full fulfillment takes than quoted.
    result[
        "full_vs_quoted_gap_days"
    ] = (
        result[
            "avg_full_lead_time_days"
        ]
        -
        result[
            "avg_quoted_lead_time_days"
        ]
    )

    # Same concept as a percentage.
    result[
        "full_vs_quoted_gap_pct"
    ] = np.where(
        result[
            "avg_quoted_lead_time_days"
        ] > 0,

        result[
            "full_vs_quoted_gap_days"
        ]
        /
        result[
            "avg_quoted_lead_time_days"
        ],

        np.nan
    )

    # P90 exposure above quoted lead time.
    result[
        "p90_vs_quoted_gap_days"
    ] = (
        result[
            "p90_full_lead_time_days"
        ]
        -
        result[
            "avg_quoted_lead_time_days"
        ]
    )

    return result


# =========================================================
# FULFILLMENT PATTERN
# =========================================================

def classify_fulfillment_pattern(row):

    partial_rate = row[
        "partial_shipment_rate"
    ]

    first_fill = row[
        "avg_first_receipt_fill_rate"
    ]

    fulfillment_gap = row[
        "first_to_full_gap_days"
    ]

    if (
        partial_rate >= HIGH_PARTIAL_RATE
        and first_fill < LOW_FIRST_FILL_RATE
    ):
        return "highly_fragmented"

    if (
        partial_rate >= HIGH_PARTIAL_RATE
        or fulfillment_gap >= 14
    ):
        return "fragmented"

    if (
        partial_rate >= MODERATE_PARTIAL_RATE
        or first_fill < MODERATE_FIRST_FILL_RATE
    ):
        return "moderately_fragmented"

    return "mostly_complete"


def add_fulfillment_pattern(df):

    print(
        "Classifying fulfillment patterns..."
    )

    result = df.copy()

    result[
        "fulfillment_pattern"
    ] = result.apply(
        classify_fulfillment_pattern,
        axis=1
    )

    return result


# =========================================================
# LEAD-TIME STABILITY
# =========================================================

def classify_lead_time_stability(row):

    cv = row[
        "full_lead_time_cv"
    ]

    if pd.isna(cv):
        return "unknown"

    if cv >= HIGH_VARIABILITY_CV:
        return "high_variability"

    if cv >= MODERATE_VARIABILITY_CV:
        return "moderate_variability"

    return "consistent"


def add_lead_time_stability(df):

    print(
        "Classifying lead-time stability..."
    )

    result = df.copy()

    result[
        "lead_time_stability"
    ] = result.apply(
        classify_lead_time_stability,
        axis=1
    )

    return result


# =========================================================
# EFFECTIVE PLANNING LEAD TIME
# =========================================================

def calculate_effective_lead_time(row):

    """
    Effective planning lead time is a behavior-aware
    baseline for later inventory planning.

    It is intentionally NOT the final reorder-point lead
    time. Phase 7 will combine this supplier intelligence
    with demand uncertainty and service-level policy.

    Logic:

    1. Begin with average full fulfillment lead time.

    2. For reliable, consistent suppliers with mostly
       complete shipments, the historical average is a
       reasonable operational baseline.

    3. For moderate uncertainty, move halfway from average
       full lead time toward P90.

    4. For highly fragmented or highly variable suppliers,
       move 75% toward P90.

    This captures supplier uncertainty without blindly
    treating every purchase order as a P90 event.
    """

    avg_full = row[
        "avg_full_lead_time_days"
    ]

    p90 = row[
        "p90_full_lead_time_days"
    ]

    pattern = row[
        "fulfillment_pattern"
    ]

    stability = row[
        "lead_time_stability"
    ]

    if pd.isna(avg_full):
        return np.nan

    if pd.isna(p90):
        return avg_full

    tail_gap = max(
        p90 - avg_full,
        0
    )

    # -----------------------------------------------------
    # Severe uncertainty
    # -----------------------------------------------------

    if (
        pattern == "highly_fragmented"
        or stability == "high_variability"
    ):

        effective = (
            avg_full
            +
            0.75 * tail_gap
        )

    # -----------------------------------------------------
    # Moderate uncertainty
    # -----------------------------------------------------

    elif (
        pattern
        in [
            "fragmented",
            "moderately_fragmented",
        ]
        or stability == "moderate_variability"
    ):

        effective = (
            avg_full
            +
            0.50 * tail_gap
        )

    # -----------------------------------------------------
    # Consistent supplier
    # -----------------------------------------------------

    else:

        effective = avg_full

    # Never plan below observed average full lead time.
    effective = max(
        effective,
        avg_full
    )

    return effective


def add_effective_lead_time(df):

    print(
        "Calculating effective planning lead time..."
    )

    result = df.copy()

    result[
        "effective_planning_lead_time_days"
    ] = result.apply(
        calculate_effective_lead_time,
        axis=1
    )

    result[
        "effective_planning_lead_time_weeks"
    ] = (
        result[
            "effective_planning_lead_time_days"
        ]
        / DAYS_PER_WEEK
    )

    result[
        "planning_lead_time_buffer_days"
    ] = (
        result[
            "effective_planning_lead_time_days"
        ]
        -
        result[
            "avg_quoted_lead_time_days"
        ]
    )

    result[
        "planning_lead_time_buffer_weeks"
    ] = (
        result[
            "planning_lead_time_buffer_days"
        ]
        / DAYS_PER_WEEK
    )

    # Buffer should not become negative.
    result[
        "planning_lead_time_buffer_days"
    ] = (
        result[
            "planning_lead_time_buffer_days"
        ]
        .clip(lower=0)
    )

    result[
        "planning_lead_time_buffer_weeks"
    ] = (
        result[
            "planning_lead_time_buffer_weeks"
        ]
        .clip(lower=0)
    )

    return result


# =========================================================
# FIRST-RECEIPT USABILITY
# =========================================================

def classify_first_receipt_value(row):

    """
    Describes whether the first shipment usually provides
    meaningful inventory relief.
    """

    fill = row[
        "avg_first_receipt_fill_rate"
    ]

    if pd.isna(fill):
        return "unknown"

    if fill >= 0.90:
        return "substantial"

    if fill >= 0.70:
        return "meaningful"

    if fill >= 0.40:
        return "partial_relief"

    return "limited_relief"


def add_first_receipt_intelligence(df):

    print(
        "Evaluating first-receipt usefulness..."
    )

    result = df.copy()

    result[
        "first_receipt_inventory_relief"
    ] = result.apply(
        classify_first_receipt_value,
        axis=1
    )

    return result


# =========================================================
# HISTORICAL CONFIDENCE
# =========================================================

def classify_history_confidence(row):

    po_count = row[
        "po_count"
    ]

    if po_count >= HIGH_CONFIDENCE_PO_COUNT:
        return "high"

    if po_count >= MEDIUM_CONFIDENCE_PO_COUNT:
        return "medium"

    return "low"


def add_history_confidence(df):

    print(
        "Assigning lead-time evidence confidence..."
    )

    result = df.copy()

    result[
        "lead_time_evidence_confidence"
    ] = result.apply(
        classify_history_confidence,
        axis=1
    )

    return result


# =========================================================
# PLANNING FLAGS
# =========================================================

def add_planning_flags(df):

    print(
        "Adding planning lead-time flags..."
    )

    result = df.copy()

    result[
        "quoted_lead_time_understates_actual_flag"
    ] = (
        result[
            "full_vs_quoted_gap_days"
        ] > 7
    )

    result[
        "significant_tail_risk_flag"
    ] = (
        result[
            "p90_vs_quoted_gap_days"
        ] > 14
    )

    result[
        "partial_receipt_dependency_flag"
    ] = (
        (
            result[
                "partial_shipment_rate"
            ] >= HIGH_PARTIAL_RATE
        )
        &
        (
            result[
                "avg_first_receipt_fill_rate"
            ] < LOW_FIRST_FILL_RATE
        )
    )

    result[
        "planning_lead_time_adjustment_required"
    ] = (
        result[
            "planning_lead_time_buffer_days"
        ] > 7
    )

    return result


# =========================================================
# PLANNING NOTE
# =========================================================

def build_planning_note(row):

    notes = []

    if row[
        "quoted_lead_time_understates_actual_flag"
    ]:
        notes.append(
            "quoted lead time understates observed fulfillment"
        )

    if row[
        "significant_tail_risk_flag"
    ]:
        notes.append(
            "material long-tail lead-time exposure"
        )

    if row[
        "partial_receipt_dependency_flag"
    ]:
        notes.append(
            "first receipt provides only partial inventory relief"
        )

    if (
        row[
            "lead_time_stability"
        ] == "high_variability"
    ):
        notes.append(
            "lead time is highly variable"
        )

    if (
        row[
            "fulfillment_pattern"
        ] == "highly_fragmented"
    ):
        notes.append(
            "orders are highly fragmented"
        )

    if not notes:
        return "historical fulfillment broadly consistent"

    return "; ".join(notes)


def add_planning_notes(df):

    print(
        "Building supplier planning notes..."
    )

    result = df.copy()

    result[
        "lead_time_planning_note"
    ] = result.apply(
        build_planning_note,
        axis=1
    )

    return result


# =========================================================
# BUILD INTELLIGENCE TABLE
# =========================================================

def build_lead_time_intelligence(df):

    result = add_lead_time_gaps(
        df
    )

    result = add_fulfillment_pattern(
        result
    )

    result = add_lead_time_stability(
        result
    )

    result = add_effective_lead_time(
        result
    )

    result = add_first_receipt_intelligence(
        result
    )

    result = add_history_confidence(
        result
    )

    result = add_planning_flags(
        result
    )

    result = add_planning_notes(
        result
    )

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate_output(
    result,
    source
):

    print("\n" + "=" * 90)
    print("LEAD-TIME INTELLIGENCE VALIDATION")
    print("=" * 90)

    assert len(result) == len(source)

    print(
        f"✓ Exactly {len(result)} supplier rows"
    )

    assert result[
        "supplier_id"
    ].is_unique

    print(
        "✓ One row per supplier"
    )

    assert set(
        result["supplier_id"]
    ) == set(
        source["supplier_id"]
    )

    print(
        "✓ All suppliers preserved"
    )

    assert (
        result[
            "avg_full_lead_time_days"
        ]
        >=
        result[
            "avg_first_lead_time_days"
        ]
    ).all()

    print(
        "✓ Full lead time >= first-receipt lead time"
    )

    assert (
        result[
            "p90_full_lead_time_days"
        ]
        >=
        result[
            "avg_full_lead_time_days"
        ]
    ).all()

    print(
        "✓ P90 lead time >= average full lead time"
    )

    assert (
        result[
            "p95_full_lead_time_days"
        ]
        >=
        result[
            "p90_full_lead_time_days"
        ]
    ).all()

    print(
        "✓ P95 lead time >= P90 lead time"
    )

    assert (
        result[
            "effective_planning_lead_time_days"
        ]
        >=
        result[
            "avg_full_lead_time_days"
        ]
    ).all()

    print(
        "✓ Effective planning lead time never "
        "below average full lead time"
    )

    assert (
        result[
            "planning_lead_time_buffer_days"
        ] >= 0
    ).all()

    print(
        "✓ Planning lead-time buffers non-negative"
    )

    valid_patterns = {
        "mostly_complete",
        "moderately_fragmented",
        "fragmented",
        "highly_fragmented",
    }

    assert set(
        result[
            "fulfillment_pattern"
        ].unique()
    ).issubset(
        valid_patterns
    )

    print(
        "✓ Fulfillment patterns valid"
    )

    valid_stability = {
        "consistent",
        "moderate_variability",
        "high_variability",
        "unknown",
    }

    assert set(
        result[
            "lead_time_stability"
        ].unique()
    ).issubset(
        valid_stability
    )

    print(
        "✓ Lead-time stability classes valid"
    )

    valid_relief = {
        "substantial",
        "meaningful",
        "partial_relief",
        "limited_relief",
        "unknown",
    }

    assert set(
        result[
            "first_receipt_inventory_relief"
        ].unique()
    ).issubset(
        valid_relief
    )

    print(
        "✓ First-receipt inventory-relief classes valid"
    )

    # Synthetic behavior checks.

    partial_late = result[
        result[
            "behavior_type"
        ] == "partial_and_late"
    ]

    excellent = result[
        result[
            "behavior_type"
        ] == "excellent"
    ]

    assert (
        partial_late[
            "effective_planning_lead_time_days"
        ].mean()
        >
        excellent[
            "effective_planning_lead_time_days"
        ].mean()
    )

    print(
        "✓ Partial-and-late supplier requires "
        "longer planning lead time than excellent suppliers"
    )

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

    assert (
        partial_suppliers[
            "first_to_full_gap_days"
        ].mean()
        >
        excellent[
            "first_to_full_gap_days"
        ].mean()
    )

    print(
        "✓ Partial suppliers show larger "
        "first-to-full fulfillment gaps"
    )

    print(
        "\nAll Phase 5B validation checks passed."
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary(result):

    print("\n" + "=" * 155)
    print("SUPPLIER LEAD-TIME INTELLIGENCE")
    print("=" * 155)

    columns = [
        "supplier_id",
        "behavior_type",
        "avg_quoted_lead_time_days",
        "avg_first_lead_time_days",
        "avg_full_lead_time_days",
        "avg_qty_weighted_receipt_lead_days",
        "p90_full_lead_time_days",
        "p95_full_lead_time_days",
        "effective_planning_lead_time_days",
        "planning_lead_time_buffer_days",
        "fulfillment_pattern",
        "lead_time_stability",
    ]

    print(
        result[
            columns
        ]
        .round(2)
        .to_string(
            index=False
        )
    )


# =========================================================
# FULFILLMENT SUMMARY
# =========================================================

def print_fulfillment_summary(result):

    print("\n" + "=" * 135)
    print("FULFILLMENT INTELLIGENCE")
    print("=" * 135)

    columns = [
        "supplier_id",
        "behavior_type",
        "partial_shipment_rate",
        "avg_shipments_per_po",
        "avg_first_receipt_fill_rate",
        "first_to_full_gap_days",
        "first_receipt_inventory_relief",
        "fulfillment_pattern",
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
# PLANNING FLAGS SUMMARY
# =========================================================

def print_planning_flags(result):

    print("\n" + "=" * 120)
    print("SUPPLIER PLANNING FLAGS")
    print("=" * 120)

    columns = [
        "supplier_id",
        "quoted_lead_time_understates_actual_flag",
        "significant_tail_risk_flag",
        "partial_receipt_dependency_flag",
        "planning_lead_time_adjustment_required",
        "lead_time_evidence_confidence",
        "lead_time_planning_note",
    ]

    print(
        result[
            columns
        ]
        .to_string(
            index=False
        )
    )


# =========================================================
# SAVE
# =========================================================

def save_output(result):

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nCreated: {OUTPUT_FILE}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 90)

    print(
        "PROCUREMENT AI - "
        "LEAD-TIME & FULFILLMENT INTELLIGENCE"
    )

    print("=" * 90)

    supplier_performance = load_data()

    validate_input_schema(
        supplier_performance
    )

    result = build_lead_time_intelligence(
        supplier_performance
    )

    validate_output(
        result,
        supplier_performance
    )

    save_output(
        result
    )

    print_summary(
        result
    )

    print_fulfillment_summary(
        result
    )

    print_planning_flags(
        result
    )

    print("\n" + "=" * 90)

    print(
        "PHASE 5B COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()