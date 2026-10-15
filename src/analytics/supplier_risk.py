from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "supplier_lead_time_intelligence.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "supplier_risk.csv"
)


# =========================================================
# SCORE WEIGHTS
# =========================================================
#
# Total = 100 points
#
# Higher score = higher procurement / fulfillment risk.
#
# We keep each dimension separate so the future AI
# assistant can explain WHY a supplier is risky.
# =========================================================

WEIGHT_DELIVERY = 25
WEIGHT_LEAD_TIME = 20
WEIGHT_FRAGMENTATION = 20
WEIGHT_TAIL = 15
WEIGHT_FIRST_RECEIPT = 10
WEIGHT_VARIABILITY = 10


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\nLoading Phase 5C inputs...")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing required file: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"✓ supplier_lead_time_intelligence"
        f"{len(df):>8,} rows"
    )

    return df


# =========================================================
# VALIDATE INPUT
# =========================================================

def validate_input_schema(df):

    required_columns = [
        "supplier_id",
        "supplier_name",
        "behavior_type",
        "po_count",
        "avg_quoted_lead_time_days",
        "avg_full_lead_time_days",
        "p90_full_lead_time_days",
        "full_po_on_time_rate",
        "partial_shipment_rate",
        "avg_first_receipt_fill_rate",
        "first_to_full_gap_days",
        "full_lead_time_cv",
        "effective_planning_lead_time_days",
        "planning_lead_time_buffer_days",
        "fulfillment_pattern",
        "lead_time_stability",
        "lead_time_evidence_confidence",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "supplier_lead_time_intelligence.csv "
            f"is missing required columns: {missing}"
        )

    print(
        "✓ Required lead-time intelligence "
        "columns present"
    )


# =========================================================
# HELPER
# =========================================================

def clip_score(value, maximum):

    if pd.isna(value):
        return 0.0

    return float(
        np.clip(
            value,
            0,
            maximum
        )
    )


# =========================================================
# 1. DELIVERY RELIABILITY RISK
# =========================================================

def score_delivery_reliability(row):

    """
    Uses full-PO on-time performance.

    100% on time -> 0 risk points
    0% on time   -> full 25 risk points
    """

    on_time_rate = row[
        "full_po_on_time_rate"
    ]

    if pd.isna(on_time_rate):
        return 0.0

    risk = (
        1 - on_time_rate
    ) * WEIGHT_DELIVERY

    return clip_score(
        risk,
        WEIGHT_DELIVERY
    )


# =========================================================
# 2. LEAD-TIME OVERRUN RISK
# =========================================================

def score_lead_time_overrun(row):

    """
    Measures how much actual full fulfillment exceeds
    the supplier's quoted lead time.

    A 50%+ overrun receives the maximum score for this
    dimension.
    """

    quoted = row[
        "avg_quoted_lead_time_days"
    ]

    actual = row[
        "avg_full_lead_time_days"
    ]

    if (
        pd.isna(quoted)
        or pd.isna(actual)
        or quoted <= 0
    ):
        return 0.0

    overrun_pct = max(
        (actual - quoted) / quoted,
        0
    )

    normalized = min(
        overrun_pct / 0.50,
        1
    )

    return (
        normalized
        * WEIGHT_LEAD_TIME
    )


# =========================================================
# 3. PARTIAL-SHIPMENT RISK
# =========================================================

def score_fragmentation(row):

    """
    Directly measures historical split-shipment exposure.

    100% partial shipment rate -> 20 points.
    """

    partial_rate = row[
        "partial_shipment_rate"
    ]

    if pd.isna(partial_rate):
        return 0.0

    return clip_score(
        partial_rate
        * WEIGHT_FRAGMENTATION,

        WEIGHT_FRAGMENTATION
    )


# =========================================================
# 4. LONG-TAIL RISK
# =========================================================

def score_tail_risk(row):

    """
    Compares P90 full lead time with average full lead time.

    A tail that is 50%+ above average receives the full
    tail-risk score.
    """

    average = row[
        "avg_full_lead_time_days"
    ]

    p90 = row[
        "p90_full_lead_time_days"
    ]

    if (
        pd.isna(average)
        or pd.isna(p90)
        or average <= 0
    ):
        return 0.0

    tail_pct = max(
        (p90 - average) / average,
        0
    )

    normalized = min(
        tail_pct / 0.50,
        1
    )

    return (
        normalized
        * WEIGHT_TAIL
    )


# =========================================================
# 5. FIRST-RECEIPT RISK
# =========================================================

def score_first_receipt(row):

    """
    Low first-receipt fill means the first delivery
    provides weak inventory relief.

    100% first fill -> 0 points
    0% first fill   -> 10 points
    """

    fill_rate = row[
        "avg_first_receipt_fill_rate"
    ]

    if pd.isna(fill_rate):
        return 0.0

    risk = (
        1 - fill_rate
    ) * WEIGHT_FIRST_RECEIPT

    return clip_score(
        risk,
        WEIGHT_FIRST_RECEIPT
    )


# =========================================================
# 6. LEAD-TIME VARIABILITY RISK
# =========================================================

def score_variability(row):

    """
    Uses coefficient of variation.

    CV >= 0.30 receives the maximum variability score.
    """

    cv = row[
        "full_lead_time_cv"
    ]

    if pd.isna(cv):
        return 0.0

    normalized = min(
        max(cv, 0) / 0.30,
        1
    )

    return (
        normalized
        * WEIGHT_VARIABILITY
    )


# =========================================================
# BUILD COMPONENT SCORES
# =========================================================

def add_risk_components(df):

    print(
        "Calculating supplier risk components..."
    )

    result = df.copy()

    result[
        "delivery_reliability_risk"
    ] = result.apply(
        score_delivery_reliability,
        axis=1
    )

    result[
        "lead_time_overrun_risk"
    ] = result.apply(
        score_lead_time_overrun,
        axis=1
    )

    result[
        "fragmentation_risk"
    ] = result.apply(
        score_fragmentation,
        axis=1
    )

    result[
        "tail_risk"
    ] = result.apply(
        score_tail_risk,
        axis=1
    )

    result[
        "first_receipt_risk"
    ] = result.apply(
        score_first_receipt,
        axis=1
    )

    result[
        "lead_time_variability_risk"
    ] = result.apply(
        score_variability,
        axis=1
    )

    return result


# =========================================================
# TOTAL RISK SCORE
# =========================================================

def add_total_risk_score(df):

    print(
        "Calculating total supplier risk score..."
    )

    result = df.copy()

    component_columns = [
        "delivery_reliability_risk",
        "lead_time_overrun_risk",
        "fragmentation_risk",
        "tail_risk",
        "first_receipt_risk",
        "lead_time_variability_risk",
    ]

    result[
        "supplier_risk_score"
    ] = (
        result[
            component_columns
        ]
        .sum(axis=1)
        .clip(
            lower=0,
            upper=100
        )
        .round(2)
    )

    return result


# =========================================================
# RISK CLASSIFICATION
# =========================================================

def classify_supplier_risk(score):

    if score >= 70:
        return "critical"

    if score >= 50:
        return "high"

    if score >= 30:
        return "moderate"

    return "low"


def add_risk_classification(df):

    print(
        "Assigning supplier risk classifications..."
    )

    result = df.copy()

    result[
        "supplier_risk_class"
    ] = result[
        "supplier_risk_score"
    ].apply(
        classify_supplier_risk
    )

    return result


# =========================================================
# DOMINANT RISK DRIVER
# =========================================================

def add_dominant_risk_driver(df):

    print(
        "Identifying dominant supplier risk drivers..."
    )

    result = df.copy()

    component_map = {
        "delivery_reliability_risk":
            "delivery_reliability",

        "lead_time_overrun_risk":
            "lead_time_overrun",

        "fragmentation_risk":
            "partial_shipments",

        "tail_risk":
            "long_tail_lead_time",

        "first_receipt_risk":
            "weak_first_receipt",

        "lead_time_variability_risk":
            "lead_time_variability",
    }

    component_columns = list(
        component_map.keys()
    )

    dominant_column = (
        result[
            component_columns
        ]
        .idxmax(axis=1)
    )

    result[
        "dominant_risk_driver"
    ] = dominant_column.map(
        component_map
    )

    return result


# =========================================================
# RISK REASONS
# =========================================================

def build_risk_reasons(row):

    reasons = []

    if (
        row[
            "full_po_on_time_rate"
        ] < 0.50
    ):
        reasons.append(
            "low full-PO on-time performance"
        )

    if (
        row[
            "planning_lead_time_buffer_days"
        ] > 14
    ):
        reasons.append(
            "material lead-time adjustment required"
        )

    if (
        row[
            "partial_shipment_rate"
        ] >= 0.40
    ):
        reasons.append(
            "frequent partial shipments"
        )

    if (
        row[
            "avg_first_receipt_fill_rate"
        ] < 0.70
    ):
        reasons.append(
            "first receipt provides limited order coverage"
        )

    if (
        row[
            "full_lead_time_cv"
        ] >= 0.20
    ):
        reasons.append(
            "high lead-time variability"
        )

    average = row[
        "avg_full_lead_time_days"
    ]

    p90 = row[
        "p90_full_lead_time_days"
    ]

    if (
        pd.notna(average)
        and pd.notna(p90)
        and average > 0
        and (
            (p90 - average)
            / average
        ) >= 0.25
    ):
        reasons.append(
            "meaningful long-tail delay exposure"
        )

    if not reasons:

        return (
            "historical supplier performance "
            "shows limited fulfillment risk"
        )

    return "; ".join(
        reasons
    )


def add_risk_reasons(df):

    print(
        "Building explainable supplier risk reasons..."
    )

    result = df.copy()

    result[
        "supplier_risk_reasons"
    ] = result.apply(
        build_risk_reasons,
        axis=1
    )

    return result


# =========================================================
# PLANNING RESPONSE
# =========================================================

def build_planning_response(row):

    risk_class = row[
        "supplier_risk_class"
    ]

    if risk_class == "critical":

        return (
            "strong planning protection required"
        )

    if risk_class == "high":

        return (
            "additional planning protection required"
        )

    if risk_class == "moderate":

        return (
            "monitor and apply measured lead-time protection"
        )

    return (
        "standard supplier planning treatment"
    )


def add_planning_response(df):

    print(
        "Assigning supplier planning responses..."
    )

    result = df.copy()

    result[
        "supplier_planning_response"
    ] = result.apply(
        build_planning_response,
        axis=1
    )

    return result


# =========================================================
# BUILD FINAL TABLE
# =========================================================

def build_supplier_risk(df):

    result = add_risk_components(
        df
    )

    result = add_total_risk_score(
        result
    )

    result = add_risk_classification(
        result
    )

    result = add_dominant_risk_driver(
        result
    )

    result = add_risk_reasons(
        result
    )

    result = add_planning_response(
        result
    )

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate_output(result, source):

    print("\n" + "=" * 90)
    print("SUPPLIER RISK VALIDATION")
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
        (
            result[
                "supplier_risk_score"
            ] >= 0
        )
        &
        (
            result[
                "supplier_risk_score"
            ] <= 100
        )
    ).all()

    print(
        "✓ Supplier risk scores between 0 and 100"
    )

    valid_classes = {
        "low",
        "moderate",
        "high",
        "critical",
    }

    assert set(
        result[
            "supplier_risk_class"
        ].unique()
    ).issubset(
        valid_classes
    )

    print(
        "✓ Supplier risk classes valid"
    )

    risk_columns = [
        "delivery_reliability_risk",
        "lead_time_overrun_risk",
        "fragmentation_risk",
        "tail_risk",
        "first_receipt_risk",
        "lead_time_variability_risk",
    ]

    assert (
        result[
            risk_columns
        ] >= 0
    ).all().all()

    print(
        "✓ All component risk scores non-negative"
    )

    # -----------------------------------------------------
    # Important synthetic validation
    #
    # behavior_type is ONLY used here to check whether the
    # model discovers the synthetic behavior we generated.
    # It does not create the score.
    # -----------------------------------------------------

    excellent = result[
        result[
            "behavior_type"
        ] == "excellent"
    ]

    partial_late = result[
        result[
            "behavior_type"
        ] == "partial_and_late"
    ]

    frequently_late = result[
        result[
            "behavior_type"
        ] == "frequently_late"
    ]

    assert (
        partial_late[
            "supplier_risk_score"
        ].mean()
        >
        excellent[
            "supplier_risk_score"
        ].mean()
    )

    print(
        "✓ Partial-and-late supplier scores "
        "riskier than excellent suppliers"
    )

    assert (
        frequently_late[
            "supplier_risk_score"
        ].mean()
        >
        excellent[
            "supplier_risk_score"
        ].mean()
    )

    print(
        "✓ Frequently-late suppliers score "
        "riskier than excellent suppliers"
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
            "fragmentation_risk"
        ].mean()
        >
        excellent[
            "fragmentation_risk"
        ].mean()
    )

    print(
        "✓ Partial suppliers show greater "
        "fragmentation risk"
    )

    print(
        "\nAll Phase 5C validation checks passed."
    )


# =========================================================
# PRINT RISK SCORECARD
# =========================================================

def print_risk_scorecard(result):

    print("\n" + "=" * 150)
    print("SUPPLIER RISK SCORECARD")
    print("=" * 150)

    columns = [
        "supplier_id",
        "behavior_type",
        "supplier_risk_score",
        "supplier_risk_class",
        "dominant_risk_driver",
        "full_po_on_time_rate",
        "partial_shipment_rate",
        "avg_full_lead_time_days",
        "p90_full_lead_time_days",
        "effective_planning_lead_time_days",
    ]

    print(
        result[
            columns
        ]
        .sort_values(
            "supplier_risk_score",
            ascending=False
        )
        .round(3)
        .to_string(
            index=False
        )
    )


# =========================================================
# PRINT COMPONENT BREAKDOWN
# =========================================================

def print_component_breakdown(result):

    print("\n" + "=" * 145)
    print("SUPPLIER RISK COMPONENTS")
    print("=" * 145)

    columns = [
        "supplier_id",
        "delivery_reliability_risk",
        "lead_time_overrun_risk",
        "fragmentation_risk",
        "tail_risk",
        "first_receipt_risk",
        "lead_time_variability_risk",
        "supplier_risk_score",
    ]

    print(
        result[
            columns
        ]
        .sort_values(
            "supplier_risk_score",
            ascending=False
        )
        .round(2)
        .to_string(
            index=False
        )
    )


# =========================================================
# PRINT EXPLANATIONS
# =========================================================

def print_risk_explanations(result):

    print("\n" + "=" * 120)
    print("EXPLAINABLE SUPPLIER RISK")
    print("=" * 120)

    ordered = result.sort_values(
        "supplier_risk_score",
        ascending=False
    )

    for _, row in ordered.iterrows():

        print(
            f"\n{row['supplier_id']} | "
            f"Score: {row['supplier_risk_score']:.2f} | "
            f"Class: {row['supplier_risk_class']}"
        )

        print(
            f"Driver: "
            f"{row['dominant_risk_driver']}"
        )

        print(
            f"Why: "
            f"{row['supplier_risk_reasons']}"
        )

        print(
            f"Planning response: "
            f"{row['supplier_planning_response']}"
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
        "SUPPLIER RISK CLASSIFICATION"
    )

    print("=" * 90)

    source = load_data()

    validate_input_schema(
        source
    )

    result = build_supplier_risk(
        source
    )

    validate_output(
        result,
        source
    )

    save_output(
        result
    )

    print_risk_scorecard(
        result
    )

    print_component_breakdown(
        result
    )

    print_risk_explanations(
        result
    )

    print("\n" + "=" * 90)

    print(
        "PHASE 5C COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()