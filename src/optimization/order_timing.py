from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RECOMMENDATIONS_PATH = (
    PROCESSED_DIR / "procurement_recommendations.csv"
)

PLANNING_MASTER_PATH = (
    PROCESSED_DIR / "planning_master.csv"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "order_timing_recommendations.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

DAYS_PER_WEEK = 7.0

ORDER_NOW_DAYS = 0
ORDER_THIS_WEEK_DAYS = 7
ORDER_SOON_DAYS = 28

HIGH_FORECAST_WAPE = 50.0


# =========================================================
# DISPLAY HELPERS
# =========================================================

def section(title, width=105):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def check(message):
    print(f"✓ {message}")


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    section(
        "PROCUREMENT AI - ORDER TIMING & PRIORITIZATION ENGINE"
    )

    print("\nLoading Phase 7C inputs...")

    recommendations = pd.read_csv(
        RECOMMENDATIONS_PATH
    )

    planning = pd.read_csv(
        PLANNING_MASTER_PATH
    )

    check(
        f"procurement_recommendations "
        f"{len(recommendations):>8,} rows"
    )

    check(
        f"planning_master             "
        f"{len(planning):>8,} rows"
    )

    return recommendations, planning


# =========================================================
# VALIDATE INPUTS
# =========================================================

def validate_inputs(
    recommendations,
    planning
):

    print("Validating order-timing inputs...")

    required_recommendations = {
        "sku",
        "product_name",
        "category",
        "supplier_id",
        "unit_cost",
        "moq",
        "case_pack",
        "final_demand_class",
        "forecast_readiness",
        "production_model",
        "model_source",
        "selected_wape",
        "service_level_target",
        "planning_weekly_demand",
        "planning_lead_time_weeks",
        "safety_stock",
        "reorder_point",
        "net_available",
        "weeks_until_stockout",
        "projected_stockout_date",
        "stockout_before_replenishment",
        "total_open_po_qty",
        "timely_open_po_qty",
        "late_to_need_open_po_qty",
        "at_risk_open_po_qty",
        "timely_inventory_position",
        "low_risk_inventory_position",
        "base_order_requirement",
        "recommended_order_qty",
        "recommended_order_value",
        "supplier_risk_score",
        "supplier_risk_class",
        "inventory_policy_status",
        "procurement_priority",
        "procurement_action",
        "procurement_reason",
    }

    required_planning = {
        "sku",
        "snapshot_date",
    }

    missing_rec = (
        required_recommendations
        - set(recommendations.columns)
    )

    missing_planning = (
        required_planning
        - set(planning.columns)
    )

    if missing_rec:
        raise ValueError(
            "procurement_recommendations.csv "
            "missing columns: "
            + ", ".join(sorted(missing_rec))
        )

    if missing_planning:
        raise ValueError(
            "planning_master.csv missing columns: "
            + ", ".join(sorted(missing_planning))
        )

    if recommendations["sku"].duplicated().any():
        raise ValueError(
            "Procurement recommendations contain "
            "duplicate SKUs"
        )

    if planning["sku"].duplicated().any():
        raise ValueError(
            "Planning master contains duplicate SKUs"
        )

    if len(recommendations) != 200:
        raise ValueError(
            f"Expected 200 SKUs, "
            f"found {len(recommendations)}"
        )

    check("Required Phase 7C columns present")
    check("One recommendation row per SKU")
    check("Planning master has one row per SKU")


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(
    recommendations,
    planning
):

    print("Preparing order-timing dataset...")

    snapshot = planning[
        [
            "sku",
            "snapshot_date",
        ]
    ].copy()

    snapshot[
        "snapshot_date"
    ] = pd.to_datetime(
        snapshot["snapshot_date"],
        errors="coerce"
    )

    df = recommendations.merge(
        snapshot,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    df[
        "projected_stockout_date"
    ] = pd.to_datetime(
        df["projected_stockout_date"],
        errors="coerce"
    )

    numeric_columns = [
        "unit_cost",
        "planning_weekly_demand",
        "planning_lead_time_weeks",
        "safety_stock",
        "reorder_point",
        "net_available",
        "weeks_until_stockout",
        "total_open_po_qty",
        "timely_open_po_qty",
        "late_to_need_open_po_qty",
        "at_risk_open_po_qty",
        "timely_inventory_position",
        "low_risk_inventory_position",
        "base_order_requirement",
        "recommended_order_qty",
        "recommended_order_value",
        "supplier_risk_score",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0.0)

    df["selected_wape"] = pd.to_numeric(
        df["selected_wape"],
        errors="coerce"
    )

    if df["snapshot_date"].isna().any():
        raise ValueError(
            "Missing snapshot date for one or more SKUs"
        )

    return df


# =========================================================
# REQUIRED RECEIPT DATE
# =========================================================

def calculate_required_receipt_date(df):

    print("Calculating required receipt dates...")

    result = df.copy()

    # For purchasing timing, projected stockout is
    # treated as the latest date replenishment can
    # arrive without exhausting current net supply.
    result[
        "required_receipt_date"
    ] = result[
        "projected_stockout_date"
    ]

    # SKUs with no projected stockout do not require
    # a receipt date.
    no_order = (
        result[
            "recommended_order_qty"
        ] <= 0
    )

    result.loc[
        no_order,
        "required_receipt_date"
    ] = pd.NaT

    return result


# =========================================================
# LATEST PO RELEASE DATE
# =========================================================

def calculate_latest_release_date(df):

    print("Calculating latest PO release dates...")

    result = df.copy()

    lead_days = (
        result[
            "planning_lead_time_weeks"
        ]
        * DAYS_PER_WEEK
    )

    result[
        "planning_lead_time_days"
    ] = lead_days

    result[
        "latest_po_release_date"
    ] = (
        result[
            "required_receipt_date"
        ]
        - pd.to_timedelta(
            lead_days,
            unit="D"
        )
    )

    result[
        "days_until_po_release"
    ] = (
        result[
            "latest_po_release_date"
        ]
        - result[
            "snapshot_date"
        ]
    ).dt.total_seconds() / 86400.0

    result[
        "weeks_until_po_release"
    ] = (
        result[
            "days_until_po_release"
        ]
        / DAYS_PER_WEEK
    )

    return result


# =========================================================
# ORDER TIMING STATUS
# =========================================================

def assign_order_timing(df):

    print("Assigning order timing...")

    result = df.copy()

    timing = []

    for _, row in result.iterrows():

        qty = row[
            "recommended_order_qty"
        ]

        days = row[
            "days_until_po_release"
        ]

        late_open_qty = row[
            "late_to_need_open_po_qty"
        ]

        risk_qty = row[
            "at_risk_open_po_qty"
        ]

        if qty <= 0:

            if late_open_qty > 0:
                timing.append(
                    "EXPEDITE EXISTING PO"
                )

            elif risk_qty > 0:
                timing.append(
                    "MONITOR OPEN PO"
                )

            else:
                timing.append(
                    "NO ACTION"
                )

            continue

        if pd.isna(days):
            timing.append(
                "ORDER NOW"
            )

        elif days <= ORDER_NOW_DAYS:
            timing.append(
                "ORDER NOW"
            )

        elif days <= ORDER_THIS_WEEK_DAYS:
            timing.append(
                "ORDER THIS WEEK"
            )

        elif days <= ORDER_SOON_DAYS:
            timing.append(
                "ORDER SOON"
            )

        else:
            timing.append(
                "PLANNED ORDER"
            )

    result[
        "order_timing_action"
    ] = timing

    return result


# =========================================================
# URGENCY SCORE
# =========================================================

def calculate_urgency_score(df):

    print("Calculating procurement urgency scores...")

    result = df.copy()

    scores = []

    for _, row in result.iterrows():

        if row["recommended_order_qty"] <= 0:
            scores.append(0.0)
            continue

        score = 0.0

        # ---------------------------------------------
        # 1. Time-to-stockout risk: maximum 40 points
        # ---------------------------------------------

        weeks = row[
            "weeks_until_stockout"
        ]

        if weeks <= 0:
            score += 40

        elif weeks <= 1:
            score += 38

        elif weeks <= 2:
            score += 35

        elif weeks <= 4:
            score += 30

        elif weeks <= 8:
            score += 20

        elif weeks <= 12:
            score += 10

        else:
            score += 5

        # ---------------------------------------------
        # 2. Release-date urgency: maximum 25 points
        # ---------------------------------------------

        days = row[
            "days_until_po_release"
        ]

        if pd.isna(days):
            score += 25

        elif days <= 0:
            score += 25

        elif days <= 7:
            score += 20

        elif days <= 28:
            score += 12

        else:
            score += 5

        # ---------------------------------------------
        # 3. Supplier risk: maximum 15 points
        # ---------------------------------------------

        supplier_risk = row[
            "supplier_risk_class"
        ]

        if supplier_risk == "critical":
            score += 15

        elif supplier_risk == "moderate":
            score += 8

        else:
            score += 3

        # ---------------------------------------------
        # 4. Open PO risk: maximum 10 points
        # ---------------------------------------------

        if row[
            "late_to_need_open_po_qty"
        ] > 0:
            score += 6

        if row[
            "at_risk_open_po_qty"
        ] > 0:
            score += 4

        # ---------------------------------------------
        # 5. Demand / service protection: max 10
        # ---------------------------------------------

        demand_class = row[
            "final_demand_class"
        ]

        if demand_class in {
            "growth",
            "new_product",
            "volatile",
        }:
            score += 5

        elif demand_class == "intermittent":
            score += 3

        else:
            score += 2

        if row[
            "service_level_target"
        ] >= 0.99:
            score += 5

        elif row[
            "service_level_target"
        ] >= 0.97:
            score += 3

        else:
            score += 1

        scores.append(
            min(score, 100.0)
        )

    result[
        "procurement_urgency_score"
    ] = scores

    return result


# =========================================================
# URGENCY CLASS
# =========================================================

def classify_urgency(df):

    print("Classifying procurement urgency...")

    result = df.copy()

    def classify(row):

        action = row[
            "order_timing_action"
        ]

        score = row[
            "procurement_urgency_score"
        ]

        if action == "ORDER NOW":

            if score >= 80:
                return "critical"

            return "high"

        if action == "ORDER THIS WEEK":
            return "high"

        if action == "ORDER SOON":

            if score >= 70:
                return "high"

            return "medium"

        if action == "PLANNED ORDER":
            return "planned"

        if action == "EXPEDITE EXISTING PO":
            return "expedite"

        if action == "MONITOR OPEN PO":
            return "monitor"

        return "no_action"

    result[
        "timing_priority"
    ] = result.apply(
        classify,
        axis=1
    )

    rank = {
        "critical": 1,
        "high": 2,
        "expedite": 3,
        "medium": 4,
        "planned": 5,
        "monitor": 6,
        "no_action": 7,
    }

    result[
        "timing_priority_rank"
    ] = result[
        "timing_priority"
    ].map(rank)

    return result


# =========================================================
# FORECAST CONFIDENCE / REVIEW
# =========================================================

def add_planner_review_flags(df):

    print("Adding planner review controls...")

    result = df.copy()

    result[
        "forecast_review_flag"
    ] = (
        result[
            "selected_wape"
        ].fillna(0)
        >= HIGH_FORECAST_WAPE
    )

    result[
        "supplier_review_flag"
    ] = (
        result[
            "supplier_risk_class"
        ] == "critical"
    )

    result[
        "open_po_review_flag"
    ] = (
        (
            result[
                "late_to_need_open_po_qty"
            ] > 0
        )
        | (
            result[
                "at_risk_open_po_qty"
            ] > 0
        )
    )

    result[
        "manual_review_required"
    ] = (
        result[
            "forecast_review_flag"
        ]
        | result[
            "supplier_review_flag"
        ]
        | result[
            "open_po_review_flag"
        ]
    )

    return result


# =========================================================
# CASH COMMITMENT BUCKETS
# =========================================================

def calculate_commitment_buckets(df):

    print("Calculating procurement commitment buckets...")

    result = df.copy()

    action = result[
        "order_timing_action"
    ]

    value = result[
        "recommended_order_value"
    ]

    result[
        "commit_now_value"
    ] = np.where(
        action == "ORDER NOW",
        value,
        0.0
    )

    result[
        "commit_this_week_value"
    ] = np.where(
        action == "ORDER THIS WEEK",
        value,
        0.0
    )

    result[
        "commit_soon_value"
    ] = np.where(
        action == "ORDER SOON",
        value,
        0.0
    )

    result[
        "planned_future_value"
    ] = np.where(
        action == "PLANNED ORDER",
        value,
        0.0
    )

    return result


# =========================================================
# EXPLAINABILITY
# =========================================================

def build_timing_reason(row):

    reasons = []

    action = row[
        "order_timing_action"
    ]

    qty = int(
        row[
            "recommended_order_qty"
        ]
    )

    if qty > 0:

        reasons.append(
            f"recommended order is {qty:,} units"
        )

        if pd.notna(
            row[
                "latest_po_release_date"
            ]
        ):

            release_date = (
                row[
                    "latest_po_release_date"
                ]
                .strftime("%Y-%m-%d")
            )

            reasons.append(
                f"latest calculated PO release "
                f"date is {release_date}"
            )

    if (
        row[
            "weeks_until_stockout"
        ] <= 2
        and qty > 0
    ):

        reasons.append(
            "less than two weeks of current "
            "net supply remain"
        )

    elif (
        row[
            "weeks_until_stockout"
        ] <= 4
        and qty > 0
    ):

        reasons.append(
            "less than four weeks of current "
            "net supply remain"
        )

    if (
        row[
            "late_to_need_open_po_qty"
        ] > 0
    ):

        reasons.append(
            f"{int(row['late_to_need_open_po_qty']):,} "
            "open-PO units are expected too late"
        )

    if (
        row[
            "at_risk_open_po_qty"
        ] > 0
    ):

        reasons.append(
            f"{int(row['at_risk_open_po_qty']):,} "
            "open-PO units carry delivery risk"
        )

    if (
        row[
            "supplier_risk_class"
        ] == "critical"
    ):

        reasons.append(
            "supplier has critical delivery risk"
        )

    if bool(
        row[
            "forecast_review_flag"
        ]
    ):

        reasons.append(
            "forecast error is high and should "
            "be reviewed before release"
        )

    if action == "NO ACTION":

        reasons.append(
            "no new purchase is currently required"
        )

    elif action == "MONITOR OPEN PO":

        reasons.append(
            "existing supply currently covers need "
            "but requires monitoring"
        )

    elif action == "EXPEDITE EXISTING PO":

        reasons.append(
            "existing PO should be reviewed or expedited "
            "before creating additional supply"
        )

    return "; ".join(reasons)


def add_explanations(df):

    print("Building explainable timing recommendations...")

    result = df.copy()

    result[
        "timing_reason"
    ] = result.apply(
        build_timing_reason,
        axis=1
    )

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate_output(df):

    section(
        "ORDER TIMING VALIDATION"
    )

    assert len(df) == 200
    check("Exactly 200 SKU timing rows")

    assert df["sku"].nunique() == 200
    check("One timing recommendation per SKU")

    assert (
        df[
            "recommended_order_qty"
        ] >= 0
    ).all()

    check(
        "Recommended order quantities are non-negative"
    )

    valid_actions = {
        "ORDER NOW",
        "ORDER THIS WEEK",
        "ORDER SOON",
        "PLANNED ORDER",
        "EXPEDITE EXISTING PO",
        "MONITOR OPEN PO",
        "NO ACTION",
    }

    assert set(
        df[
            "order_timing_action"
        ].unique()
    ).issubset(
        valid_actions
    )

    check("Order timing actions valid")

    positive_orders = (
        df[
            "recommended_order_qty"
        ] > 0
    )

    assert df.loc[
        positive_orders,
        "latest_po_release_date"
    ].notna().all()

    check(
        "All positive orders have latest release dates"
    )

    no_order = (
        df[
            "recommended_order_qty"
        ] <= 0
    )

    assert (
        df.loc[
            no_order,
            "order_timing_action"
        ].isin(
            {
                "EXPEDITE EXISTING PO",
                "MONITOR OPEN PO",
                "NO ACTION",
            }
        )
    ).all()

    check(
        "Zero-order SKUs do not receive new-order actions"
    )

    assert (
        df[
            "procurement_urgency_score"
        ].between(
            0,
            100
        )
    ).all()

    check(
        "Urgency scores are between 0 and 100"
    )

    total_value = (
        df[
            "recommended_order_value"
        ].sum()
    )

    bucket_value = (
        df[
            "commit_now_value"
        ].sum()
        + df[
            "commit_this_week_value"
        ].sum()
        + df[
            "commit_soon_value"
        ].sum()
        + df[
            "planned_future_value"
        ].sum()
    )

    assert np.isclose(
        total_value,
        bucket_value
    )

    check(
        "Purchase value reconciles across timing buckets"
    )

    assert df[
        "timing_reason"
    ].notna().all()

    check(
        "All SKUs have explainable timing reasons"
    )

    print(
        "\nAll Phase 7C validation checks passed."
    )


# =========================================================
# SELECT OUTPUT
# =========================================================

def select_output_columns(df):

    columns = [
        "sku",
        "product_name",
        "category",
        "supplier_id",

        "unit_cost",
        "moq",
        "case_pack",

        "final_demand_class",
        "forecast_readiness",
        "production_model",
        "model_source",
        "selected_wape",

        "service_level_target",
        "planning_weekly_demand",
        "planning_lead_time_weeks",
        "planning_lead_time_days",

        "safety_stock",
        "reorder_point",
        "net_available",

        "weeks_until_stockout",
        "projected_stockout_date",
        "required_receipt_date",

        "latest_po_release_date",
        "days_until_po_release",
        "weeks_until_po_release",

        "total_open_po_qty",
        "timely_open_po_qty",
        "late_to_need_open_po_qty",
        "at_risk_open_po_qty",

        "timely_inventory_position",
        "low_risk_inventory_position",

        "base_order_requirement",
        "recommended_order_qty",
        "recommended_order_value",

        "supplier_risk_score",
        "supplier_risk_class",

        "inventory_policy_status",
        "procurement_priority",

        "order_timing_action",
        "procurement_urgency_score",
        "timing_priority",
        "timing_priority_rank",

        "forecast_review_flag",
        "supplier_review_flag",
        "open_po_review_flag",
        "manual_review_required",

        "commit_now_value",
        "commit_this_week_value",
        "commit_soon_value",
        "planned_future_value",

        "timing_reason",
    ]

    return df[columns].copy()


# =========================================================
# SAVE
# =========================================================

def save_output(df):

    section(
        "SAVING ORDER TIMING RECOMMENDATIONS"
    )

    output = df.copy()

    date_columns = [
        "projected_stockout_date",
        "required_receipt_date",
        "latest_po_release_date",
    ]

    for column in date_columns:

        output[column] = pd.to_datetime(
            output[column],
            errors="coerce"
        ).dt.strftime(
            "%Y-%m-%d"
        )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    check(
        OUTPUT_PATH.name
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary(df):

    section(
        "ORDER TIMING SUMMARY"
    )

    print(
        f"\nSKUs evaluated: "
        f"{len(df):,}"
    )

    print(
        f"Total forward recommended value: "
        f"${df['recommended_order_value'].sum():,.2f}"
    )

    print(
        f"\nCommit NOW: "
        f"${df['commit_now_value'].sum():,.2f}"
    )

    print(
        f"Commit THIS WEEK: "
        f"${df['commit_this_week_value'].sum():,.2f}"
    )

    print(
        f"Commit SOON: "
        f"${df['commit_soon_value'].sum():,.2f}"
    )

    print(
        f"Planned future: "
        f"${df['planned_future_value'].sum():,.2f}"
    )

    print(
        "\nOrder timing actions:"
    )

    print(
        df[
            "order_timing_action"
        ].value_counts()
    )

    print(
        "\nTiming priority:"
    )

    print(
        df[
            "timing_priority"
        ].value_counts()
    )

    print(
        f"\nManual review required: "
        f"{df['manual_review_required'].sum():,} SKUs"
    )

    section(
        "ACTION VALUE SUMMARY",
        width=125
    )

    action_summary = (
        df.groupby(
            "order_timing_action"
        )
        .agg(
            sku_count=(
                "sku",
                "count"
            ),
            recommended_units=(
                "recommended_order_qty",
                "sum"
            ),
            recommended_value=(
                "recommended_order_value",
                "sum"
            ),
            avg_urgency_score=(
                "procurement_urgency_score",
                "mean"
            ),
        )
        .round(2)
        .sort_values(
            "recommended_value",
            ascending=False
        )
    )

    print(
        action_summary.to_string()
    )

    section(
        "TOP IMMEDIATE PROCUREMENT ACTIONS",
        width=190
    )

    display_columns = [
        "sku",
        "product_name",
        "supplier_id",
        "supplier_risk_class",
        "final_demand_class",
        "planning_weekly_demand",
        "weeks_until_stockout",
        "latest_po_release_date",
        "late_to_need_open_po_qty",
        "at_risk_open_po_qty",
        "recommended_order_qty",
        "recommended_order_value",
        "order_timing_action",
        "procurement_urgency_score",
        "timing_priority",
        "timing_priority_rank",
    ]

    top = (
        df[
            display_columns
        ]
        .sort_values(
            [
                "timing_priority_rank",
                "procurement_urgency_score",
                "recommended_order_value",
            ],
            ascending=[
                True,
                False,
                False,
            ]
        )
        .head(25)
    )

    print(
        top.to_string(
            index=False
        )
    )


# =========================================================
# MAIN
# =========================================================

def main():

    recommendations, planning = (
        load_data()
    )

    validate_inputs(
        recommendations,
        planning
    )

    df = prepare_data(
        recommendations,
        planning
    )

    df = calculate_required_receipt_date(
        df
    )

    df = calculate_latest_release_date(
        df
    )

    df = assign_order_timing(
        df
    )

    df = calculate_urgency_score(
        df
    )

    df = classify_urgency(
        df
    )

    df = add_planner_review_flags(
        df
    )

    df = calculate_commitment_buckets(
        df
    )

    df = add_explanations(
        df
    )

    validate_output(
        df
    )

    output = select_output_columns(
        df
    )

    save_output(
        output
    )

    print_summary(
        output
    )

    section(
        "PHASE 7C COMPLETE"
    )


if __name__ == "__main__":
    main()