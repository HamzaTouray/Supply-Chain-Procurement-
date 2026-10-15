from pathlib import Path
import math

import numpy as np
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INVENTORY_POLICY_PATH = (
    PROCESSED_DIR / "inventory_policy.csv"
)

OPEN_PO_PATH = (
    PROCESSED_DIR / "open_purchase_orders_clean.csv"
)

PLANNING_MASTER_PATH = (
    PROCESSED_DIR / "planning_master.csv"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "procurement_recommendations.csv"
)

OPEN_PO_ANALYSIS_PATH = (
    PROCESSED_DIR / "open_po_timing_analysis.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

DAYS_PER_WEEK = 7.0

VERY_URGENT_WEEKS = 2.0
URGENT_WEEKS = 4.0

MIN_ORDER_QTY = 0


# =========================================================
# DISPLAY HELPERS
# =========================================================

def section(title, width=100):

    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def check(message):

    print(f"✓ {message}")


# =========================================================
# BOOLEAN NORMALIZATION
# =========================================================

def normalize_boolean(series):

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    section(
        "PROCUREMENT AI - PROCUREMENT RECOMMENDATION ENGINE"
    )

    print("\nLoading Phase 7B inputs...")

    policy = pd.read_csv(
        INVENTORY_POLICY_PATH
    )

    open_pos = pd.read_csv(
        OPEN_PO_PATH
    )

    planning = pd.read_csv(
        PLANNING_MASTER_PATH
    )

    check(
        f"inventory_policy          "
        f"{len(policy):>8,} rows"
    )

    check(
        f"open_purchase_orders      "
        f"{len(open_pos):>8,} rows"
    )

    check(
        f"planning_master           "
        f"{len(planning):>8,} rows"
    )

    return policy, open_pos, planning


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_inputs(
    policy,
    open_pos,
    planning
):

    print("Validating procurement inputs...")

    required_policy = {
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
        "forecast_lead_time_demand",
        "supplier_risk_score",
        "supplier_risk_class",
        "safety_stock",
        "reorder_point",
        "on_hand",
        "allocated",
        "backordered",
        "net_available",
        "total_open_po_qty",
        "total_overdue_po_qty",
        "policy_inventory_position",
        "gross_replenishment_need",
        "inventory_policy_status",
        "inventory_policy_reason",
    }

    required_open_po = {
        "po_number",
        "po_line",
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
        "open_po_scenario",
        "partial_receipt_flag",
        "overdue_flag",
        "expected_after_promise_flag",
    }

    required_planning = {
        "sku",
        "snapshot_date",
    }

    missing_policy = (
        required_policy
        - set(policy.columns)
    )

    missing_open_po = (
        required_open_po
        - set(open_pos.columns)
    )

    missing_planning = (
        required_planning
        - set(planning.columns)
    )

    if missing_policy:

        raise ValueError(
            "inventory_policy.csv missing columns: "
            + ", ".join(sorted(missing_policy))
        )

    if missing_open_po:

        raise ValueError(
            "open_purchase_orders_clean.csv "
            "missing columns: "
            + ", ".join(sorted(missing_open_po))
        )

    if missing_planning:

        raise ValueError(
            "planning_master.csv missing columns: "
            + ", ".join(sorted(missing_planning))
        )

    if policy["sku"].duplicated().any():

        raise ValueError(
            "inventory_policy contains duplicate SKUs"
        )

    if planning["sku"].duplicated().any():

        raise ValueError(
            "planning_master contains duplicate SKUs"
        )

    if len(policy) != 200:

        raise ValueError(
            f"Expected 200 policy SKUs, found {len(policy)}"
        )

    check("Required procurement columns present")
    check("Inventory policy has one row per SKU")
    check("Planning master has one row per SKU")


# =========================================================
# PREPARE OPEN PO DATA
# =========================================================

def prepare_open_pos(open_pos):

    print("Preparing open purchase orders...")

    df = open_pos.copy()

    date_columns = [
        "order_date",
        "promised_date",
        "expected_date",
    ]

    for column in date_columns:

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    numeric_columns = [
        "ordered_qty",
        "received_qty",
        "open_qty",
        "days_late",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0.0)

    boolean_columns = [
        "partial_receipt_flag",
        "overdue_flag",
        "expected_after_promise_flag",
    ]

    for column in boolean_columns:

        df[column] = normalize_boolean(
            df[column]
        )

    if df["expected_date"].isna().any():

        raise ValueError(
            "Open PO rows contain missing expected dates"
        )

    if (df["open_qty"] <= 0).any():

        raise ValueError(
            "Open PO rows must have positive open quantity"
        )

    check(
        f"Open PO units: "
        f"{df['open_qty'].sum():,.0f}"
    )

    check(
        f"SKUs with open supply: "
        f"{df['sku'].nunique():,}"
    )

    return df


# =========================================================
# SNAPSHOT DATE
# =========================================================

def add_snapshot_date(
    policy,
    planning
):

    print("Adding planning snapshot date...")

    snapshot = planning[
        ["sku", "snapshot_date"]
    ].copy()

    snapshot[
        "snapshot_date"
    ] = pd.to_datetime(
        snapshot["snapshot_date"],
        errors="coerce"
    )

    result = policy.merge(
        snapshot,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    if result[
        "snapshot_date"
    ].isna().any():

        raise ValueError(
            "Missing snapshot date for one or more SKUs"
        )

    return result


# =========================================================
# STOCKOUT / EXPOSURE TIMING
# =========================================================

def calculate_exposure_timing(df):

    print("Calculating inventory exposure timing...")

    result = df.copy()

    demand = pd.to_numeric(
        result["planning_weekly_demand"],
        errors="coerce"
    ).fillna(0.0)

    net_available = pd.to_numeric(
        result["net_available"],
        errors="coerce"
    ).fillna(0.0)

    result[
        "weeks_until_stockout"
    ] = np.where(
        demand > 0,
        np.maximum(
            net_available,
            0
        ) / demand,
        np.inf
    )

    result[
        "days_until_stockout"
    ] = np.where(
        np.isfinite(
            result["weeks_until_stockout"]
        ),
        result[
            "weeks_until_stockout"
        ] * DAYS_PER_WEEK,
        np.nan
    )

    result[
        "projected_stockout_date"
    ] = pd.NaT

    finite_mask = np.isfinite(
        result["weeks_until_stockout"]
    )

    result.loc[
        finite_mask,
        "projected_stockout_date"
    ] = (
        result.loc[
            finite_mask,
            "snapshot_date"
        ]
        + pd.to_timedelta(
            result.loc[
                finite_mask,
                "days_until_stockout"
            ],
            unit="D"
        )
    )

    result[
        "stockout_before_replenishment"
    ] = (
        result[
            "weeks_until_stockout"
        ]
        < result[
            "planning_lead_time_weeks"
        ]
    )

    return result


# =========================================================
# OPEN PO TIMING ANALYSIS
# =========================================================

def analyze_open_po_timing(
    policy,
    open_pos
):

    print("Analyzing open PO timing...")

    timing_base = policy[
        [
            "sku",
            "snapshot_date",
            "projected_stockout_date",
            "planning_weekly_demand",
            "weeks_until_stockout",
            "planning_lead_time_weeks",
        ]
    ].copy()

    po = open_pos.merge(
        timing_base,
        on="sku",
        how="left",
        validate="many_to_one"
    )

    if po[
        "snapshot_date"
    ].isna().any():

        raise ValueError(
            "Open PO contains SKU not found "
            "in inventory policy"
        )

    po[
        "days_until_expected_receipt"
    ] = (
        po["expected_date"]
        - po["snapshot_date"]
    ).dt.days

    po[
        "weeks_until_expected_receipt"
    ] = (
        po[
            "days_until_expected_receipt"
        ]
        / DAYS_PER_WEEK
    )

    # PO is considered timely if it is expected
    # before the projected stockout date.
    po[
        "arrives_before_stockout"
    ] = False

    finite_stockout = (
        po[
            "projected_stockout_date"
        ].notna()
    )

    po.loc[
        finite_stockout,
        "arrives_before_stockout"
    ] = (
        po.loc[
            finite_stockout,
            "expected_date"
        ]
        <= po.loc[
            finite_stockout,
            "projected_stockout_date"
        ]
    )

    # If demand is zero, there is no stockout exposure.
    zero_demand = (
        po[
            "planning_weekly_demand"
        ] <= 0
    )

    po.loc[
        zero_demand,
        "arrives_before_stockout"
    ] = True

    po[
        "po_risk_flag"
    ] = (
        po["overdue_flag"]
        | po["partial_receipt_flag"]
        | po["expected_after_promise_flag"]
    )

    po[
        "timely_open_qty"
    ] = np.where(
        po[
            "arrives_before_stockout"
        ],
        po["open_qty"],
        0.0
    )

    po[
        "late_to_need_open_qty"
    ] = np.where(
        ~po[
            "arrives_before_stockout"
        ],
        po["open_qty"],
        0.0
    )

    po[
        "at_risk_open_qty"
    ] = np.where(
        po[
            "po_risk_flag"
        ],
        po["open_qty"],
        0.0
    )

    po[
        "timely_low_risk_open_qty"
    ] = np.where(
        po[
            "arrives_before_stockout"
        ]
        & ~po[
            "po_risk_flag"
        ],
        po["open_qty"],
        0.0
    )

    return po


# =========================================================
# AGGREGATE OPEN PO TIMING
# =========================================================

def aggregate_open_po_timing(
    po_analysis
):

    print("Aggregating open PO timing by SKU...")

    summary = (
        po_analysis
        .groupby(
            "sku",
            as_index=False
        )
        .agg(
            open_po_line_count=(
                "po_number",
                "count"
            ),
            open_po_qty_check=(
                "open_qty",
                "sum"
            ),
            timely_open_po_qty=(
                "timely_open_qty",
                "sum"
            ),
            timely_low_risk_open_po_qty=(
                "timely_low_risk_open_qty",
                "sum"
            ),
            late_to_need_open_po_qty=(
                "late_to_need_open_qty",
                "sum"
            ),
            at_risk_open_po_qty=(
                "at_risk_open_qty",
                "sum"
            ),
            overdue_open_po_lines=(
                "overdue_flag",
                "sum"
            ),
            partial_open_po_lines=(
                "partial_receipt_flag",
                "sum"
            ),
            delayed_expected_lines=(
                "expected_after_promise_flag",
                "sum"
            ),
            next_open_po_expected_date=(
                "expected_date",
                "min"
            ),
            latest_open_po_expected_date=(
                "expected_date",
                "max"
            ),
        )
    )

    return summary


# =========================================================
# MERGE OPEN PO ANALYSIS
# =========================================================

def add_open_po_analysis(
    policy,
    po_summary
):

    print("Adding open PO timing to policy...")

    result = policy.merge(
        po_summary,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    quantity_columns = [
        "open_po_line_count",
        "open_po_qty_check",
        "timely_open_po_qty",
        "timely_low_risk_open_po_qty",
        "late_to_need_open_po_qty",
        "at_risk_open_po_qty",
        "overdue_open_po_lines",
        "partial_open_po_lines",
        "delayed_expected_lines",
    ]

    for column in quantity_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        ).fillna(0.0)

    return result


# =========================================================
# SUPPLY ADEQUACY
# =========================================================

def calculate_supply_adequacy(df):

    print("Calculating time-phased supply adequacy...")

    result = df.copy()

    # Inventory available before considering future POs.
    result[
        "current_net_supply"
    ] = (
        pd.to_numeric(
            result["on_hand"],
            errors="coerce"
        ).fillna(0.0)
        - pd.to_numeric(
            result["allocated"],
            errors="coerce"
        ).fillna(0.0)
        - pd.to_numeric(
            result["backordered"],
            errors="coerce"
        ).fillna(0.0)
    )

    # Supply that is expected before projected stockout.
    result[
        "timely_inventory_position"
    ] = (
        result[
            "current_net_supply"
        ]
        + result[
            "timely_open_po_qty"
        ]
    )

    # More conservative position:
    # only incoming POs that are both timely and not
    # flagged as overdue/partial/delayed.
    result[
        "low_risk_inventory_position"
    ] = (
        result[
            "current_net_supply"
        ]
        + result[
            "timely_low_risk_open_po_qty"
        ]
    )

    result[
        "timely_supply_gap"
    ] = (
        result[
            "reorder_point"
        ]
        - result[
            "timely_inventory_position"
        ]
    )

    result[
        "low_risk_supply_gap"
    ] = (
        result[
            "reorder_point"
        ]
        - result[
            "low_risk_inventory_position"
        ]
    )

    result[
        "timely_replenishment_need"
    ] = np.ceil(
        np.maximum(
            result[
                "timely_supply_gap"
            ],
            0.0
        )
    ).astype(int)

    result[
        "low_risk_replenishment_need"
    ] = np.ceil(
        np.maximum(
            result[
                "low_risk_supply_gap"
            ],
            0.0
        )
    ).astype(int)

    return result


# =========================================================
# BASE PROCUREMENT REQUIREMENT
# =========================================================

def calculate_procurement_requirement(df):

    print("Calculating procurement requirement...")

    result = df.copy()

    # The executable base requirement uses timely supply.
    # We do NOT automatically replace all risky PO units.
    result[
        "base_order_requirement"
    ] = result[
        "timely_replenishment_need"
    ].astype(int)

    result[
        "risk_exposure_qty"
    ] = np.maximum(
        result[
            "low_risk_replenishment_need"
        ]
        - result[
            "timely_replenishment_need"
        ],
        0
    ).astype(int)

    result[
        "requires_order"
    ] = (
        result[
            "base_order_requirement"
        ] > 0
    )

    return result


# =========================================================
# MOQ AND CASE PACK
# =========================================================

def round_up_to_multiple(
    quantity,
    multiple
):

    quantity = float(quantity)
    multiple = float(multiple)

    if quantity <= 0:
        return 0

    if multiple <= 0:
        return int(math.ceil(quantity))

    return int(
        math.ceil(
            quantity / multiple
        )
        * multiple
    )


def apply_order_constraints(df):

    print("Applying MOQ and case-pack constraints...")

    result = df.copy()

    result[
        "moq"
    ] = pd.to_numeric(
        result["moq"],
        errors="coerce"
    ).fillna(0.0)

    result[
        "case_pack"
    ] = pd.to_numeric(
        result["case_pack"],
        errors="coerce"
    ).fillna(1.0)

    result[
        "case_pack"
    ] = result[
        "case_pack"
    ].clip(
        lower=1.0
    )

    constrained = []

    for _, row in result.iterrows():

        need = int(
            row[
                "base_order_requirement"
            ]
        )

        if need <= 0:

            constrained.append(0)
            continue

        qty = max(
            need,
            int(
                math.ceil(
                    row["moq"]
                )
            )
        )

        qty = round_up_to_multiple(
            qty,
            row["case_pack"]
        )

        constrained.append(
            qty
        )

    result[
        "recommended_order_qty"
    ] = constrained

    result[
        "constraint_added_qty"
    ] = (
        result[
            "recommended_order_qty"
        ]
        - result[
            "base_order_requirement"
        ]
    ).clip(
        lower=0
    )

    result[
        "recommended_order_value"
    ] = (
        result[
            "recommended_order_qty"
        ]
        * pd.to_numeric(
            result["unit_cost"],
            errors="coerce"
        ).fillna(0.0)
    )

    return result


# =========================================================
# POST-ORDER POSITION
# =========================================================

def calculate_post_order_position(df):

    print("Calculating projected post-order position...")

    result = df.copy()

    result[
        "projected_position_after_order"
    ] = (
        result[
            "timely_inventory_position"
        ]
        + result[
            "recommended_order_qty"
        ]
    )

    result[
        "projected_units_above_reorder_point"
    ] = (
        result[
            "projected_position_after_order"
        ]
        - result[
            "reorder_point"
        ]
    )

    result[
        "projected_weeks_after_order"
    ] = np.where(
        result[
            "planning_weekly_demand"
        ] > 0,
        result[
            "projected_position_after_order"
        ]
        / result[
            "planning_weekly_demand"
        ],
        np.inf
    )

    return result


# =========================================================
# PRIORITY
# =========================================================

def calculate_priority(df):

    print("Assigning procurement priority...")

    result = df.copy()

    priority = []

    for _, row in result.iterrows():

        qty = row[
            "recommended_order_qty"
        ]

        weeks = row[
            "weeks_until_stockout"
        ]

        stockout_before_lt = bool(
            row[
                "stockout_before_replenishment"
            ]
        )

        supplier_risk = row[
            "supplier_risk_class"
        ]

        at_risk_qty = row[
            "at_risk_open_po_qty"
        ]

        late_to_need_qty = row[
            "late_to_need_open_po_qty"
        ]

        if qty <= 0:

            if (
                at_risk_qty > 0
                or late_to_need_qty > 0
            ):
                priority.append(
                    "monitor"
                )
            else:
                priority.append(
                    "no_action"
                )

            continue

        if (
            weeks <= VERY_URGENT_WEEKS
            or (
                stockout_before_lt
                and supplier_risk
                == "critical"
            )
        ):

            priority.append(
                "critical"
            )

        elif (
            weeks <= URGENT_WEEKS
            or stockout_before_lt
            or supplier_risk
            == "critical"
        ):

            priority.append(
                "high"
            )

        else:

            priority.append(
                "standard"
            )

    result[
        "procurement_priority"
    ] = priority

    priority_rank = {
        "critical": 1,
        "high": 2,
        "standard": 3,
        "monitor": 4,
        "no_action": 5,
    }

    result[
        "priority_rank"
    ] = result[
        "procurement_priority"
    ].map(
        priority_rank
    )

    return result


# =========================================================
# RECOMMENDATION ACTION
# =========================================================

def determine_action(row):

    qty = int(
        row[
            "recommended_order_qty"
        ]
    )

    risk_qty = int(
        row[
            "risk_exposure_qty"
        ]
    )

    late_qty = int(
        row[
            "late_to_need_open_po_qty"
        ]
    )

    if qty > 0:

        if row[
            "procurement_priority"
        ] == "critical":

            return "place order urgently"

        if row[
            "procurement_priority"
        ] == "high":

            return "place order"

        return "plan replenishment order"

    if risk_qty > 0:

        return "monitor open PO risk"

    if late_qty > 0:

        return "expedite or review open PO"

    return "no purchase required"


def build_recommendation_reason(row):

    reasons = []

    qty = int(
        row[
            "recommended_order_qty"
        ]
    )

    if qty > 0:

        reasons.append(
            f"timely supply is short by "
            f"{int(row['base_order_requirement']):,} units"
        )

    if bool(
        row[
            "stockout_before_replenishment"
        ]
    ):

        reasons.append(
            "projected stockout occurs before "
            "normal replenishment lead time"
        )

    if (
        row[
            "late_to_need_open_po_qty"
        ] > 0
    ):

        reasons.append(
            f"{int(row['late_to_need_open_po_qty']):,} "
            "open-PO units are expected after "
            "projected stockout"
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
            "risk_exposure_qty"
        ] > 0
    ):

        reasons.append(
            f"{int(row['risk_exposure_qty']):,} "
            "additional units depend on risky supply"
        )

    if (
        row[
            "supplier_risk_class"
        ] == "critical"
    ):

        reasons.append(
            "supplier risk is critical"
        )

    if (
        row[
            "constraint_added_qty"
        ] > 0
    ):

        reasons.append(
            f"MOQ/case-pack rounding added "
            f"{int(row['constraint_added_qty']):,} units"
        )

    if (
        row[
            "model_source"
        ] == "fallback"
    ):

        reasons.append(
            "forecast uses controlled fallback model"
        )

    if not reasons:

        reasons.append(
            "available and incoming supply "
            "covers current inventory policy"
        )

    return "; ".join(reasons)


def add_recommendations(df):

    print("Building procurement recommendations...")

    result = df.copy()

    result[
        "procurement_action"
    ] = result.apply(
        determine_action,
        axis=1
    )

    result[
        "procurement_reason"
    ] = result.apply(
        build_recommendation_reason,
        axis=1
    )

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate_outputs(
    recommendations,
    po_analysis
):

    section(
        "PROCUREMENT RECOMMENDATION VALIDATION"
    )

    assert len(
        recommendations
    ) == 200

    check(
        "Exactly 200 SKU recommendation rows"
    )

    assert (
        recommendations["sku"].nunique()
        == 200
    )

    check(
        "One recommendation row per SKU"
    )

    assert (
        recommendations[
            "recommended_order_qty"
        ] >= 0
    ).all()

    check(
        "Recommended order quantities are non-negative"
    )

    assert (
        recommendations[
            "recommended_order_value"
        ] >= 0
    ).all()

    check(
        "Recommended order values are non-negative"
    )

    # Open PO reconciliation.
    total_open_po_source = (
        po_analysis[
            "open_qty"
        ].sum()
    )

    total_open_po_recommendation = (
        recommendations[
            "open_po_qty_check"
        ].sum()
    )

    assert np.isclose(
        total_open_po_source,
        total_open_po_recommendation
    )

    check(
        f"Open PO quantities reconcile: "
        f"{total_open_po_source:,.0f} units"
    )

    # Timing buckets must reconcile.
    timing_total = (
        recommendations[
            "timely_open_po_qty"
        ]
        + recommendations[
            "late_to_need_open_po_qty"
        ]
    ).sum()

    assert np.isclose(
        timing_total,
        total_open_po_source
    )

    check(
        "Timely + late-to-need PO quantities reconcile"
    )

    # MOQ validation.
    ordered = recommendations[
        "recommended_order_qty"
    ] > 0

    assert (
        recommendations.loc[
            ordered,
            "recommended_order_qty"
        ]
        >= recommendations.loc[
            ordered,
            "moq"
        ]
    ).all()

    check(
        "All positive recommendations satisfy MOQ"
    )

    # Case-pack validation.
    case_pack = (
        recommendations.loc[
            ordered,
            "case_pack"
        ]
        .astype(int)
    )

    order_qty = (
        recommendations.loc[
            ordered,
            "recommended_order_qty"
        ]
        .astype(int)
    )

    assert (
        order_qty
        % case_pack
        == 0
    ).all()

    check(
        "All positive recommendations satisfy case pack"
    )

    assert (
        recommendations[
            "projected_position_after_order"
        ]
        + 1e-9
        >= np.where(
            recommendations[
                "recommended_order_qty"
            ] > 0,
            recommendations[
                "reorder_point"
            ],
            recommendations[
                "projected_position_after_order"
            ]
        )
    ).all()

    check(
        "Positive orders restore timely position "
        "to at least reorder point"
    )

    valid_priorities = {
        "critical",
        "high",
        "standard",
        "monitor",
        "no_action",
    }

    assert set(
        recommendations[
            "procurement_priority"
        ].unique()
    ).issubset(
        valid_priorities
    )

    check(
        "Procurement priorities valid"
    )

    assert recommendations[
        "procurement_reason"
    ].notna().all()

    check(
        "Every SKU has an explainable recommendation"
    )

    print(
        "\nAll Phase 7B validation checks passed."
    )


# =========================================================
# SELECT OUTPUT COLUMNS
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
        "safety_stock",
        "reorder_point",

        "on_hand",
        "allocated",
        "backordered",
        "net_available",

        "weeks_until_stockout",
        "projected_stockout_date",
        "stockout_before_replenishment",

        "total_open_po_qty",
        "open_po_line_count",
        "timely_open_po_qty",
        "timely_low_risk_open_po_qty",
        "late_to_need_open_po_qty",
        "at_risk_open_po_qty",

        "overdue_open_po_lines",
        "partial_open_po_lines",
        "delayed_expected_lines",

        "next_open_po_expected_date",
        "latest_open_po_expected_date",

        "policy_inventory_position",
        "timely_inventory_position",
        "low_risk_inventory_position",

        "gross_replenishment_need",
        "timely_replenishment_need",
        "low_risk_replenishment_need",
        "risk_exposure_qty",

        "base_order_requirement",
        "recommended_order_qty",
        "constraint_added_qty",
        "recommended_order_value",

        "projected_position_after_order",
        "projected_units_above_reorder_point",
        "projected_weeks_after_order",

        "supplier_risk_score",
        "supplier_risk_class",

        "inventory_policy_status",
        "procurement_priority",
        "priority_rank",
        "procurement_action",
        "procurement_reason",
    ]

    return df[columns].copy()


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(
    recommendations,
    po_analysis
):

    section(
        "SAVING PROCUREMENT RECOMMENDATIONS"
    )

    output = recommendations.copy()

    date_columns = [
        "projected_stockout_date",
        "next_open_po_expected_date",
        "latest_open_po_expected_date",
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

    po_output = po_analysis.copy()

    for column in [
        "order_date",
        "promised_date",
        "expected_date",
        "snapshot_date",
        "projected_stockout_date",
    ]:

        if column in po_output.columns:

            po_output[column] = pd.to_datetime(
                po_output[column],
                errors="coerce"
            ).dt.strftime(
                "%Y-%m-%d"
            )

    po_output.to_csv(
        OPEN_PO_ANALYSIS_PATH,
        index=False
    )

    check(
        OUTPUT_PATH.name
    )

    check(
        OPEN_PO_ANALYSIS_PATH.name
    )


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(df):

    section(
        "PROCUREMENT RECOMMENDATION SUMMARY"
    )

    order_mask = (
        df[
            "recommended_order_qty"
        ] > 0
    )

    print(
        f"\nSKUs evaluated: "
        f"{len(df):,}"
    )

    print(
        f"SKUs with purchase recommendations: "
        f"{order_mask.sum():,}"
    )

    print(
        f"Recommended units: "
        f"{df['recommended_order_qty'].sum():,.0f}"
    )

    print(
        f"Recommended purchase value: "
        f"${df['recommended_order_value'].sum():,.2f}"
    )

    print(
        f"\nOpen PO units: "
        f"{df['total_open_po_qty'].sum():,.0f}"
    )

    print(
        f"Timely open PO units: "
        f"{df['timely_open_po_qty'].sum():,.0f}"
    )

    print(
        f"Late-to-need open PO units: "
        f"{df['late_to_need_open_po_qty'].sum():,.0f}"
    )

    print(
        f"At-risk open PO units: "
        f"{df['at_risk_open_po_qty'].sum():,.0f}"
    )

    print(
        "\nProcurement priority:"
    )

    print(
        df[
            "procurement_priority"
        ].value_counts()
    )

    print(
        "\nProcurement action:"
    )

    print(
        df[
            "procurement_action"
        ].value_counts()
    )

    section(
        "RECOMMENDATIONS BY SUPPLIER RISK",
        width=120
    )

    supplier_summary = (
        df.groupby(
            "supplier_risk_class"
        )
        .agg(
            sku_count=(
                "sku",
                "count"
            ),
            order_skus=(
                "recommended_order_qty",
                lambda x: int(
                    (x > 0).sum()
                )
            ),
            recommended_units=(
                "recommended_order_qty",
                "sum"
            ),
            recommended_value=(
                "recommended_order_value",
                "sum"
            ),
            at_risk_open_units=(
                "at_risk_open_po_qty",
                "sum"
            ),
        )
        .round(2)
    )

    print(
        supplier_summary.to_string()
    )

    section(
        "TOP PROCUREMENT RECOMMENDATIONS",
        width=175
    )

    columns = [
    "sku",
    "product_name",
    "supplier_id",
    "supplier_risk_class",
    "final_demand_class",
    "planning_weekly_demand",
    "weeks_until_stockout",
    "reorder_point",
    "net_available",
    "timely_open_po_qty",
    "late_to_need_open_po_qty",
    "at_risk_open_po_qty",
    "base_order_requirement",
    "recommended_order_qty",
    "recommended_order_value",
    "procurement_priority",
    "priority_rank",
]

    top = (
        df[
            columns
        ]
        .sort_values(
            [
                "priority_rank",
                "recommended_order_value",
            ],
            ascending=[
                True,
                False,
            ]
        )
        .head(20)
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

    policy, open_pos, planning = (
        load_data()
    )

    validate_inputs(
        policy,
        open_pos,
        planning
    )

    open_pos = prepare_open_pos(
        open_pos
    )

    policy = add_snapshot_date(
        policy,
        planning
    )

    policy = calculate_exposure_timing(
        policy
    )

    po_analysis = analyze_open_po_timing(
        policy,
        open_pos
    )

    po_summary = aggregate_open_po_timing(
        po_analysis
    )

    recommendations = add_open_po_analysis(
        policy,
        po_summary
    )

    recommendations = calculate_supply_adequacy(
        recommendations
    )

    recommendations = calculate_procurement_requirement(
        recommendations
    )

    recommendations = apply_order_constraints(
        recommendations
    )

    recommendations = calculate_post_order_position(
        recommendations
    )

    recommendations = calculate_priority(
        recommendations
    )

    recommendations = add_recommendations(
        recommendations
    )

    validate_outputs(
        recommendations,
        po_analysis
    )

    output = select_output_columns(
        recommendations
    )

    save_outputs(
        output,
        po_analysis
    )

    print_summary(
        output
    )

    section(
        "PHASE 7B COMPLETE"
    )


if __name__ == "__main__":
    main()