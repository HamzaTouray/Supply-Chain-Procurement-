from pathlib import Path
import math

import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WEEKLY_PLAN_PATH = (
    PROCESSED_DIR / "time_phased_inventory_plan.csv"
)

SUMMARY_PATH = (
    PROCESSED_DIR / "time_phased_planning_summary.csv"
)

RAW_ORDERS_PATH = (
    PROCESSED_DIR / "planned_orders.csv"
)

OUTPUT_SKU_ORDERS = (
    PROCESSED_DIR / "consolidated_sku_orders.csv"
)

OUTPUT_SUPPLIER_POS = (
    PROCESSED_DIR / "supplier_po_proposals.csv"
)

OUTPUT_SUMMARY = (
    PROCESSED_DIR / "consolidation_summary.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

DAYS_PER_WEEK = 7.0

# Review cycle by demand behavior.
REVIEW_CYCLE_WEEKS = {
    "stable": 4,
    "growth": 3,
    "volatile": 3,
    "seasonal": 4,
    "intermittent": 6,
    "new_product": 2,
    "declining": 6,
}

DEFAULT_REVIEW_CYCLE_WEEKS = 4

ORDER_THIS_WEEK_DAYS = 7
ORDER_SOON_DAYS = 28


# =========================================================
# DISPLAY HELPERS
# =========================================================

def section(title, width=115):
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def check(message):
    print(f"✓ {message}")


# =========================================================
# GENERAL HELPERS
# =========================================================

def round_up_to_multiple(quantity, multiple):

    quantity = float(quantity)
    multiple = float(multiple)

    if quantity <= 0:
        return 0

    if multiple <= 0:
        return int(math.ceil(quantity))

    return int(
        math.ceil(quantity / multiple)
        * multiple
    )


def apply_constraints(requirement, moq, case_pack):

    if requirement <= 0:
        return 0

    moq = max(float(moq), 0.0)
    case_pack = max(float(case_pack), 1.0)

    quantity = max(
        float(requirement),
        moq
    )

    return round_up_to_multiple(
        quantity,
        case_pack
    )


def review_cycle_for_class(demand_class):

    demand_class = (
        str(demand_class)
        .strip()
        .lower()
    )

    return REVIEW_CYCLE_WEEKS.get(
        demand_class,
        DEFAULT_REVIEW_CYCLE_WEEKS
    )


def release_status(
    release_date,
    snapshot_date
):

    if pd.isna(release_date):
        return "NO ACTION"

    days_until_release = (
        release_date - snapshot_date
    ).total_seconds() / 86400.0

    if days_until_release <= 0:
        return "ORDER NOW"

    if days_until_release <= ORDER_THIS_WEEK_DAYS:
        return "ORDER THIS WEEK"

    if days_until_release <= ORDER_SOON_DAYS:
        return "ORDER SOON"

    return "PLANNED ORDER"


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    section(
        "PROCUREMENT AI - ORDER CONSOLIDATION & EXECUTABLE PO ENGINE"
    )

    print("\nLoading Phase 7E inputs...")

    weekly = pd.read_csv(
        WEEKLY_PLAN_PATH
    )

    summary = pd.read_csv(
        SUMMARY_PATH
    )

    raw_orders = pd.read_csv(
        RAW_ORDERS_PATH
    )

    check(
        f"time_phased_inventory_plan "
        f"{len(weekly):>7,} rows"
    )

    check(
        f"time_phased_planning_summary "
        f"{len(summary):>5,} rows"
    )

    check(
        f"raw planned orders "
        f"{len(raw_orders):>5,} rows"
    )

    return (
        weekly,
        summary,
        raw_orders
    )


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_inputs(
    weekly,
    summary,
    raw_orders
):

    print("Validating Phase 7E inputs...")

    weekly_required = {
        "sku",
        "product_name",
        "category",
        "supplier_id",
        "forecast_week",
        "forecast_horizon_week",
        "final_demand_class",
        "forecast_demand",
        "beginning_projected_inventory",
        "existing_po_receipt_qty",
        "existing_po_at_risk_qty",
        "safety_stock",
        "unit_cost",
        "supplier_risk_score",
        "supplier_risk_class",
    }

    summary_required = {
        "sku",
        "product_name",
        "category",
        "supplier_id",
        "unit_cost",
        "moq",
        "case_pack",
        "service_level_target",
        "planning_lead_time_weeks",
        "safety_stock",
        "reorder_point",
        "net_available",
        "supplier_risk_score",
        "supplier_risk_class",
        "forecast_26w_units",
        "existing_po_receipts_26w",
        "planned_order_units_26w",
        "planned_order_value_26w",
    }

    raw_required = {
        "sku",
        "required_receipt_week",
        "planned_release_date",
        "planned_order_qty",
        "planned_order_value",
        "release_status",
    }

    datasets = [
        (
            "weekly plan",
            weekly_required,
            weekly
        ),
        (
            "planning summary",
            summary_required,
            summary
        ),
        (
            "raw planned orders",
            raw_required,
            raw_orders
        ),
    ]

    for name, required, df in datasets:

        missing = (
            required
            - set(df.columns)
        )

        if missing:

            raise ValueError(
                f"{name} missing columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

    if len(weekly) != 5200:

        raise ValueError(
            "Expected 5,200 weekly planning rows"
        )

    if weekly["sku"].nunique() != 200:

        raise ValueError(
            "Expected exactly 200 SKUs"
        )

    if len(summary) != 200:

        raise ValueError(
            "Expected exactly 200 SKU summary rows"
        )

    counts = (
        weekly.groupby("sku")
        .size()
    )

    if not (
        counts == 26
    ).all():

        raise ValueError(
            "Every SKU must have exactly 26 weeks"
        )

    check(
        "Required Phase 7E columns present"
    )

    check(
        "Weekly planning grid has 5,200 rows"
    )

    check(
        "Every SKU contains exactly 26 weeks"
    )

    check(
        "Planning summary has 200 SKUs"
    )


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(
    weekly,
    summary,
    raw_orders
):

    print(
        "Preparing consolidation data..."
    )

    weekly = weekly.copy()
    summary = summary.copy()
    raw_orders = raw_orders.copy()

    weekly[
        "forecast_week"
    ] = pd.to_datetime(
        weekly["forecast_week"],
        errors="coerce"
    )

    raw_orders[
        "required_receipt_week"
    ] = pd.to_datetime(
        raw_orders[
            "required_receipt_week"
        ],
        errors="coerce"
    )

    raw_orders[
        "planned_release_date"
    ] = pd.to_datetime(
        raw_orders[
            "planned_release_date"
        ],
        errors="coerce"
    )

    snapshot_date = (
        weekly[
            "forecast_week"
        ].min()
    )

    if pd.isna(snapshot_date):

        raise ValueError(
            "Unable to determine planning snapshot date"
        )

    print(
        f"Planning snapshot date: "
        f"{snapshot_date.date()}"
    )

    return (
        weekly,
        summary,
        raw_orders,
        snapshot_date
    )


# =========================================================
# CONSOLIDATED SKU ORDER ENGINE
# =========================================================

def build_consolidated_orders(
    weekly,
    summary,
    snapshot_date
):

    print(
        "Building consolidated SKU replenishment orders..."
    )

    results = []

    summary_lookup = (
        summary.set_index("sku")
    )

    for sku, group in weekly.groupby(
        "sku",
        sort=False
    ):

        group = (
            group
            .sort_values(
                "forecast_horizon_week"
            )
            .reset_index(drop=True)
            .copy()
        )

        item = (
            summary_lookup.loc[sku]
        )

        demand_class = str(
            group.loc[
                0,
                "final_demand_class"
            ]
        )

        review_weeks = (
            review_cycle_for_class(
                demand_class
            )
        )

        lead_time_weeks = float(
            item[
                "planning_lead_time_weeks"
            ]
        )

        safety_stock = float(
            item["safety_stock"]
        )

        moq = float(
            item["moq"]
        )

        case_pack = float(
            item["case_pack"]
        )

        unit_cost = float(
            item["unit_cost"]
        )

        # -------------------------------------------------
        # Start with actual inventory position at the
        # beginning of the planning horizon.
        #
        # We intentionally DO NOT carry the planned receipts
        # created by Phase 7D.
        # -------------------------------------------------

        projected_inventory = float(
            group.loc[
                0,
                "beginning_projected_inventory"
            ]
        )

        i = 0

        while i < len(group):

            row = group.loc[i]

            existing_receipt = float(
                row[
                    "existing_po_receipt_qty"
                ]
            )

            forecast_demand = float(
                row[
                    "forecast_demand"
                ]
            )

            # Inventory after this week's existing PO
            # receipt and demand.
            projected_before_order = (
                projected_inventory
                + existing_receipt
                - forecast_demand
            )

            # -------------------------------------------------
            # NO ORDER REQUIRED
            # -------------------------------------------------

            if (
                projected_before_order
                >= safety_stock
            ):

                projected_inventory = (
                    projected_before_order
                )

                i += 1
                continue

            # -------------------------------------------------
            # CONSOLIDATED REQUIREMENT
            #
            # Inventory has fallen below safety stock.
            #
            # We now determine how much must arrive in this
            # week so that inventory remains at or above
            # safety stock throughout the entire review cycle.
            #
            # Existing PO receipts are respected in the exact
            # week they arrive.
            # -------------------------------------------------

            coverage_end_index = min(
                i + review_weeks - 1,
                len(group) - 1
            )

            simulated_inventory = (
                projected_before_order
            )

            minimum_inventory = (
                simulated_inventory
            )

            cumulative_future_demand = 0.0

            cumulative_future_receipts = 0.0

            for j in range(
                i + 1,
                coverage_end_index + 1
            ):

                future_row = (
                    group.loc[j]
                )

                future_receipt = float(
                    future_row[
                        "existing_po_receipt_qty"
                    ]
                )

                future_demand = float(
                    future_row[
                        "forecast_demand"
                    ]
                )

                cumulative_future_receipts += (
                    future_receipt
                )

                cumulative_future_demand += (
                    future_demand
                )

                simulated_inventory += (
                    future_receipt
                    - future_demand
                )

                minimum_inventory = min(
                    minimum_inventory,
                    simulated_inventory
                )

            # -------------------------------------------------
            # Quantity needed today to lift the lowest point
            # in the review cycle back to safety stock.
            # -------------------------------------------------

            raw_requirement = max(
                safety_stock
                - minimum_inventory,
                0.0
            )

            order_qty = (
                apply_constraints(
                    raw_requirement,
                    moq,
                    case_pack
                )
            )

            if order_qty <= 0:

                projected_inventory = (
                    projected_before_order
                )

                i += 1
                continue

            required_receipt_week = (
                row[
                    "forecast_week"
                ]
            )

            release_date = (
                required_receipt_week
                - pd.to_timedelta(
                    lead_time_weeks
                    * DAYS_PER_WEEK,
                    unit="D"
                )
            )

            status = (
                release_status(
                    release_date,
                    snapshot_date
                )
            )

            order_value = (
                order_qty
                * unit_cost
            )

            results.append(
                {
                    "sku":
                        sku,

                    "product_name":
                        row[
                            "product_name"
                        ],

                    "category":
                        row[
                            "category"
                        ],

                    "supplier_id":
                        row[
                            "supplier_id"
                        ],

                    "supplier_risk_class":
                        row[
                            "supplier_risk_class"
                        ],

                    "supplier_risk_score":
                        float(
                            row[
                                "supplier_risk_score"
                            ]
                        ),

                    "final_demand_class":
                        demand_class,

                    "review_cycle_weeks":
                        review_weeks,

                    "planning_lead_time_weeks":
                        lead_time_weeks,

                    "required_receipt_week":
                        required_receipt_week,

                    "planned_release_date":
                        release_date,

                    "forecast_horizon_week":
                        int(
                            row[
                                "forecast_horizon_week"
                            ]
                        ),

                    "projected_inventory_before_order":
                        projected_before_order,

                    "minimum_inventory_during_review":
                        minimum_inventory,

                    "safety_stock":
                        safety_stock,

                    "review_cycle_demand":
                        cumulative_future_demand,

                    "review_cycle_existing_receipts":
                        cumulative_future_receipts,

                    "raw_consolidated_requirement":
                        raw_requirement,

                    "moq":
                        moq,

                    "case_pack":
                        case_pack,

                    "recommended_order_qty":
                        order_qty,

                    "unit_cost":
                        unit_cost,

                    "recommended_order_value":
                        order_value,

                    "release_status":
                        status,
                }
            )

            # -------------------------------------------------
            # The new supply is required in the current week.
            # Add it to inventory and continue simulation.
            # -------------------------------------------------

            projected_inventory = (
                projected_before_order
                + order_qty
            )

            i += 1

    orders = pd.DataFrame(
        results
    )

    if orders.empty:

        raise ValueError(
            "No consolidated orders were generated"
        )

    return orders


# =========================================================
# RE-SIMULATE CONSOLIDATED INVENTORY
# =========================================================

def validate_consolidated_inventory(
    weekly,
    consolidated_orders
):

    print(
        "Re-simulating inventory using consolidated orders..."
    )

    receipt_map = {}

    for _, row in (
        consolidated_orders.iterrows()
    ):

        key = (
            row["sku"],
            pd.Timestamp(
                row[
                    "required_receipt_week"
                ]
            )
        )

        receipt_map[key] = (
            receipt_map.get(
                key,
                0.0
            )
            + float(
                row[
                    "recommended_order_qty"
                ]
            )
        )

    validation_rows = []

    for sku, group in weekly.groupby(
        "sku",
        sort=False
    ):

        group = (
            group
            .sort_values(
                "forecast_horizon_week"
            )
            .reset_index(drop=True)
            .copy()
        )

        projected_inventory = float(
            group.loc[
                0,
                "beginning_projected_inventory"
            ]
        )

        for _, row in (
            group.iterrows()
        ):

            week = pd.Timestamp(
                row[
                    "forecast_week"
                ]
            )

            beginning_inventory = (
                projected_inventory
            )

            existing_receipt = float(
                row[
                    "existing_po_receipt_qty"
                ]
            )

            consolidated_receipt = float(
                receipt_map.get(
                    (
                        sku,
                        week
                    ),
                    0.0
                )
            )

            forecast_demand = float(
                row[
                    "forecast_demand"
                ]
            )

            projected_inventory = (
                beginning_inventory
                + existing_receipt
                + consolidated_receipt
                - forecast_demand
            )

            safety_stock = float(
                row[
                    "safety_stock"
                ]
            )

            validation_rows.append(
                {
                    "sku":
                        sku,

                    "forecast_week":
                        week,

                    "forecast_horizon_week":
                        int(
                            row[
                                "forecast_horizon_week"
                            ]
                        ),

                    "beginning_inventory":
                        beginning_inventory,

                    "existing_po_receipt_qty":
                        existing_receipt,

                    "consolidated_receipt_qty":
                        consolidated_receipt,

                    "forecast_demand":
                        forecast_demand,

                    "projected_ending_inventory":
                        projected_inventory,

                    "safety_stock":
                        safety_stock,

                    "below_safety_stock":
                        projected_inventory
                        < safety_stock
                        - 1e-9,

                    "stockout":
                        projected_inventory
                        < -1e-9,
                }
            )

    return pd.DataFrame(
        validation_rows
    )


# =========================================================
# SUPPLIER PO PROPOSALS
# =========================================================

def build_supplier_po_proposals(
    orders,
    snapshot_date
):

    print(
        "Building supplier-level PO proposals..."
    )

    if orders.empty:

        return pd.DataFrame()

    df = orders.copy()

    df[
        "planned_release_date"
    ] = pd.to_datetime(
        df[
            "planned_release_date"
        ],
        errors="coerce"
    )

    # -----------------------------------------------------
    # For execution purposes, a release date in the past
    # means the requirement is actionable today.
    # -----------------------------------------------------

    df[
        "effective_release_date"
    ] = df[
        "planned_release_date"
    ].copy()

    df.loc[
        df[
            "effective_release_date"
        ] < snapshot_date,
        "effective_release_date"
    ] = snapshot_date

    # -----------------------------------------------------
    # Consolidate supplier orders into Monday-based
    # purchasing buckets.
    # -----------------------------------------------------

    df[
        "supplier_release_bucket"
    ] = (
        df[
            "effective_release_date"
        ]
        - pd.to_timedelta(
            df[
                "effective_release_date"
            ].dt.weekday,
            unit="D"
        )
    )

    supplier_pos = (
        df.groupby(
            [
                "supplier_id",
                "supplier_release_bucket",
                "release_status",
            ],
            as_index=False
        )
        .agg(
            sku_count=(
                "sku",
                "nunique"
            ),

            line_count=(
                "sku",
                "count"
            ),

            total_units=(
                "recommended_order_qty",
                "sum"
            ),

            total_value=(
                "recommended_order_value",
                "sum"
            ),

            earliest_required_receipt=(
                "required_receipt_week",
                "min"
            ),

            latest_required_receipt=(
                "required_receipt_week",
                "max"
            ),

            max_supplier_risk_score=(
                "supplier_risk_score",
                "max"
            ),
        )
    )

    supplier_pos[
        "po_proposal_id"
    ] = (
        supplier_pos[
            "supplier_id"
        ]
        + "-"
        + supplier_pos[
            "supplier_release_bucket"
        ].dt.strftime(
            "%Y%m%d"
        )
    )

    columns = [
        "po_proposal_id",
        "supplier_id",
        "supplier_release_bucket",
        "release_status",
        "sku_count",
        "line_count",
        "total_units",
        "total_value",
        "earliest_required_receipt",
        "latest_required_receipt",
        "max_supplier_risk_score",
    ]

    supplier_pos = (
        supplier_pos[
            columns
        ]
        .sort_values(
            [
                "supplier_release_bucket",
                "supplier_id",
            ]
        )
        .reset_index(drop=True)
    )

    return supplier_pos


# =========================================================
# SKU CONSOLIDATION SUMMARY
# =========================================================

def build_consolidation_summary(
    summary,
    orders,
    raw_orders
):

    print(
        "Building Phase 7E SKU summary..."
    )

    raw = (
        raw_orders.groupby(
            "sku",
            as_index=False
        )
        .agg(
            raw_7d_order_count=(
                "planned_order_qty",
                "count"
            ),

            raw_7d_order_units=(
                "planned_order_qty",
                "sum"
            ),

            raw_7d_order_value=(
                "planned_order_value",
                "sum"
            ),
        )
    )

    consolidated = (
        orders.groupby(
            "sku",
            as_index=False
        )
        .agg(
            consolidated_order_count=(
                "recommended_order_qty",
                "count"
            ),

            consolidated_order_units=(
                "recommended_order_qty",
                "sum"
            ),

            consolidated_order_value=(
                "recommended_order_value",
                "sum"
            ),

            first_consolidated_receipt=(
                "required_receipt_week",
                "min"
            ),

            first_consolidated_release=(
                "planned_release_date",
                "min"
            ),
        )
    )

    result = (
        summary
        .merge(
            raw,
            on="sku",
            how="left",
            validate="one_to_one"
        )
        .merge(
            consolidated,
            on="sku",
            how="left",
            validate="one_to_one"
        )
    )

    zero_columns = [
        "raw_7d_order_count",
        "raw_7d_order_units",
        "raw_7d_order_value",
        "consolidated_order_count",
        "consolidated_order_units",
        "consolidated_order_value",
    ]

    for column in zero_columns:

        result[column] = (
            result[column]
            .fillna(0)
        )

    result[
        "order_event_reduction"
    ] = (
        result[
            "raw_7d_order_count"
        ]
        - result[
            "consolidated_order_count"
        ]
    )

    result[
        "order_event_reduction_pct"
    ] = np.where(
        result[
            "raw_7d_order_count"
        ] > 0,

        result[
            "order_event_reduction"
        ]
        / result[
            "raw_7d_order_count"
        ],

        0.0
    )

    result[
        "unit_change_vs_7d"
    ] = (
        result[
            "consolidated_order_units"
        ]
        - result[
            "raw_7d_order_units"
        ]
    )

    result[
        "value_change_vs_7d"
    ] = (
        result[
            "consolidated_order_value"
        ]
        - result[
            "raw_7d_order_value"
        ]
    )

    return result


# =========================================================
# OUTPUT VALIDATION
# =========================================================

def validate_outputs(
    orders,
    supplier_pos,
    sku_summary,
    inventory_validation,
    raw_orders
):

    section(
        "ORDER CONSOLIDATION VALIDATION"
    )

    # -----------------------------------------------------
    # SKU summary
    # -----------------------------------------------------

    assert len(
        sku_summary
    ) == 200

    check(
        "Exactly 200 SKU summary rows"
    )

    # -----------------------------------------------------
    # Positive quantities
    # -----------------------------------------------------

    assert (
        orders[
            "recommended_order_qty"
        ] > 0
    ).all()

    check(
        "All consolidated orders have positive quantities"
    )

    # -----------------------------------------------------
    # MOQ / case pack
    # -----------------------------------------------------

    for _, row in (
        orders.iterrows()
    ):

        qty = int(
            row[
                "recommended_order_qty"
            ]
        )

        moq = int(
            math.ceil(
                float(
                    row["moq"]
                )
            )
        )

        case_pack = max(
            int(
                float(
                    row["case_pack"]
                )
            ),
            1
        )

        if qty < moq:

            raise AssertionError(
                f"{row['sku']} "
                f"quantity {qty} below MOQ {moq}"
            )

        if qty % case_pack != 0:

            raise AssertionError(
                f"{row['sku']} "
                f"quantity {qty} not divisible "
                f"by case pack {case_pack}"
            )

    check(
        "All consolidated orders satisfy MOQ and case pack"
    )

    # -----------------------------------------------------
    # Inventory protection
    # -----------------------------------------------------

    breach_rows = (
        inventory_validation[
            inventory_validation[
                "below_safety_stock"
            ]
        ]
    )

    stockout_rows = (
        inventory_validation[
            inventory_validation[
                "stockout"
            ]
        ]
    )

    if len(breach_rows) > 0:

        print(
            "\nSafety-stock breach examples:"
        )

        print(
            breach_rows[
                [
                    "sku",
                    "forecast_week",
                    "projected_ending_inventory",
                    "safety_stock",
                ]
            ]
            .head(15)
            .to_string(
                index=False
            )
        )

        raise AssertionError(
            f"Consolidated plan leaves "
            f"{len(breach_rows)} "
            f"safety-stock breaches"
        )

    check(
        "Consolidated plan protects safety stock "
        "for all 5,200 SKU-weeks"
    )

    if len(stockout_rows) > 0:

        raise AssertionError(
            f"Consolidated plan leaves "
            f"{len(stockout_rows)} stockouts"
        )

    check(
        "Consolidated plan produces zero projected stockouts"
    )

    # -----------------------------------------------------
    # Consolidation effectiveness
    # -----------------------------------------------------

    raw_count = len(
        raw_orders
    )

    consolidated_count = len(
        orders
    )

    if consolidated_count >= raw_count:

        raise AssertionError(
            "Consolidation did not reduce "
            "the number of order events"
        )

    reduction = (
        raw_count
        - consolidated_count
    )

    reduction_pct = (
        reduction
        / raw_count
    )

    check(
        f"Order events reduced from "
        f"{raw_count:,} to "
        f"{consolidated_count:,} "
        f"({reduction_pct:.1%} reduction)"
    )

    # -----------------------------------------------------
    # Supplier proposal reconciliation
    # -----------------------------------------------------

    order_units = (
        orders[
            "recommended_order_qty"
        ].sum()
    )

    supplier_units = (
        supplier_pos[
            "total_units"
        ].sum()
        if not supplier_pos.empty
        else 0.0
    )

    if not np.isclose(
        order_units,
        supplier_units
    ):

        raise AssertionError(
            "Supplier PO proposal units "
            "do not reconcile"
        )

    check(
        "Supplier PO proposal units reconcile "
        "to SKU order units"
    )

    order_value = (
        orders[
            "recommended_order_value"
        ].sum()
    )

    supplier_value = (
        supplier_pos[
            "total_value"
        ].sum()
        if not supplier_pos.empty
        else 0.0
    )

    if not np.isclose(
        order_value,
        supplier_value
    ):

        raise AssertionError(
            "Supplier PO proposal value "
            "does not reconcile"
        )

    check(
        "Supplier PO proposal value reconciles "
        "to SKU order value"
    )

    print(
        "\nAll Phase 7E validation checks passed."
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(
    orders,
    supplier_pos,
    sku_summary
):

    section(
        "SAVING PHASE 7E OUTPUTS"
    )

    orders_out = (
        orders.copy()
    )

    for column in [
        "required_receipt_week",
        "planned_release_date",
    ]:

        orders_out[
            column
        ] = pd.to_datetime(
            orders_out[column],
            errors="coerce"
        ).dt.strftime(
            "%Y-%m-%d"
        )

    orders_out.to_csv(
        OUTPUT_SKU_ORDERS,
        index=False
    )

    supplier_out = (
        supplier_pos.copy()
    )

    if not supplier_out.empty:

        for column in [
            "supplier_release_bucket",
            "earliest_required_receipt",
            "latest_required_receipt",
        ]:

            supplier_out[
                column
            ] = pd.to_datetime(
                supplier_out[column],
                errors="coerce"
            ).dt.strftime(
                "%Y-%m-%d"
            )

    supplier_out.to_csv(
        OUTPUT_SUPPLIER_POS,
        index=False
    )

    summary_out = (
        sku_summary.copy()
    )

    date_columns = [
        "first_safety_stock_breach_week",
        "first_stockout_week_without_plan",
        "first_planned_receipt_week",
        "first_planned_release_date",
        "first_consolidated_receipt",
        "first_consolidated_release",
    ]

    for column in date_columns:

        if column in summary_out.columns:

            summary_out[
                column
            ] = pd.to_datetime(
                summary_out[column],
                errors="coerce"
            ).dt.strftime(
                "%Y-%m-%d"
            )

    summary_out.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    check(
        OUTPUT_SKU_ORDERS.name
    )

    check(
        OUTPUT_SUPPLIER_POS.name
    )

    check(
        OUTPUT_SUMMARY.name
    )


# =========================================================
# PRINT RESULTS
# =========================================================

def print_summary(
    orders,
    supplier_pos,
    sku_summary,
    raw_orders
):

    section(
        "PHASE 7E CONSOLIDATION SUMMARY"
    )

    raw_events = len(
        raw_orders
    )

    consolidated_events = len(
        orders
    )

    event_reduction = (
        raw_events
        - consolidated_events
    )

    event_reduction_pct = (
        event_reduction
        / raw_events
        if raw_events > 0
        else 0.0
    )

    raw_units = float(
        raw_orders[
            "planned_order_qty"
        ].sum()
    )

    consolidated_units = float(
        orders[
            "recommended_order_qty"
        ].sum()
    )

    raw_value = float(
        raw_orders[
            "planned_order_value"
        ].sum()
    )

    consolidated_value = float(
        orders[
            "recommended_order_value"
        ].sum()
    )

    print(
        f"\nRaw Phase 7D order events: "
        f"{raw_events:,}"
    )

    print(
        f"Consolidated SKU order events: "
        f"{consolidated_events:,}"
    )

    print(
        f"Order-event reduction: "
        f"{event_reduction:,} "
        f"({event_reduction_pct:.1%})"
    )

    print(
        f"\nRaw Phase 7D planned units: "
        f"{raw_units:,.0f}"
    )

    print(
        f"Consolidated planned units: "
        f"{consolidated_units:,.0f}"
    )

    print(
        f"Unit change: "
        f"{consolidated_units - raw_units:+,.0f}"
    )

    print(
        f"\nRaw Phase 7D planned value: "
        f"${raw_value:,.2f}"
    )

    print(
        f"Consolidated planned value: "
        f"${consolidated_value:,.2f}"
    )

    print(
        f"Value change: "
        f"${consolidated_value - raw_value:+,.2f}"
    )

    print(
        f"\nSupplier PO proposals: "
        f"{len(supplier_pos):,}"
    )

    print(
        f"SKUs with consolidated supply: "
        f"{orders['sku'].nunique():,}"
    )

    # =====================================================
    # RELEASE STATUS
    # =====================================================

    section(
        "RELEASE STATUS",
        width=105
    )

    release_summary = (
        orders.groupby(
            "release_status"
        )
        .agg(
            order_events=(
                "sku",
                "count"
            ),

            sku_count=(
                "sku",
                "nunique"
            ),

            units=(
                "recommended_order_qty",
                "sum"
            ),

            value=(
                "recommended_order_value",
                "sum"
            ),
        )
        .round(2)
        .sort_values(
            "value",
            ascending=False
        )
    )

    print(
        release_summary.to_string()
    )

    # =====================================================
    # DEMAND CLASS
    # =====================================================

    section(
        "CONSOLIDATION BY DEMAND CLASS",
        width=120
    )

    demand_summary = (
        orders.groupby(
            "final_demand_class"
        )
        .agg(
            order_events=(
                "sku",
                "count"
            ),

            sku_count=(
                "sku",
                "nunique"
            ),

            units=(
                "recommended_order_qty",
                "sum"
            ),

            value=(
                "recommended_order_value",
                "sum"
            ),

            avg_review_cycle_weeks=(
                "review_cycle_weeks",
                "mean"
            ),
        )
        .round(2)
        .sort_values(
            "value",
            ascending=False
        )
    )

    print(
        demand_summary.to_string()
    )

    # =====================================================
    # SUPPLIER PO PROPOSALS
    # =====================================================

    section(
        "SUPPLIER PO PROPOSALS - FIRST 30",
        width=145
    )

    if supplier_pos.empty:

        print(
            "No supplier PO proposals generated."
        )

    else:

        supplier_display = (
            supplier_pos
            .sort_values(
                [
                    "supplier_release_bucket",
                    "total_value",
                ],
                ascending=[
                    True,
                    False,
                ]
            )
            .head(30)
        )

        print(
            supplier_display.to_string(
                index=False
            )
        )

    # =====================================================
    # TOP SKU ORDERS
    # =====================================================

    section(
        "TOP CONSOLIDATED SKU ORDERS",
        width=175
    )

    top_columns = [
        "sku",
        "product_name",
        "supplier_id",
        "supplier_risk_class",
        "final_demand_class",
        "review_cycle_weeks",
        "required_receipt_week",
        "planned_release_date",
        "projected_inventory_before_order",
        "minimum_inventory_during_review",
        "safety_stock",
        "review_cycle_demand",
        "raw_consolidated_requirement",
        "recommended_order_qty",
        "recommended_order_value",
        "release_status",
    ]

    top = (
        orders[
            top_columns
        ]
        .sort_values(
            "recommended_order_value",
            ascending=False
        )
        .head(25)
    )

    print(
        top.to_string(
            index=False
        )
    )

    # =====================================================
    # BIGGEST EVENT REDUCTIONS
    # =====================================================

    section(
        "BIGGEST SKU ORDER-EVENT REDUCTIONS",
        width=130
    )

    reduction_columns = [
        "sku",
        "product_name",
        "supplier_id",
        "raw_7d_order_count",
        "consolidated_order_count",
        "order_event_reduction",
        "order_event_reduction_pct",
        "raw_7d_order_units",
        "consolidated_order_units",
        "unit_change_vs_7d",
    ]

    reduction_display = (
        sku_summary[
            reduction_columns
        ]
        .sort_values(
            [
                "order_event_reduction",
                "raw_7d_order_count",
            ],
            ascending=False
        )
        .head(25)
    )

    print(
        reduction_display.to_string(
            index=False
        )
    )


# =========================================================
# MAIN
# =========================================================

def main():

    (
        weekly,
        summary,
        raw_orders
    ) = load_data()

    validate_inputs(
        weekly,
        summary,
        raw_orders
    )

    (
        weekly,
        summary,
        raw_orders,
        snapshot_date
    ) = prepare_data(
        weekly,
        summary,
        raw_orders
    )

    consolidated_orders = (
        build_consolidated_orders(
            weekly,
            summary,
            snapshot_date
        )
    )

    inventory_validation = (
        validate_consolidated_inventory(
            weekly,
            consolidated_orders
        )
    )

    supplier_pos = (
        build_supplier_po_proposals(
            consolidated_orders,
            snapshot_date
        )
    )

    sku_summary = (
        build_consolidation_summary(
            summary,
            consolidated_orders,
            raw_orders
        )
    )

    validate_outputs(
        consolidated_orders,
        supplier_pos,
        sku_summary,
        inventory_validation,
        raw_orders
    )

    save_outputs(
        consolidated_orders,
        supplier_pos,
        sku_summary
    )

    print_summary(
        consolidated_orders,
        supplier_pos,
        sku_summary,
        raw_orders
    )

    section(
        "PHASE 7E COMPLETE"
    )


if __name__ == "__main__":
    main()