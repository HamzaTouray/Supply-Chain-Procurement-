from pathlib import Path
import math

import numpy as np
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FORECAST_PATH = (
    PROCESSED_DIR / "production_forecasts.csv"
)

POLICY_PATH = (
    PROCESSED_DIR / "inventory_policy.csv"
)

OPEN_PO_PATH = (
    PROCESSED_DIR / "open_purchase_orders_clean.csv"
)

PLANNING_MASTER_PATH = (
    PROCESSED_DIR / "planning_master.csv"
)

WEEKLY_OUTPUT_PATH = (
    PROCESSED_DIR / "time_phased_inventory_plan.csv"
)

SUMMARY_OUTPUT_PATH = (
    PROCESSED_DIR / "time_phased_planning_summary.csv"
)

PLANNED_ORDERS_PATH = (
    PROCESSED_DIR / "planned_orders.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

DAYS_PER_WEEK = 7.0
PLANNING_HORIZON_WEEKS = 26

ORDER_NOW_DAYS = 0
ORDER_THIS_WEEK_DAYS = 7
ORDER_SOON_DAYS = 28


# =========================================================
# DISPLAY HELPERS
# =========================================================

def section(title, width=110):
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
# ROUNDING
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


def apply_order_constraints(
    requirement,
    moq,
    case_pack
):

    if requirement <= 0:
        return 0

    moq = max(
        float(moq),
        0.0
    )

    case_pack = max(
        float(case_pack),
        1.0
    )

    quantity = max(
        float(requirement),
        moq
    )

    return round_up_to_multiple(
        quantity,
        case_pack
    )


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    section(
        "PROCUREMENT AI - 26-WEEK TIME-PHASED PLANNING ENGINE"
    )

    print("\nLoading Phase 7D inputs...")

    forecasts = pd.read_csv(
        FORECAST_PATH
    )

    policy = pd.read_csv(
        POLICY_PATH
    )

    open_pos = pd.read_csv(
        OPEN_PO_PATH
    )

    planning = pd.read_csv(
        PLANNING_MASTER_PATH
    )

    check(
        f"production_forecasts       "
        f"{len(forecasts):>8,} rows"
    )

    check(
        f"inventory_policy           "
        f"{len(policy):>8,} rows"
    )

    check(
        f"open_purchase_orders       "
        f"{len(open_pos):>8,} rows"
    )

    check(
        f"planning_master            "
        f"{len(planning):>8,} rows"
    )

    return (
        forecasts,
        policy,
        open_pos,
        planning
    )


# =========================================================
# VALIDATE INPUTS
# =========================================================

def validate_inputs(
    forecasts,
    policy,
    open_pos,
    planning
):

    print("Validating Phase 7D inputs...")

    forecast_required = {
        "sku",
        "forecast_week",
        "forecast_horizon_week",
        "forecast_demand",
        "production_model",
        "model_source",
        "final_demand_class",
        "forecast_readiness",
        "forecast_difficulty",
        "classification_confidence",
        "selected_wape",
        "selected_bias",
    }

    policy_required = {
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
        "on_hand",
        "allocated",
        "backordered",
        "net_available",
        "supplier_risk_score",
        "supplier_risk_class",
    }

    open_po_required = {
        "po_number",
        "supplier_id",
        "sku",
        "open_qty",
        "expected_date",
        "status",
        "days_late",
        "partial_receipt_flag",
        "overdue_flag",
        "expected_after_promise_flag",
    }

    planning_required = {
        "sku",
        "snapshot_date",
    }

    checks = [
        (
            "production_forecasts",
            forecast_required,
            forecasts
        ),
        (
            "inventory_policy",
            policy_required,
            policy
        ),
        (
            "open_purchase_orders",
            open_po_required,
            open_pos
        ),
        (
            "planning_master",
            planning_required,
            planning
        ),
    ]

    for name, required, df in checks:

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: "
                + ", ".join(sorted(missing))
            )

    if len(forecasts) != 5200:
        raise ValueError(
            f"Expected 5,200 forecast rows, "
            f"found {len(forecasts)}"
        )

    if forecasts["sku"].nunique() != 200:
        raise ValueError(
            "Forecast must contain exactly 200 SKUs"
        )

    counts = forecasts.groupby("sku").size()

    if not (counts == 26).all():
        raise ValueError(
            "Every SKU must have exactly 26 forecast weeks"
        )

    if policy["sku"].duplicated().any():
        raise ValueError(
            "Inventory policy contains duplicate SKUs"
        )

    if planning["sku"].duplicated().any():
        raise ValueError(
            "Planning master contains duplicate SKUs"
        )

    check("Required Phase 7D columns present")
    check("Forecast contains exactly 200 SKUs")
    check("Every SKU contains exactly 26 forecast weeks")
    check("Inventory policy contains one row per SKU")


# =========================================================
# PREPARE DATA
# =========================================================

def prepare_data(
    forecasts,
    policy,
    open_pos,
    planning
):

    print("Preparing time-phased planning data...")

    fc = forecasts.copy()

    fc["forecast_week"] = pd.to_datetime(
        fc["forecast_week"],
        errors="coerce"
    )

    fc["forecast_demand"] = pd.to_numeric(
        fc["forecast_demand"],
        errors="coerce"
    ).fillna(0.0)

    fc = fc.sort_values(
        [
            "sku",
            "forecast_horizon_week",
        ]
    ).reset_index(drop=True)

    po = open_pos.copy()

    po["expected_date"] = pd.to_datetime(
        po["expected_date"],
        errors="coerce"
    )

    po["open_qty"] = pd.to_numeric(
        po["open_qty"],
        errors="coerce"
    ).fillna(0.0)

    for column in [
        "partial_receipt_flag",
        "overdue_flag",
        "expected_after_promise_flag",
    ]:
        po[column] = normalize_boolean(
            po[column]
        )

    plan = planning[
        [
            "sku",
            "snapshot_date",
        ]
    ].copy()

    plan["snapshot_date"] = pd.to_datetime(
        plan["snapshot_date"],
        errors="coerce"
    )

    policy_data = policy.merge(
        plan,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    if policy_data["snapshot_date"].isna().any():
        raise ValueError(
            "Missing snapshot date"
        )

    return (
        fc,
        policy_data,
        po
    )


# =========================================================
# ASSIGN OPEN PO RECEIPTS TO FORECAST WEEKS
# =========================================================

def assign_po_receipt_weeks(
    forecasts,
    open_pos
):

    print("Assigning existing PO receipts to planning weeks...")

    calendar = (
        forecasts[
            [
                "forecast_horizon_week",
                "forecast_week",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "forecast_horizon_week"
        )
        .reset_index(drop=True)
    )

    if len(calendar) != 26:
        raise ValueError(
            "Expected exactly 26 planning weeks"
        )

    first_week = calendar[
        "forecast_week"
    ].min()

    last_week = calendar[
        "forecast_week"
    ].max()

    horizon_end = (
        last_week
        + pd.Timedelta(days=6)
    )

    po = open_pos.copy()

    # Forecast weeks begin on Monday.
    # Convert expected receipt dates to their Monday bucket.
    po["receipt_week"] = (
        po["expected_date"]
        - pd.to_timedelta(
            po["expected_date"].dt.weekday,
            unit="D"
        )
    )

    # Anything already expected before the first forecast
    # week is treated as a Week-1 receipt because it is
    # still open in the snapshot.
    po.loc[
        po["receipt_week"] < first_week,
        "receipt_week"
    ] = first_week

    po[
        "within_planning_horizon"
    ] = (
        po["expected_date"]
        <= horizon_end
    )

    po[
        "beyond_planning_horizon"
    ] = (
        po["expected_date"]
        > horizon_end
    )

    return po, calendar


# =========================================================
# AGGREGATE EXISTING PO RECEIPTS
# =========================================================

def aggregate_existing_receipts(po):

    print("Aggregating existing PO receipts by SKU/week...")

    within = po[
        po["within_planning_horizon"]
    ].copy()

    if within.empty:

        return pd.DataFrame(
            columns=[
                "sku",
                "forecast_week",
                "existing_po_receipt_qty",
                "existing_po_line_count",
                "existing_po_at_risk_qty",
            ]
        )

    within[
        "po_risk_flag"
    ] = (
        within["overdue_flag"]
        | within["partial_receipt_flag"]
        | within["expected_after_promise_flag"]
    )

    within[
        "at_risk_qty"
    ] = np.where(
        within["po_risk_flag"],
        within["open_qty"],
        0.0
    )

    receipts = (
        within.groupby(
            [
                "sku",
                "receipt_week",
            ],
            as_index=False
        )
        .agg(
            existing_po_receipt_qty=(
                "open_qty",
                "sum"
            ),
            existing_po_line_count=(
                "po_number",
                "count"
            ),
            existing_po_at_risk_qty=(
                "at_risk_qty",
                "sum"
            ),
        )
        .rename(
            columns={
                "receipt_week":
                "forecast_week"
            }
        )
    )

    return receipts


# =========================================================
# BUILD WEEKLY PLANNING BASE
# =========================================================

def build_weekly_base(
    forecasts,
    policy,
    receipts
):

    print("Building 26-week planning grid...")

    policy_columns = [
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
        "on_hand",
        "allocated",
        "backordered",
        "net_available",
        "supplier_risk_score",
        "supplier_risk_class",
        "snapshot_date",
    ]

    base = forecasts.merge(
        policy[
            policy_columns
        ],
        on="sku",
        how="left",
        validate="many_to_one"
    )

    base = base.merge(
        receipts,
        on=[
            "sku",
            "forecast_week",
        ],
        how="left",
        validate="one_to_one"
    )

    fill_zero = [
        "existing_po_receipt_qty",
        "existing_po_line_count",
        "existing_po_at_risk_qty",
    ]

    for column in fill_zero:
        base[column] = pd.to_numeric(
            base[column],
            errors="coerce"
        ).fillna(0.0)

    base[
        "starting_net_inventory"
    ] = (
        pd.to_numeric(
            base["on_hand"],
            errors="coerce"
        ).fillna(0.0)
        - pd.to_numeric(
            base["allocated"],
            errors="coerce"
        ).fillna(0.0)
        - pd.to_numeric(
            base["backordered"],
            errors="coerce"
        ).fillna(0.0)
    )

    return base


# =========================================================
# TIME-PHASED INVENTORY SIMULATION
# =========================================================

def simulate_inventory(base):

    print("Running 26-week inventory simulation...")

    results = []
    planned_orders = []

    for sku, group in base.groupby(
        "sku",
        sort=False
    ):

        group = (
            group.sort_values(
                "forecast_horizon_week"
            )
            .copy()
        )

        first = group.iloc[0]

        projected_inventory = float(
            first["starting_net_inventory"]
        )

        safety_stock = float(
            first["safety_stock"]
        )

        moq = float(
            first["moq"]
        )

        case_pack = float(
            first["case_pack"]
        )

        lead_time_weeks = float(
            first["planning_lead_time_weeks"]
        )

        lead_time_days = (
            lead_time_weeks
            * DAYS_PER_WEEK
        )

        unit_cost = float(
            first["unit_cost"]
        )

        for _, row in group.iterrows():

            week = row["forecast_week"]

            beginning_inventory = (
                projected_inventory
            )

            existing_receipt = float(
                row[
                    "existing_po_receipt_qty"
                ]
            )

            forecast_demand = float(
                row["forecast_demand"]
            )

            # Position before creating a new planned receipt.
            projected_before_planned = (
                beginning_inventory
                + existing_receipt
                - forecast_demand
            )

            safety_stock_gap = max(
                safety_stock
                - projected_before_planned,
                0.0
            )

            planned_receipt_qty = 0

            # Create a planned receipt only when projected
            # ending inventory would fall below safety stock.
            if safety_stock_gap > 0:

                planned_receipt_qty = (
                    apply_order_constraints(
                        safety_stock_gap,
                        moq,
                        case_pack
                    )
                )

            projected_inventory = (
                projected_before_planned
                + planned_receipt_qty
            )

            projected_without_planned_order = (
                projected_before_planned
            )

            projected_shortage_without_plan = max(
                -projected_without_planned_order,
                0.0
            )

            safety_stock_breach_without_plan = (
                projected_without_planned_order
                < safety_stock
            )

            stockout_without_plan = (
                projected_without_planned_order
                < 0
            )

            if forecast_demand > 0:

                ending_weeks_of_supply = (
                    projected_inventory
                    / forecast_demand
                )

            else:

                ending_weeks_of_supply = np.inf

            planned_release_date = pd.NaT
            release_status = "NO NEW ORDER"

            if planned_receipt_qty > 0:

                planned_release_date = (
                    week
                    - pd.to_timedelta(
                        lead_time_days,
                        unit="D"
                    )
                )

                days_until_release = (
                    planned_release_date
                    - row["snapshot_date"]
                ).total_seconds() / 86400.0

                if days_until_release <= 0:
                    release_status = "ORDER NOW"

                elif days_until_release <= 7:
                    release_status = "ORDER THIS WEEK"

                elif days_until_release <= 28:
                    release_status = "ORDER SOON"

                else:
                    release_status = "PLANNED ORDER"

                planned_orders.append(
                    {
                        "sku": sku,
                        "product_name":
                            row["product_name"],
                        "category":
                            row["category"],
                        "supplier_id":
                            row["supplier_id"],
                        "supplier_risk_class":
                            row["supplier_risk_class"],
                        "final_demand_class":
                            row["final_demand_class"],
                        "production_model":
                            row["production_model"],
                        "forecast_readiness":
                            row["forecast_readiness"],
                        "forecast_difficulty":
                            row["forecast_difficulty"],
                        "required_receipt_week":
                            week,
                        "planned_release_date":
                            planned_release_date,
                        "planning_lead_time_weeks":
                            lead_time_weeks,
                        "raw_net_requirement":
                            safety_stock_gap,
                        "planned_order_qty":
                            planned_receipt_qty,
                        "unit_cost":
                            unit_cost,
                        "planned_order_value":
                            planned_receipt_qty
                            * unit_cost,
                        "release_status":
                            release_status,
                        "forecast_horizon_week":
                            row[
                                "forecast_horizon_week"
                            ],
                    }
                )

            results.append(
                {
                    "sku":
                        sku,
                    "product_name":
                        row["product_name"],
                    "category":
                        row["category"],
                    "supplier_id":
                        row["supplier_id"],

                    "forecast_week":
                        week,
                    "forecast_horizon_week":
                        row[
                            "forecast_horizon_week"
                        ],

                    "production_model":
                        row["production_model"],
                    "model_source":
                        row["model_source"],
                    "final_demand_class":
                        row["final_demand_class"],
                    "forecast_readiness":
                        row["forecast_readiness"],
                    "forecast_difficulty":
                        row["forecast_difficulty"],
                    "classification_confidence":
                        row[
                            "classification_confidence"
                        ],
                    "selected_wape":
                        row["selected_wape"],

                    "forecast_demand":
                        forecast_demand,

                    "beginning_projected_inventory":
                        beginning_inventory,

                    "existing_po_receipt_qty":
                        existing_receipt,

                    "existing_po_at_risk_qty":
                        row[
                            "existing_po_at_risk_qty"
                        ],

                    "projected_inventory_before_plan":
                        projected_before_planned,

                    "safety_stock":
                        safety_stock,

                    "safety_stock_gap":
                        safety_stock_gap,

                    "safety_stock_breach_without_plan":
                        safety_stock_breach_without_plan,

                    "stockout_without_plan":
                        stockout_without_plan,

                    "projected_shortage_without_plan":
                        projected_shortage_without_plan,

                    "planned_order_receipt_qty":
                        planned_receipt_qty,

                    "projected_ending_inventory":
                        projected_inventory,

                    "ending_weeks_of_supply":
                        ending_weeks_of_supply,

                    "planned_order_release_date":
                        planned_release_date,

                    "release_status":
                        release_status,

                    "unit_cost":
                        unit_cost,

                    "planned_order_value":
                        planned_receipt_qty
                        * unit_cost,

                    "supplier_risk_score":
                        row["supplier_risk_score"],

                    "supplier_risk_class":
                        row["supplier_risk_class"],
                }
            )

    weekly = pd.DataFrame(results)
    orders = pd.DataFrame(planned_orders)

    return weekly, orders


# =========================================================
# BUILD SKU SUMMARY
# =========================================================

def build_sku_summary(
    weekly,
    planned_orders,
    policy,
    open_pos
):

    print("Building SKU-level time-phased summary...")

    weekly_summary = (
        weekly.groupby(
            "sku",
            as_index=False
        )
        .agg(
            forecast_26w_units=(
                "forecast_demand",
                "sum"
            ),
            existing_po_receipts_26w=(
                "existing_po_receipt_qty",
                "sum"
            ),
            at_risk_po_receipts_26w=(
                "existing_po_at_risk_qty",
                "sum"
            ),
            planned_order_units_26w=(
                "planned_order_receipt_qty",
                "sum"
            ),
            planned_order_value_26w=(
                "planned_order_value",
                "sum"
            ),
            minimum_inventory_without_plan=(
                "projected_inventory_before_plan",
                "min"
            ),
            minimum_planned_inventory=(
                "projected_ending_inventory",
                "min"
            ),
            safety_stock_breach_weeks=(
                "safety_stock_breach_without_plan",
                "sum"
            ),
            stockout_weeks_without_plan=(
                "stockout_without_plan",
                "sum"
            ),
        )
    )

    first_breach = (
        weekly[
            weekly[
                "safety_stock_breach_without_plan"
            ]
        ]
        .groupby("sku")[
            "forecast_week"
        ]
        .min()
        .rename(
            "first_safety_stock_breach_week"
        )
        .reset_index()
    )

    first_stockout = (
        weekly[
            weekly[
                "stockout_without_plan"
            ]
        ]
        .groupby("sku")[
            "forecast_week"
        ]
        .min()
        .rename(
            "first_stockout_week_without_plan"
        )
        .reset_index()
    )

    summary = (
        policy[
            [
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
            ]
        ]
        .merge(
            weekly_summary,
            on="sku",
            how="left",
            validate="one_to_one"
        )
        .merge(
            first_breach,
            on="sku",
            how="left",
            validate="one_to_one"
        )
        .merge(
            first_stockout,
            on="sku",
            how="left",
            validate="one_to_one"
        )
    )

    # Total open PO quantity including receipts that fall
    # outside the 26-week horizon.
    total_open = (
        open_pos.groupby(
            "sku",
            as_index=False
        )["open_qty"]
        .sum()
        .rename(
            columns={
                "open_qty":
                "total_open_po_qty"
            }
        )
    )

    beyond = (
        open_pos[
            open_pos[
                "beyond_planning_horizon"
            ]
        ]
        .groupby(
            "sku",
            as_index=False
        )["open_qty"]
        .sum()
        .rename(
            columns={
                "open_qty":
                "open_po_beyond_horizon_qty"
            }
        )
    )

    summary = summary.merge(
        total_open,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    summary = summary.merge(
        beyond,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    summary[
        "total_open_po_qty"
    ] = summary[
        "total_open_po_qty"
    ].fillna(0.0)

    summary[
        "open_po_beyond_horizon_qty"
    ] = summary[
        "open_po_beyond_horizon_qty"
    ].fillna(0.0)

    if planned_orders.empty:

        order_summary = pd.DataFrame(
            {
                "sku": summary["sku"],
                "planned_order_count": 0,
                "first_planned_receipt_week":
                    pd.NaT,
                "first_planned_release_date":
                    pd.NaT,
                "immediate_order_units": 0,
                "immediate_order_value": 0.0,
            }
        )

    else:

        order_summary = (
            planned_orders.groupby(
                "sku",
                as_index=False
            )
            .agg(
                planned_order_count=(
                    "planned_order_qty",
                    "count"
                ),
                first_planned_receipt_week=(
                    "required_receipt_week",
                    "min"
                ),
                first_planned_release_date=(
                    "planned_release_date",
                    "min"
                ),
            )
        )

        immediate = (
            planned_orders[
                planned_orders[
                    "release_status"
                ] == "ORDER NOW"
            ]
            .groupby(
                "sku",
                as_index=False
            )
            .agg(
                immediate_order_units=(
                    "planned_order_qty",
                    "sum"
                ),
                immediate_order_value=(
                    "planned_order_value",
                    "sum"
                ),
            )
        )

        order_summary = (
            order_summary.merge(
                immediate,
                on="sku",
                how="left"
            )
        )

        order_summary[
            "immediate_order_units"
        ] = order_summary[
            "immediate_order_units"
        ].fillna(0.0)

        order_summary[
            "immediate_order_value"
        ] = order_summary[
            "immediate_order_value"
        ].fillna(0.0)

    summary = summary.merge(
        order_summary,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    summary[
        "planned_order_count"
    ] = summary[
        "planned_order_count"
    ].fillna(0).astype(int)

    summary[
        "immediate_order_units"
    ] = summary[
        "immediate_order_units"
    ].fillna(0.0)

    summary[
        "immediate_order_value"
    ] = summary[
        "immediate_order_value"
    ].fillna(0.0)

    return summary


# =========================================================
# PLANNING ACTION
# =========================================================

def add_planning_action(summary):

    print("Assigning final time-phased planning actions...")

    result = summary.copy()

    actions = []

    for _, row in result.iterrows():

        immediate = row[
            "immediate_order_units"
        ]

        planned_units = row[
            "planned_order_units_26w"
        ]

        breach_weeks = row[
            "safety_stock_breach_weeks"
        ]

        stockout_weeks = row[
            "stockout_weeks_without_plan"
        ]

        if immediate > 0:

            if stockout_weeks > 0:
                actions.append(
                    "ORDER NOW - STOCKOUT RISK"
                )
            else:
                actions.append(
                    "ORDER NOW - SAFETY STOCK"
                )

        elif planned_units > 0:

            actions.append(
                "PLANNED ORDER"
            )

        elif breach_weeks > 0:

            actions.append(
                "REVIEW"
            )

        else:

            actions.append(
                "NO ACTION"
            )

    result[
        "time_phased_action"
    ] = actions

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate_outputs(
    weekly,
    summary,
    planned_orders,
    forecasts,
    open_pos
):

    section(
        "TIME-PHASED PLANNING VALIDATION"
    )

    assert len(weekly) == 5200
    check("Exactly 5,200 weekly planning rows")

    assert weekly["sku"].nunique() == 200
    check("Exactly 200 SKUs")

    counts = weekly.groupby("sku").size()

    assert (counts == 26).all()
    check("Every SKU has exactly 26 planning weeks")

    assert (
        weekly["forecast_demand"] >= 0
    ).all()

    check("Forecast demand is non-negative")

    assert (
        weekly[
            "existing_po_receipt_qty"
        ] >= 0
    ).all()

    check("Existing PO receipts are non-negative")

    assert (
        weekly[
            "planned_order_receipt_qty"
        ] >= 0
    ).all()

    check("Planned order receipts are non-negative")

    forecast_total_source = (
        forecasts[
            "forecast_demand"
        ].sum()
    )

    forecast_total_plan = (
        weekly[
            "forecast_demand"
        ].sum()
    )

    assert np.isclose(
        forecast_total_source,
        forecast_total_plan
    )

    check(
        f"26-week forecast reconciles: "
        f"{forecast_total_plan:,.1f} units"
    )

    po_within_horizon = (
        open_pos.loc[
            open_pos[
                "within_planning_horizon"
            ],
            "open_qty"
        ].sum()
    )

    po_receipts_plan = (
        weekly[
            "existing_po_receipt_qty"
        ].sum()
    )

    assert np.isclose(
        po_within_horizon,
        po_receipts_plan
    )

    check(
        f"In-horizon PO receipts reconcile: "
        f"{po_receipts_plan:,.0f} units"
    )

    assert len(summary) == 200
    check("Exactly 200 SKU summary rows")

    # Once a planned receipt is inserted,
    # ending inventory should be at or above safety stock.
    assert (
        weekly[
            "projected_ending_inventory"
        ]
        + 1e-9
        >= weekly[
            "safety_stock"
        ]
    ).all()

    check(
        "Planned receipts protect safety stock"
    )

    positive_orders = weekly[
        weekly[
            "planned_order_receipt_qty"
        ] > 0
    ]

    if not positive_orders.empty:

        moq_lookup = (
            summary.set_index("sku")[
                "moq"
            ]
        )

        case_lookup = (
            summary.set_index("sku")[
                "case_pack"
            ]
        )

        for _, row in positive_orders.iterrows():

            sku = row["sku"]

            qty = int(
                row[
                    "planned_order_receipt_qty"
                ]
            )

            moq = int(
                math.ceil(
                    moq_lookup.loc[sku]
                )
            )

            case_pack = max(
                int(
                    case_lookup.loc[sku]
                ),
                1
            )

            assert qty >= moq
            assert qty % case_pack == 0

    check(
        "All planned receipts satisfy MOQ and case pack"
    )

    if not planned_orders.empty:

        assert (
            planned_orders[
                "planned_order_qty"
            ] > 0
        ).all()

    check(
        "Planned-order table contains only positive orders"
    )

    print(
        "\nAll Phase 7D validation checks passed."
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(
    weekly,
    summary,
    planned_orders
):

    section(
        "SAVING TIME-PHASED PLANNING OUTPUTS"
    )

    weekly_out = weekly.copy()

    for column in [
        "forecast_week",
        "planned_order_release_date",
    ]:

        weekly_out[column] = pd.to_datetime(
            weekly_out[column],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    weekly_out.to_csv(
        WEEKLY_OUTPUT_PATH,
        index=False
    )

    summary_out = summary.copy()

    for column in [
        "first_safety_stock_breach_week",
        "first_stockout_week_without_plan",
        "first_planned_receipt_week",
        "first_planned_release_date",
    ]:

        if column in summary_out.columns:

            summary_out[column] = pd.to_datetime(
                summary_out[column],
                errors="coerce"
            ).dt.strftime("%Y-%m-%d")

    summary_out.to_csv(
        SUMMARY_OUTPUT_PATH,
        index=False
    )

    orders_out = planned_orders.copy()

    if not orders_out.empty:

        for column in [
            "required_receipt_week",
            "planned_release_date",
        ]:

            orders_out[column] = pd.to_datetime(
                orders_out[column],
                errors="coerce"
            ).dt.strftime("%Y-%m-%d")

    orders_out.to_csv(
        PLANNED_ORDERS_PATH,
        index=False
    )

    check(WEEKLY_OUTPUT_PATH.name)
    check(SUMMARY_OUTPUT_PATH.name)
    check(PLANNED_ORDERS_PATH.name)


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    weekly,
    summary,
    planned_orders,
    open_pos
):

    section(
        "26-WEEK TIME-PHASED PLANNING SUMMARY"
    )

    print(
        f"\nSKUs evaluated: "
        f"{summary['sku'].nunique():,}"
    )

    print(
        f"Planning weeks: "
        f"{weekly['forecast_horizon_week'].nunique():,}"
    )

    print(
        f"Total 26-week forecast demand: "
        f"{weekly['forecast_demand'].sum():,.1f} units"
    )

    print(
        f"\nExisting open PO units: "
        f"{open_pos['open_qty'].sum():,.0f}"
    )

    print(
        f"Open PO receipts inside horizon: "
        f"{weekly['existing_po_receipt_qty'].sum():,.0f}"
    )

    print(
        f"Open PO units beyond horizon: "
        f"{open_pos.loc[open_pos['beyond_planning_horizon'], 'open_qty'].sum():,.0f}"
    )

    print(
        f"\nTime-phased planned order units: "
        f"{weekly['planned_order_receipt_qty'].sum():,.0f}"
    )

    print(
        f"Time-phased planned order value: "
        f"${weekly['planned_order_value'].sum():,.2f}"
    )

    print(
        f"\nImmediate planned order units: "
        f"{summary['immediate_order_units'].sum():,.0f}"
    )

    print(
        f"Immediate planned order value: "
        f"${summary['immediate_order_value'].sum():,.2f}"
    )

    print(
        f"\nSKUs with at least one "
        f"safety-stock breach: "
        f"{(summary['safety_stock_breach_weeks'] > 0).sum():,}"
    )

    print(
        f"SKUs with at least one projected "
        f"stockout without new supply: "
        f"{(summary['stockout_weeks_without_plan'] > 0).sum():,}"
    )

    print(
        f"SKUs requiring planned supply: "
        f"{(summary['planned_order_units_26w'] > 0).sum():,}"
    )

    print(
        "\nTime-phased actions:"
    )

    print(
        summary[
            "time_phased_action"
        ].value_counts()
    )

    if not planned_orders.empty:

        section(
            "PLANNED ORDER RELEASE STATUS",
            width=125
        )

        release_summary = (
            planned_orders.groupby(
                "release_status"
            )
            .agg(
                order_count=(
                    "sku",
                    "count"
                ),
                sku_count=(
                    "sku",
                    "nunique"
                ),
                planned_units=(
                    "planned_order_qty",
                    "sum"
                ),
                planned_value=(
                    "planned_order_value",
                    "sum"
                ),
            )
            .round(2)
            .sort_values(
                "planned_value",
                ascending=False
            )
        )

        print(
            release_summary.to_string()
        )

    section(
        "TOP IMMEDIATE TIME-PHASED REQUIREMENTS",
        width=175
    )

    top_columns = [
        "sku",
        "product_name",
        "supplier_id",
        "supplier_risk_class",
        "net_available",
        "safety_stock",
        "reorder_point",
        "forecast_26w_units",
        "existing_po_receipts_26w",
        "first_safety_stock_breach_week",
        "first_stockout_week_without_plan",
        "immediate_order_units",
        "immediate_order_value",
        "planned_order_units_26w",
        "planned_order_value_26w",
        "time_phased_action",
    ]

    top = (
        summary[
            top_columns
        ]
        .sort_values(
            [
                "immediate_order_value",
                "planned_order_value_26w",
            ],
            ascending=False
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

    (
        forecasts,
        policy,
        open_pos,
        planning
    ) = load_data()

    validate_inputs(
        forecasts,
        policy,
        open_pos,
        planning
    )

    (
        forecasts,
        policy,
        open_pos
    ) = prepare_data(
        forecasts,
        policy,
        open_pos,
        planning
    )

    (
        open_pos,
        calendar
    ) = assign_po_receipt_weeks(
        forecasts,
        open_pos
    )

    receipts = aggregate_existing_receipts(
        open_pos
    )

    base = build_weekly_base(
        forecasts,
        policy,
        receipts
    )

    (
        weekly,
        planned_orders
    ) = simulate_inventory(
        base
    )

    summary = build_sku_summary(
        weekly,
        planned_orders,
        policy,
        open_pos
    )

    summary = add_planning_action(
        summary
    )

    validate_outputs(
        weekly,
        summary,
        planned_orders,
        forecasts,
        open_pos
    )

    save_outputs(
        weekly,
        summary,
        planned_orders
    )

    print_summary(
        weekly,
        summary,
        planned_orders,
        open_pos
    )

    section(
        "PHASE 7D COMPLETE"
    )


if __name__ == "__main__":
    main()