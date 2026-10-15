from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PLANNING_MASTER_PATH = (
    PROCESSED_DIR / "planning_master.csv"
)

SUPPLIER_RISK_PATH = (
    PROCESSED_DIR / "supplier_risk.csv"
)

FORECAST_SUMMARY_PATH = (
    PROCESSED_DIR / "production_forecast_summary.csv"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "inventory_policy.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

DAYS_PER_WEEK = 7.0

MIN_LEAD_TIME_WEEKS = 0.25

MIN_DEMAND_STD = 0.0

MAX_SAFETY_STOCK_WEEKS = 26.0

INTERMITTENT_MIN_PROTECTION_WEEKS = 2.0

NEW_PRODUCT_MIN_PROTECTION_WEEKS = 2.0

CRITICAL_SUPPLIER_MIN_PROTECTION_WEEKS = 1.0


# =========================================================
# DISPLAY HELPERS
# =========================================================

def section(title, width=90):

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
        "PROCUREMENT AI - INVENTORY POLICY ENGINE"
    )

    print("\nLoading Phase 7A inputs...")

    planning = pd.read_csv(
        PLANNING_MASTER_PATH
    )

    supplier_risk = pd.read_csv(
        SUPPLIER_RISK_PATH
    )

    forecasts = pd.read_csv(
        FORECAST_SUMMARY_PATH
    )

    check(
        f"planning_master            "
        f"{len(planning):>8,} rows"
    )

    check(
        f"supplier_risk              "
        f"{len(supplier_risk):>8,} rows"
    )

    check(
        f"production_forecast_summary"
        f"{len(forecasts):>8,} rows"
    )

    return planning, supplier_risk, forecasts


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_inputs(
    planning,
    supplier_risk,
    forecasts
):

    print("Validating inventory-policy inputs...")

    required_planning = {
        "sku",
        "product_name",
        "category",
        "supplier_id",
        "unit_cost",
        "moq",
        "case_pack",
        "service_level_target",
        "on_hand",
        "allocated",
        "available",
        "backordered",
        "net_available",
        "inventory_position",
        "total_open_po_qty",
        "total_overdue_po_qty",
        "13w_demand_std",
        "52w_demand_std",
        "13w_avg_demand",
        "52w_avg_demand",
        "13w_cv",
        "scenario_label",
        "has_open_po",
        "has_overdue_po",
        "has_partial_po",
    }

    required_supplier = {
        "supplier_id",
        "effective_planning_lead_time_weeks",
        "std_full_lead_time_days",
        "supplier_risk_score",
        "supplier_risk_class",
        "full_po_on_time_rate",
        "partial_shipment_rate",
        "planning_lead_time_buffer_weeks",
    }

    required_forecast = {
        "sku",
        "final_demand_class",
        "classification_confidence",
        "forecast_readiness",
        "forecast_difficulty",
        "production_model",
        "model_source",
        "selected_wape",
        "selected_bias",
        "forecast_4w_units",
        "forecast_13w_units",
        "forecast_26w_units",
        "avg_weekly_forecast",
    }

    missing_planning = (
        required_planning
        - set(planning.columns)
    )

    missing_supplier = (
        required_supplier
        - set(supplier_risk.columns)
    )

    missing_forecast = (
        required_forecast
        - set(forecasts.columns)
    )

    if missing_planning:

        raise ValueError(
            "planning_master.csv missing columns: "
            + ", ".join(
                sorted(missing_planning)
            )
        )

    if missing_supplier:

        raise ValueError(
            "supplier_risk.csv missing columns: "
            + ", ".join(
                sorted(missing_supplier)
            )
        )

    if missing_forecast:

        raise ValueError(
            "production_forecast_summary.csv "
            "missing columns: "
            + ", ".join(
                sorted(missing_forecast)
            )
        )

    if planning["sku"].duplicated().any():

        raise ValueError(
            "planning_master contains duplicate SKUs"
        )

    if forecasts["sku"].duplicated().any():

        raise ValueError(
            "production forecast summary "
            "contains duplicate SKUs"
        )

    if supplier_risk[
        "supplier_id"
    ].duplicated().any():

        raise ValueError(
            "supplier_risk contains "
            "duplicate suppliers"
        )

    if len(planning) != 200:

        raise ValueError(
            f"Expected 200 planning SKUs, "
            f"found {len(planning)}"
        )

    check("Required inventory-policy columns present")
    check("Planning master has one row per SKU")
    check("Forecast summary has one row per SKU")
    check("Supplier risk has one row per supplier")


# =========================================================
# BUILD POLICY BASE
# =========================================================

def build_policy_base(
    planning,
    supplier_risk,
    forecasts
):

    print("Building inventory-policy base...")

    supplier_columns = [
        "supplier_id",
        "effective_planning_lead_time_weeks",
        "std_full_lead_time_days",
        "supplier_risk_score",
        "supplier_risk_class",
        "full_po_on_time_rate",
        "partial_shipment_rate",
        "planning_lead_time_buffer_weeks",
    ]

    forecast_columns = [
        "sku",
        "final_demand_class",
        "classification_confidence",
        "forecast_readiness",
        "forecast_difficulty",
        "production_model",
        "model_source",
        "selected_wape",
        "selected_bias",
        "forecast_4w_units",
        "forecast_13w_units",
        "forecast_26w_units",
        "avg_weekly_forecast",
    ]

    result = planning.merge(
        supplier_risk[
            supplier_columns
        ],
        on="supplier_id",
        how="left",
        validate="many_to_one"
    )

    result = result.merge(
        forecasts[
            forecast_columns
        ],
        on="sku",
        how="left",
        validate="one_to_one"
    )

    if result[
        "effective_planning_lead_time_weeks"
    ].isna().any():

        raise ValueError(
            "Missing effective planning lead time "
            "after supplier merge"
        )

    if result[
        "avg_weekly_forecast"
    ].isna().any():

        raise ValueError(
            "Missing production forecast "
            "after forecast merge"
        )

    check(
        f"Inventory-policy base: "
        f"{len(result):,} SKUs"
    )

    return result


# =========================================================
# SERVICE LEVEL
# =========================================================

def calculate_service_level_features(df):

    print("Calculating service-level protection...")

    result = df.copy()

    result[
        "service_level_target"
    ] = pd.to_numeric(
        result["service_level_target"],
        errors="coerce"
    )

    if result[
        "service_level_target"
    ].isna().any():

        raise ValueError(
            "Missing service-level targets"
        )

    if not result[
        "service_level_target"
    ].between(
        0.50,
        0.9999
    ).all():

        raise ValueError(
            "Service-level targets must be "
            "stored as decimal probabilities"
        )

    result[
        "service_level_z"
    ] = result[
        "service_level_target"
    ].apply(
        lambda x: NormalDist().inv_cdf(
            float(x)
        )
    )

    return result


# =========================================================
# FORECAST DEMAND RATE
# =========================================================

def calculate_demand_rate(df):

    print("Calculating planning demand rates...")

    result = df.copy()

    result[
        "forecast_weekly_demand"
    ] = pd.to_numeric(
        result["avg_weekly_forecast"],
        errors="coerce"
    ).fillna(0.0)

    result[
        "recent_13w_weekly_demand"
    ] = pd.to_numeric(
        result["13w_avg_demand"],
        errors="coerce"
    ).fillna(0.0)

    result[
        "long_term_weekly_demand"
    ] = pd.to_numeric(
        result["52w_avg_demand"],
        errors="coerce"
    ).fillna(0.0)

    # Production forecast is the primary demand signal.
    result[
        "planning_weekly_demand"
    ] = result[
        "forecast_weekly_demand"
    ]

    # Guard against pathological zero forecasts for
    # intermittent demand where recent actual demand exists.
    intermittent_mask = (
        result["final_demand_class"]
        == "intermittent"
    )

    intermittent_floor = np.maximum(
        result["recent_13w_weekly_demand"]
        * 0.50,
        result["long_term_weekly_demand"]
        * 0.35
    )

    result.loc[
        intermittent_mask,
        "planning_weekly_demand"
    ] = np.maximum(
        result.loc[
            intermittent_mask,
            "planning_weekly_demand"
        ],
        intermittent_floor.loc[
            intermittent_mask
        ]
    )

    result[
        "planning_weekly_demand"
    ] = result[
        "planning_weekly_demand"
    ].clip(
        lower=0.0
    )

    return result


# =========================================================
# DEMAND UNCERTAINTY
# =========================================================

def calculate_demand_uncertainty(df):

    print("Calculating demand uncertainty...")

    result = df.copy()

    recent_std = pd.to_numeric(
        result["13w_demand_std"],
        errors="coerce"
    )

    long_std = pd.to_numeric(
        result["52w_demand_std"],
        errors="coerce"
    )

    # Prefer recent variability when enough signal exists,
    # but avoid understating risk when long-term
    # variability is materially higher.
    result[
        "planning_weekly_demand_std"
    ] = np.maximum(
        recent_std.fillna(0.0),
        long_std.fillna(0.0) * 0.75
    )

    result[
        "planning_weekly_demand_std"
    ] = result[
        "planning_weekly_demand_std"
    ].clip(
        lower=MIN_DEMAND_STD
    )

    return result


# =========================================================
# LEAD-TIME UNCERTAINTY
# =========================================================

def calculate_lead_time_features(df):

    print("Calculating lead-time exposure...")

    result = df.copy()

    result[
        "planning_lead_time_weeks"
    ] = pd.to_numeric(
        result[
            "effective_planning_lead_time_weeks"
        ],
        errors="coerce"
    ).clip(
        lower=MIN_LEAD_TIME_WEEKS
    )

    result[
        "lead_time_std_weeks"
    ] = (
        pd.to_numeric(
            result[
                "std_full_lead_time_days"
            ],
            errors="coerce"
        )
        .fillna(0.0)
        / DAYS_PER_WEEK
    )

    result[
        "lead_time_std_weeks"
    ] = result[
        "lead_time_std_weeks"
    ].clip(
        lower=0.0
    )

    result[
        "forecast_lead_time_demand"
    ] = (
        result[
            "planning_weekly_demand"
        ]
        * result[
            "planning_lead_time_weeks"
        ]
    )

    return result


# =========================================================
# SAFETY STOCK
# =========================================================

def calculate_safety_stock(df):

    print("Calculating dynamic safety stock...")

    result = df.copy()

    demand_variance_component = (
        result[
            "planning_lead_time_weeks"
        ]
        * np.square(
            result[
                "planning_weekly_demand_std"
            ]
        )
    )

    lead_time_variance_component = (
        np.square(
            result[
                "planning_weekly_demand"
            ]
        )
        * np.square(
            result[
                "lead_time_std_weeks"
            ]
        )
    )

    result[
        "lead_time_demand_variance"
    ] = (
        demand_variance_component
        + lead_time_variance_component
    )

    result[
        "statistical_safety_stock"
    ] = (
        result[
            "service_level_z"
        ]
        * np.sqrt(
            result[
                "lead_time_demand_variance"
            ]
        )
    )

    result[
        "statistical_safety_stock"
    ] = result[
        "statistical_safety_stock"
    ].clip(
        lower=0.0
    )

    # -----------------------------------------------------
    # Specialized policy floors
    # -----------------------------------------------------

    result[
        "policy_floor_safety_stock"
    ] = 0.0

    intermittent_mask = (
        result[
            "final_demand_class"
        ]
        == "intermittent"
    )

    result.loc[
        intermittent_mask,
        "policy_floor_safety_stock"
    ] = (
        result.loc[
            intermittent_mask,
            "planning_weekly_demand"
        ]
        * INTERMITTENT_MIN_PROTECTION_WEEKS
    )

    new_product_mask = (
        result[
            "final_demand_class"
        ]
        == "new_product"
    )

    result.loc[
        new_product_mask,
        "policy_floor_safety_stock"
    ] = np.maximum(
        result.loc[
            new_product_mask,
            "policy_floor_safety_stock"
        ],
        result.loc[
            new_product_mask,
            "planning_weekly_demand"
        ]
        * NEW_PRODUCT_MIN_PROTECTION_WEEKS
    )

    critical_supplier_mask = (
        result[
            "supplier_risk_class"
        ]
        == "critical"
    )

    critical_supplier_floor = (
        result[
            "planning_weekly_demand"
        ]
        * CRITICAL_SUPPLIER_MIN_PROTECTION_WEEKS
    )

    result.loc[
        critical_supplier_mask,
        "policy_floor_safety_stock"
    ] = np.maximum(
        result.loc[
            critical_supplier_mask,
            "policy_floor_safety_stock"
        ],
        critical_supplier_floor.loc[
            critical_supplier_mask
        ]
    )

    result[
        "raw_safety_stock"
    ] = np.maximum(
        result[
            "statistical_safety_stock"
        ],
        result[
            "policy_floor_safety_stock"
        ]
    )

    # -----------------------------------------------------
    # Cap extreme statistical outcomes
    # -----------------------------------------------------

    result[
        "safety_stock_cap"
    ] = (
        result[
            "planning_weekly_demand"
        ]
        * MAX_SAFETY_STOCK_WEEKS
    )

    positive_demand_mask = (
        result[
            "planning_weekly_demand"
        ]
        > 0
    )

    result[
        "safety_stock"
    ] = result[
        "raw_safety_stock"
    ]

    result.loc[
        positive_demand_mask,
        "safety_stock"
    ] = np.minimum(
        result.loc[
            positive_demand_mask,
            "raw_safety_stock"
        ],
        result.loc[
            positive_demand_mask,
            "safety_stock_cap"
        ]
    )

    result[
        "safety_stock"
    ] = np.ceil(
        result[
            "safety_stock"
        ]
    ).astype(int)

    result[
        "safety_stock_weeks"
    ] = np.where(
        result[
            "planning_weekly_demand"
        ] > 0,
        result[
            "safety_stock"
        ]
        / result[
            "planning_weekly_demand"
        ],
        0.0
    )

    return result


# =========================================================
# REORDER POINT
# =========================================================

def calculate_reorder_point(df):

    print("Calculating reorder points...")

    result = df.copy()

    result[
        "reorder_point"
    ] = np.ceil(
        result[
            "forecast_lead_time_demand"
        ]
        + result[
            "safety_stock"
        ]
    ).astype(int)

    return result


# =========================================================
# INVENTORY POSITION
# =========================================================

def calculate_inventory_position(df):

    print("Calculating policy inventory position...")

    result = df.copy()

    numeric_columns = [
        "on_hand",
        "allocated",
        "backordered",
        "total_open_po_qty",
    ]

    for column in numeric_columns:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        ).fillna(0.0)

    # Explicitly recalculate rather than relying
    # blindly on the earlier planning-master field.
    result[
        "policy_inventory_position"
    ] = (
        result["on_hand"]
        - result["allocated"]
        - result["backordered"]
        + result["total_open_po_qty"]
    )

    result[
        "inventory_position_gap"
    ] = (
        result[
            "policy_inventory_position"
        ]
        - result[
            "reorder_point"
        ]
    )

    result[
        "below_reorder_point"
    ] = (
        result[
            "policy_inventory_position"
        ]
        < result[
            "reorder_point"
        ]
    )

    result[
        "gross_replenishment_need"
    ] = np.maximum(
        result[
            "reorder_point"
        ]
        - result[
            "policy_inventory_position"
        ],
        0.0
    )

    result[
        "gross_replenishment_need"
    ] = np.ceil(
        result[
            "gross_replenishment_need"
        ]
    ).astype(int)

    return result


# =========================================================
# RISK CLASSIFICATION
# =========================================================

def classify_inventory_policy(df):

    print("Classifying inventory policy status...")

    result = df.copy()

    conditions = [
        (
            result[
                "policy_inventory_position"
            ] <= 0
        ),
        (
            result[
                "policy_inventory_position"
            ]
            < result[
                "forecast_lead_time_demand"
            ]
        ),
        (
            result[
                "policy_inventory_position"
            ]
            < result[
                "reorder_point"
            ]
        ),
    ]

    choices = [
        "critical",
        "shortage_risk",
        "reorder",
    ]

    result[
        "inventory_policy_status"
    ] = np.select(
        conditions,
        choices,
        default="protected"
    )

    result[
        "policy_action"
    ] = np.select(
        [
            result[
                "inventory_policy_status"
            ]
            == "critical",
            result[
                "inventory_policy_status"
            ]
            == "shortage_risk",
            result[
                "inventory_policy_status"
            ]
            == "reorder",
        ],
        [
            "urgent replenishment review",
            "replenishment required",
            "reorder recommended",
        ],
        default="no immediate order signal"
    )

    return result


# =========================================================
# POLICY EXPLANATION
# =========================================================

def build_policy_reason(row):

    reasons = []

    if row[
        "policy_inventory_position"
    ] <= 0:

        reasons.append(
            "inventory position is depleted"
        )

    elif row[
        "policy_inventory_position"
    ] < row[
        "forecast_lead_time_demand"
    ]:

        reasons.append(
            "inventory position does not cover "
            "expected lead-time demand"
        )

    elif row[
        "policy_inventory_position"
    ] < row[
        "reorder_point"
    ]:

        reasons.append(
            "inventory position is below "
            "the reorder point"
        )

    if (
        row["supplier_risk_class"]
        == "critical"
    ):

        reasons.append(
            "critical supplier risk"
        )

    if (
        row["final_demand_class"]
        == "intermittent"
    ):

        reasons.append(
            "intermittent-demand protection applied"
        )

    if (
        row["final_demand_class"]
        == "new_product"
    ):

        reasons.append(
            "new-product protection applied"
        )

    if bool(
        row["has_overdue_po"]
    ):

        reasons.append(
            "open supply includes overdue PO"
        )

    if (
        row["model_source"]
        == "fallback"
    ):

        reasons.append(
            "forecast uses fallback model"
        )

    if not reasons:

        reasons.append(
            "inventory position covers "
            "lead-time demand and safety stock"
        )

    return "; ".join(reasons)


def add_policy_explanations(df):

    print("Building explainable inventory policy...")

    result = df.copy()

    result[
        "inventory_policy_reason"
    ] = result.apply(
        build_policy_reason,
        axis=1
    )

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate_output(df):

    section(
        "INVENTORY POLICY VALIDATION"
    )

    assert len(df) == 200

    check(
        "Exactly 200 SKU policy rows"
    )

    assert (
        df["sku"].nunique()
        == 200
    )

    check(
        "One inventory policy row per SKU"
    )

    assert (
        df["planning_weekly_demand"]
        >= 0
    ).all()

    check(
        "Planning demand is non-negative"
    )

    assert (
        df["planning_lead_time_weeks"]
        > 0
    ).all()

    check(
        "Planning lead times are positive"
    )

    assert (
        df["lead_time_std_weeks"]
        >= 0
    ).all()

    check(
        "Lead-time variability is non-negative"
    )

    assert (
        df["safety_stock"]
        >= 0
    ).all()

    check(
        "Safety stock is non-negative"
    )

    assert (
        df["reorder_point"]
        >= df["safety_stock"]
    ).all()

    check(
        "Reorder point includes safety stock"
    )

    assert (
        df["gross_replenishment_need"]
        >= 0
    ).all()

    check(
        "Gross replenishment need is non-negative"
    )

    expected_position = (
        df["on_hand"]
        - df["allocated"]
        - df["backordered"]
        + df["total_open_po_qty"]
    )

    assert np.allclose(
        df[
            "policy_inventory_position"
        ],
        expected_position
    )

    check(
        "Inventory position calculation reconciles"
    )

    expected_need = np.ceil(
        np.maximum(
            df["reorder_point"]
            - df[
                "policy_inventory_position"
            ],
            0
        )
    )

    assert np.allclose(
        df[
            "gross_replenishment_need"
        ],
        expected_need
    )

    check(
        "Gross replenishment need reconciles"
    )

    valid_statuses = {
        "critical",
        "shortage_risk",
        "reorder",
        "protected",
    }

    assert set(
        df[
            "inventory_policy_status"
        ].unique()
    ).issubset(
        valid_statuses
    )

    check(
        "Inventory policy statuses valid"
    )

    assert df[
        "inventory_policy_reason"
    ].notna().all()

    check(
        "All SKUs have explainable policy reasons"
    )

    print(
        "\nAll Phase 7A validation checks passed."
    )


# =========================================================
# OUTPUT COLUMN SELECTION
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
        "classification_confidence",
        "forecast_readiness",
        "forecast_difficulty",
        "production_model",
        "model_source",
        "selected_wape",

        "service_level_target",
        "service_level_z",

        "forecast_weekly_demand",
        "recent_13w_weekly_demand",
        "long_term_weekly_demand",
        "planning_weekly_demand",
        "planning_weekly_demand_std",

        "effective_planning_lead_time_weeks",
        "planning_lead_time_weeks",
        "lead_time_std_weeks",
        "forecast_lead_time_demand",

        "supplier_risk_score",
        "supplier_risk_class",
        "full_po_on_time_rate",
        "partial_shipment_rate",

        "statistical_safety_stock",
        "policy_floor_safety_stock",
        "raw_safety_stock",
        "safety_stock",
        "safety_stock_weeks",

        "reorder_point",

        "on_hand",
        "allocated",
        "backordered",
        "net_available",
        "total_open_po_qty",
        "total_overdue_po_qty",
        "policy_inventory_position",

        "inventory_position_gap",
        "below_reorder_point",
        "gross_replenishment_need",

        "scenario_label",
        "has_open_po",
        "has_overdue_po",
        "has_partial_po",

        "inventory_policy_status",
        "policy_action",
        "inventory_policy_reason",
    ]

    return df[columns].copy()


# =========================================================
# SAVE OUTPUT
# =========================================================

def save_output(df):

    section(
        "SAVING INVENTORY POLICY"
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    check(
        OUTPUT_PATH.name
    )


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(df):

    section(
        "INVENTORY POLICY SUMMARY"
    )

    print(
        f"\nSKUs: {len(df):,}"
    )

    print(
        f"Total safety stock units: "
        f"{df['safety_stock'].sum():,.0f}"
    )

    print(
        f"Total gross replenishment need: "
        f"{df['gross_replenishment_need'].sum():,.0f}"
    )

    gross_value = (
        df[
            "gross_replenishment_need"
        ]
        * df[
            "unit_cost"
        ]
    ).sum()

    print(
        f"Gross replenishment value: "
        f"${gross_value:,.2f}"
    )

    print(
        "\nInventory policy status:"
    )

    print(
        df[
            "inventory_policy_status"
        ].value_counts()
    )

    print(
        "\nSupplier risk exposure:"
    )

    print(
        df[
            "supplier_risk_class"
        ].value_counts()
    )

    section(
        "POLICY BY DEMAND CLASS",
        width=115
    )

    demand_summary = (
        df.groupby(
            "final_demand_class"
        )
        .agg(
            sku_count=(
                "sku",
                "count"
            ),
            avg_weekly_demand=(
                "planning_weekly_demand",
                "mean"
            ),
            avg_lead_time_weeks=(
                "planning_lead_time_weeks",
                "mean"
            ),
            avg_safety_stock=(
                "safety_stock",
                "mean"
            ),
            avg_safety_stock_weeks=(
                "safety_stock_weeks",
                "mean"
            ),
            total_replenishment_need=(
                "gross_replenishment_need",
                "sum"
            ),
        )
        .round(2)
    )

    print(
        demand_summary.to_string()
    )

    section(
        "HIGHEST REPLENISHMENT NEEDS",
        width=150
    )

    example_columns = [
        "sku",
        "product_name",
        "final_demand_class",
        "supplier_id",
        "supplier_risk_class",
        "planning_weekly_demand",
        "planning_lead_time_weeks",
        "safety_stock",
        "reorder_point",
        "policy_inventory_position",
        "gross_replenishment_need",
        "inventory_policy_status",
    ]

    examples = (
        df[
            example_columns
        ]
        .sort_values(
            "gross_replenishment_need",
            ascending=False
        )
        .head(15)
    )

    print(
        examples.to_string(
            index=False
        )
    )


# =========================================================
# MAIN
# =========================================================

def main():

    planning, supplier_risk, forecasts = (
        load_data()
    )

    validate_inputs(
        planning,
        supplier_risk,
        forecasts
    )

    policy = build_policy_base(
        planning,
        supplier_risk,
        forecasts
    )

    policy = calculate_service_level_features(
        policy
    )

    policy = calculate_demand_rate(
        policy
    )

    policy = calculate_demand_uncertainty(
        policy
    )

    policy = calculate_lead_time_features(
        policy
    )

    policy = calculate_safety_stock(
        policy
    )

    policy = calculate_reorder_point(
        policy
    )

    policy = calculate_inventory_position(
        policy
    )

    policy = classify_inventory_policy(
        policy
    )

    policy = add_policy_explanations(
        policy
    )

    validate_output(
        policy
    )

    output = select_output_columns(
        policy
    )

    save_output(
        output
    )

    print_summary(
        output
    )

    section(
        "PHASE 7A COMPLETE"
    )


if __name__ == "__main__":
    main()