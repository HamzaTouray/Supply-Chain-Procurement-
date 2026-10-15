from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

FORECAST_HORIZONS = [
    4,
    8,
    13,
]

MIN_HISTORY_WEEKS = 8


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

SALES_FILE = (
    PROCESSED_DATA_DIR
    / "sales_clean.csv"
)

FORECAST_READINESS_FILE = (
    PROCESSED_DATA_DIR
    / "forecast_readiness.csv"
)

DEMAND_BEHAVIOR_FILE = (
    PROCESSED_DATA_DIR
    / "demand_behavior.csv"
)

FORECASTING_DATASET_FILE = (
    PROCESSED_DATA_DIR
    / "forecasting_dataset.csv"
)

MODEL_REGISTRY_FILE = (
    PROCESSED_DATA_DIR
    / "forecast_model_registry.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\nLoading Phase 6A inputs...")

    required_files = {
        "sales": SALES_FILE,
        "forecast_readiness":
            FORECAST_READINESS_FILE,
        "demand_behavior":
            DEMAND_BEHAVIOR_FILE,
    }

    data = {}

    for name, path in required_files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

        data[name] = pd.read_csv(path)

        print(
            f"✓ {name:<22}"
            f"{len(data[name]):>7,} rows"
        )

    return data


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_input_schema(data):

    print(
        "Validating forecasting input schema..."
    )

    sales_required = [
        "week_date",
        "sku",
        "units_demanded",
        "units_sold",
        "lost_sales",
    ]

    readiness_required = [
        "sku",
        "final_demand_class",
        "forecast_strategy",
        "forecast_readiness",
        "forecast_difficulty",
        "history_weeks",
    ]

    behavior_required = [
        "sku",
        "classification_confidence",
    ]

    requirements = {
        "sales":
            sales_required,

        "forecast_readiness":
            readiness_required,

        "demand_behavior":
            behavior_required,
    }

    for dataset_name, columns in requirements.items():

        missing = [
            column
            for column in columns
            if column not in data[
                dataset_name
            ].columns
        ]

        if missing:

            raise ValueError(
                f"{dataset_name} missing "
                f"required columns: {missing}"
            )

    print(
        "✓ Required forecasting columns present"
    )


# =========================================================
# PREPARE WEEKLY DEMAND
# =========================================================

def prepare_weekly_demand(sales):

    print(
        "Preparing weekly forecasting history..."
    )

    df = sales.copy()

    df[
        "week_date"
    ] = pd.to_datetime(
        df["week_date"]
    )

    # -----------------------------------------------------
    # Pre-launch rows are represented by null demand.
    # We MUST preserve that distinction.
    #
    # Zero demand after launch is real zero demand.
    # Null before launch means the product did not yet
    # exist and must not be treated as zero.
    # -----------------------------------------------------

    df[
        "is_prelaunch"
    ] = (
        df[
            "units_demanded"
        ].isna()
    )

    df[
        "forecast_eligible"
    ] = (
        ~df[
            "is_prelaunch"
        ]
    )

    # -----------------------------------------------------
    # True demand target
    # -----------------------------------------------------

    df[
        "forecast_target"
    ] = df[
        "units_demanded"
    ]

    # -----------------------------------------------------
    # Sales capture
    # -----------------------------------------------------

    df[
        "sales_capture_rate"
    ] = np.where(
        df[
            "units_demanded"
        ] > 0,

        df[
            "units_sold"
        ]
        /
        df[
            "units_demanded"
        ],

        np.where(
            df[
                "units_demanded"
            ] == 0,
            1.0,
            np.nan
        )
    )

    # -----------------------------------------------------
    # Sort chronologically
    # -----------------------------------------------------

    df = (
        df
        .sort_values(
            [
                "sku",
                "week_date",
            ]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Time index from first valid demand observation.
    # Pre-launch periods remain NaN.
    # -----------------------------------------------------

    df[
        "demand_week_index"
    ] = np.nan

    valid_mask = (
        df[
            "forecast_eligible"
        ]
    )

    df.loc[
        valid_mask,
        "demand_week_index"
    ] = (
        df[
            valid_mask
        ]
        .groupby(
            "sku"
        )
        .cumcount()
    )

    return df


# =========================================================
# BUILD SKU FORECAST BASE
# =========================================================

def build_sku_forecast_base(
    forecast_readiness,
    demand_behavior
):

    print(
        "Building SKU forecasting registry base..."
    )

    registry = forecast_readiness.copy()

    # -----------------------------------------------------
    # classification_confidence may already exist in
    # forecast_readiness from Phase 4C.
    #
    # Only merge it from demand_behavior if it is missing.
    # This prevents pandas from creating:
    #
    # classification_confidence_x
    # classification_confidence_y
    # -----------------------------------------------------

    if (
        "classification_confidence"
        not in registry.columns
    ):

        behavior_columns = [
            "sku",
            "classification_confidence",
        ]

        registry = registry.merge(
            demand_behavior[
                behavior_columns
            ],
            on="sku",
            how="left",
            validate="one_to_one"
        )

    return registry


# =========================================================
# MODEL CANDIDATE ROUTING
# =========================================================

def get_candidate_models(row):

    demand_class = row[
        "final_demand_class"
    ]

    if demand_class == "stable":

        return [
            "naive",
            "moving_average_4",
            "moving_average_13",
            "simple_exponential_smoothing",
        ]

    if demand_class in [
        "growth",
        "declining",
    ]:

        return [
            "naive",
            "moving_average_4",
            "holt_linear",
            "holt_damped",
        ]

    if demand_class == "seasonal":

        return [
            "naive",
            "seasonal_naive_52",
            
        ]

    if demand_class == "volatile":

        return [
            "naive",
            "moving_average_4",
            "moving_average_13",
            "simple_exponential_smoothing",
        ]

    if demand_class == "intermittent":

        return [
            "naive",
            "croston_classic",
            "croston_sba",
        ]

    if demand_class == "new_product":

        return [
            "naive",
            "recent_average_4",
            "recent_average_8",
            "weighted_recent_average",
            "simple_trend",
        ]

    return [
        "naive"
    ]


def add_candidate_models(registry):

    print(
        "Assigning candidate forecasting models..."
    )

    result = registry.copy()

    result[
        "candidate_models"
    ] = result.apply(
        get_candidate_models,
        axis=1
    )

    result[
        "candidate_model_count"
    ] = result[
        "candidate_models"
    ].apply(
        len
    )

    result[
        "candidate_models"
    ] = result[
        "candidate_models"
    ].apply(
        lambda models:
            "|".join(models)
    )

    return result


# =========================================================
# FORECAST HORIZONS
# =========================================================

def add_forecast_horizons(registry):

    print(
        "Assigning forecast horizons..."
    )

    result = registry.copy()

    result[
        "forecast_horizons_weeks"
    ] = "|".join(
        str(horizon)
        for horizon
        in FORECAST_HORIZONS
    )

    result[
        "primary_forecast_horizon_weeks"
    ] = 13

    return result


# =========================================================
# BACKTEST ELIGIBILITY
# =========================================================

def determine_backtest_eligibility(row):

    history = int(
        row["history_weeks"]
    )

    minimum_training = int(
        row["minimum_training_weeks"]
    )

    backtest_horizon = int(
        row["backtest_horizon_weeks"]
    )

    # A SKU is backtest eligible only when it has
    # enough history for training plus one holdout window.
    required_history = (
        minimum_training
        + backtest_horizon
    )

    return history >= required_history


def add_backtest_eligibility(registry):

    print(
        "Evaluating backtest eligibility..."
    )

    result = registry.copy()

    result[
        "backtest_eligible"
    ] = result.apply(
        determine_backtest_eligibility,
        axis=1
    )

    return result

# =========================================================
# BACKTEST DESIGN
# =========================================================

def assign_backtest_design(row):

    history = row[
        "history_weeks"
    ]

    demand_class = row[
        "final_demand_class"
    ]

    # -----------------------------------------------------
    # New products:
    # short history, so use shorter test windows.
    # -----------------------------------------------------

    if demand_class == "new_product":

        return pd.Series(
            {
                "backtest_horizon_weeks": 4,
                "backtest_step_weeks": 4,
                "minimum_training_weeks": 12,
            }
        )

    # -----------------------------------------------------
    # Seasonal:
    # preserve enough training history for annual seasonality.
    # -----------------------------------------------------

    if demand_class == "seasonal":

        return pd.Series(
            {
                "backtest_horizon_weeks": 13,
                "backtest_step_weeks": 13,
                "minimum_training_weeks": 52,
            }
        )

    # -----------------------------------------------------
    # Standard demand patterns
    # -----------------------------------------------------

    minimum_training = 26

    if history < 52:

        minimum_training = 13

    return pd.Series(
        {
            "backtest_horizon_weeks": 13,
            "backtest_step_weeks": 13,
            "minimum_training_weeks":
                minimum_training,
        }
    )


def add_backtest_design(registry):

    print(
        "Designing rolling backtests..."
    )

    result = registry.copy()

    design = result.apply(
        assign_backtest_design,
        axis=1
    )

    result = pd.concat(
        [
            result,
            design,
        ],
        axis=1
    )

    return result


# =========================================================
# ESTIMATE NUMBER OF BACKTEST WINDOWS
# =========================================================

def estimate_backtest_windows(row):

    if not row[
        "backtest_eligible"
    ]:

        return 0

    history = int(
        row[
            "history_weeks"
        ]
    )

    minimum_training = int(
        row[
            "minimum_training_weeks"
        ]
    )

    horizon = int(
        row[
            "backtest_horizon_weeks"
        ]
    )

    step = int(
        row[
            "backtest_step_weeks"
        ]
    )

    remaining = (
        history
        -
        minimum_training
        -
        horizon
    )

    if remaining < 0:

        return 0

    return (
        1
        +
        remaining // step
    )


def add_backtest_window_counts(registry):

    print(
        "Estimating rolling backtest windows..."
    )

    result = registry.copy()

    result[
        "estimated_backtest_windows"
    ] = result.apply(
        estimate_backtest_windows,
        axis=1
    )

    return result


# =========================================================
# MODEL SELECTION METRICS
# =========================================================

def add_model_selection_policy(registry):

    print(
        "Assigning model-selection metrics..."
    )

    result = registry.copy()

    # WAPE is the primary aggregate selection metric.
    # MAE remains useful and easy to interpret.
    # Bias protects us from systematically under-forecasting.
    result[
        "primary_selection_metric"
    ] = "wape"

    result[
        "secondary_selection_metric"
    ] = "mae"

    result[
        "bias_metric"
    ] = "forecast_bias"

    # Intermittent demand has many zeros.
    # We flag it so 6B can handle evaluation carefully.
    result[
        "intermittent_evaluation_flag"
    ] = (
        result[
            "final_demand_class"
        ] == "intermittent"
    )

    return result


# =========================================================
# BUILD MODEL REGISTRY
# =========================================================

def build_model_registry(
    forecast_readiness,
    demand_behavior
):

    registry = build_sku_forecast_base(
        forecast_readiness,
        demand_behavior
    )

    registry = add_candidate_models(
        registry
    )

    registry = add_forecast_horizons(
        registry
    )

    # Build backtest design first.
    # This creates minimum_training_weeks,
    # backtest_horizon_weeks, and backtest_step_weeks.
    registry = add_backtest_design(
        registry
    )

    # Now eligibility can safely use the
    # backtest design columns.
    registry = add_backtest_eligibility(
        registry
    )

    registry = add_backtest_window_counts(
        registry
    )

    registry = add_model_selection_policy(
        registry
    )

    return registry


# =========================================================
# VALIDATION
# =========================================================

def validate_outputs(
    forecasting_dataset,
    registry,
    sales,
    readiness
):

    print("\n" + "=" * 90)
    print("FORECASTING FRAMEWORK VALIDATION")
    print("=" * 90)

    assert len(
        forecasting_dataset
    ) == len(
        sales
    )

    print(
        f"✓ Forecast dataset preserves "
        f"{len(sales):,} weekly rows"
    )

    assert len(
        registry
    ) == len(
        readiness
    )

    print(
        f"✓ Exactly {len(registry)} "
        "SKU registry rows"
    )

    assert registry[
        "sku"
    ].is_unique

    print(
        "✓ One registry row per SKU"
    )

    assert set(
        registry["sku"]
    ) == set(
        readiness["sku"]
    )

    print(
        "✓ All forecast-readiness SKUs preserved"
    )

    # -----------------------------------------------------
    # Pre-launch rows must remain null.
    # -----------------------------------------------------

    prelaunch = forecasting_dataset[
        forecasting_dataset[
            "is_prelaunch"
        ]
    ]

    assert prelaunch[
        "forecast_target"
    ].isna().all()

    print(
        f"✓ Pre-launch demand remains null: "
        f"{len(prelaunch):,} rows"
    )

    # -----------------------------------------------------
    # Eligible history cannot have missing targets.
    # -----------------------------------------------------

    eligible = forecasting_dataset[
        forecasting_dataset[
            "forecast_eligible"
        ]
    ]

    assert eligible[
        "forecast_target"
    ].notna().all()

    print(
        "✓ Forecast-eligible demand has "
        "no missing targets"
    )

    assert (
        eligible[
            "forecast_target"
        ] >= 0
    ).all()

    print(
        "✓ Forecast targets are non-negative"
    )

    # -----------------------------------------------------
    # Candidate models
    # -----------------------------------------------------

    assert (
        registry[
            "candidate_model_count"
        ] >= 1
    ).all()

    print(
        "✓ Every SKU has candidate models"
    )

    # -----------------------------------------------------
    # Strategy routing checks
    # -----------------------------------------------------

    intermittent = registry[
        registry[
            "final_demand_class"
        ] == "intermittent"
    ]

    assert intermittent[
        "candidate_models"
    ].str.contains(
        "croston"
    ).all()

    print(
        "✓ Intermittent SKUs receive "
        "Croston candidates"
    )

    seasonal = registry[
        registry[
            "final_demand_class"
        ] == "seasonal"
    ]

    assert seasonal[
        "candidate_models"
    ].str.contains(
        "seasonal_naive_52"
    ).all()

    print(
        "✓ Seasonal SKUs receive "
        "seasonal candidates"
    )

    new_products = registry[
        registry[
            "final_demand_class"
        ] == "new_product"
    ]

    assert new_products[
        "candidate_models"
    ].str.contains(
        "weighted_recent_average"
    ).all()

    print(
        "✓ New products receive "
        "short-history candidates"
    )

    # -----------------------------------------------------
    # Backtest windows
    # -----------------------------------------------------

    assert (
        registry[
            "estimated_backtest_windows"
        ] >= 0
    ).all()

    print(
        "✓ Backtest window counts valid"
    )

    # -----------------------------------------------------
    # Sales capture
    # -----------------------------------------------------

    valid_capture = eligible[
        "sales_capture_rate"
    ].dropna()

    assert (
        (
            valid_capture >= 0
        )
        &
        (
            valid_capture <= 1
        )
    ).all()

    print(
        "✓ Sales-capture rates valid"
    )

    print(
        "\nAll Phase 6A validation "
        "checks passed."
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(
    forecasting_dataset,
    registry
):

    forecasting_dataset.to_csv(
        FORECASTING_DATASET_FILE,
        index=False
    )

    registry.to_csv(
        MODEL_REGISTRY_FILE,
        index=False
    )

    print(
        f"\nCreated: "
        f"{FORECASTING_DATASET_FILE}"
    )

    print(
        f"Created: "
        f"{MODEL_REGISTRY_FILE}"
    )


# =========================================================
# PRINT ROUTING SUMMARY
# =========================================================

def print_routing_summary(registry):

    print("\n" + "=" * 120)
    print("FORECAST MODEL ROUTING")
    print("=" * 120)

    summary = (
        registry
        .groupby(
            [
                "final_demand_class",
                "forecast_strategy",
            ],
            dropna=False
        )
        .agg(
            sku_count=(
                "sku",
                "count"
            ),

            avg_history_weeks=(
                "history_weeks",
                "mean"
            ),

            avg_candidate_models=(
                "candidate_model_count",
                "mean"
            ),

            avg_backtest_windows=(
                "estimated_backtest_windows",
                "mean"
            ),
        )
        .round(2)
    )

    print(summary)


# =========================================================
# PRINT MODEL REGISTRY EXAMPLES
# =========================================================

def print_registry_examples(registry):

    print("\n" + "=" * 160)
    print("EXAMPLE FORECAST MODEL REGISTRY")
    print("=" * 160)

    examples = (
        registry
        .groupby(
            "final_demand_class",
            group_keys=False
        )
        .head(2)
    )

    columns = [
        "sku",
        "final_demand_class",
        "classification_confidence",
        "history_weeks",
        "forecast_readiness",
        "forecast_difficulty",
        "candidate_models",
        "backtest_horizon_weeks",
        "minimum_training_weeks",
        "estimated_backtest_windows",
    ]

    print(
        examples[
            columns
        ]
        .sort_values(
            [
                "final_demand_class",
                "sku",
            ]
        )
        .round(3)
        .to_string(
            index=False
        )
    )


# =========================================================
# PRINT DATASET SUMMARY
# =========================================================

def print_dataset_summary(
    forecasting_dataset
):

    print("\n" + "=" * 90)
    print("FORECASTING DATASET SUMMARY")
    print("=" * 90)

    print(
        f"Rows: "
        f"{len(forecasting_dataset):,}"
    )

    print(
        f"SKUs: "
        f"{forecasting_dataset['sku'].nunique():,}"
    )

    print(
        f"Forecast-eligible observations: "
        f"{forecasting_dataset['forecast_eligible'].sum():,}"
    )

    print(
        f"Pre-launch observations: "
        f"{forecasting_dataset['is_prelaunch'].sum():,}"
    )

    print(
        f"Total true demand: "
        f"{forecasting_dataset['forecast_target'].sum():,.0f}"
    )

    print(
        f"Total units sold: "
        f"{forecasting_dataset['units_sold'].sum():,.0f}"
    )

    print(
        f"Total lost sales: "
        f"{forecasting_dataset['lost_sales'].sum():,.0f}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 90)

    print(
        "PROCUREMENT AI - "
        "FORECASTING FRAMEWORK"
    )

    print("=" * 90)

    data = load_data()

    validate_input_schema(
        data
    )

    forecasting_dataset = (
        prepare_weekly_demand(
            data["sales"]
        )
    )

    registry = (
        build_model_registry(
            data[
                "forecast_readiness"
            ],
            data[
                "demand_behavior"
            ]
        )
    )

    validate_outputs(
        forecasting_dataset,
        registry,
        data["sales"],
        data["forecast_readiness"]
    )

    save_outputs(
        forecasting_dataset,
        registry
    )

    print_dataset_summary(
        forecasting_dataset
    )

    print_routing_summary(
        registry
    )

    print_registry_examples(
        registry
    )

    print("\n" + "=" * 90)

    print(
        "PHASE 6A COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()