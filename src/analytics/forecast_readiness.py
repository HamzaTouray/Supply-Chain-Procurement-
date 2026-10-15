from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

# These are diagnostic thresholds, not forecasting-model
# hyperparameters. We will validate model choices later
# through backtesting.

MIN_STANDARD_HISTORY = 26
MIN_TREND_HISTORY = 52
MIN_SEASONAL_HISTORY = 90

HIGH_CV_THRESHOLD = 0.40
VERY_HIGH_CV_THRESHOLD = 0.80

INTERMITTENT_ZERO_RATE = 0.30
INTERMITTENT_ADI = 1.32

LOST_SALES_WARNING_RATE = 0.05
SEVERE_LOST_SALES_RATE = 0.10


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

DEMAND_BEHAVIOR_FILE = (
    PROCESSED_DATA_DIR
    / "demand_behavior.csv"
)

PLANNING_MASTER_FILE = (
    PROCESSED_DATA_DIR
    / "planning_master.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "forecast_readiness.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\nLoading Phase 4C inputs...")

    required_files = {
        "demand_behavior": DEMAND_BEHAVIOR_FILE,
        "planning_master": PLANNING_MASTER_FILE,
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

    return data


# =========================================================
# BUILD BASE TABLE
# =========================================================

def build_base_table(
    demand_behavior,
    planning_master
):

    print(
        "\nBuilding forecast-readiness base..."
    )

    # -----------------------------------------------------
    # Only bring operational fields that are useful for
    # diagnostics. Demand behavior remains the main table.
    # -----------------------------------------------------

    planning_columns = [
        "sku",
        "net_available",
        "current_weeks_of_supply",
        "total_open_po_qty",
        "has_overdue_po",
    ]

    missing_columns = [
        col
        for col in planning_columns
        if col not in planning_master.columns
    ]

    if missing_columns:

        raise ValueError(
            "planning_master.csv is missing columns: "
            f"{missing_columns}"
        )

    planning_subset = (
        planning_master[
            planning_columns
        ]
        .copy()
    )

    result = (
        demand_behavior
        .merge(
            planning_subset,
            on="sku",
            how="left",
            validate="one_to_one"
        )
    )

    print(
        f"✓ Forecast-readiness base: "
        f"{len(result):,} SKUs"
    )

    return result


# =========================================================
# HISTORY FLAGS
# =========================================================

def add_history_flags(
    df
):

    print(
        "Adding history diagnostics..."
    )

    result = df.copy()

    result[
        "short_history_flag"
    ] = (
        result[
            "history_weeks"
        ] < MIN_STANDARD_HISTORY
    )

    result[
        "sufficient_standard_history"
    ] = (
        result[
            "history_weeks"
        ] >= MIN_STANDARD_HISTORY
    )

    result[
        "trend_model_eligible"
    ] = (
        result[
            "history_weeks"
        ] >= MIN_TREND_HISTORY
    )

    result[
        "seasonal_model_eligible"
    ] = (
        (
            result[
                "history_weeks"
            ] >= MIN_SEASONAL_HISTORY
        )
        &
        (
            result[
                "final_demand_class"
            ] == "seasonal"
        )
    )

    return result


# =========================================================
# VARIABILITY FLAGS
# =========================================================

def add_variability_flags(
    df
):

    print(
        "Adding variability diagnostics..."
    )

    result = df.copy()

    result[
        "high_variability_flag"
    ] = (
        result[
            "52w_cv"
        ] >= HIGH_CV_THRESHOLD
    )

    result[
        "very_high_variability_flag"
    ] = (
        result[
            "52w_cv"
        ] >= VERY_HIGH_CV_THRESHOLD
    )

    return result


# =========================================================
# INTERMITTENCY FLAGS
# =========================================================

def add_intermittency_flags(
    df
):

    print(
        "Adding intermittency diagnostics..."
    )

    result = df.copy()

    result[
        "intermittent_model_required"
    ] = (
        (
            result[
                "final_demand_class"
            ] == "intermittent"
        )
        |
        (
            result[
                "zero_demand_rate"
            ] >= INTERMITTENT_ZERO_RATE
        )
        |
        (
            result[
                "adi"
            ] >= INTERMITTENT_ADI
        )
    )

    return result


# =========================================================
# LOST SALES / CENSORING FLAGS
# =========================================================

def add_lost_sales_flags(
    df
):

    print(
        "Adding lost-sales diagnostics..."
    )

    result = df.copy()

    # -----------------------------------------------------
    # Recent lost sales matter more for the immediate
    # forecasting problem than lifetime lost sales.
    # -----------------------------------------------------

    result[
        "lost_sales_bias_flag"
    ] = (
        result[
            "13w_lost_sales_rate"
        ] >= LOST_SALES_WARNING_RATE
    )

    result[
        "severe_lost_sales_bias_flag"
    ] = (
        result[
            "13w_lost_sales_rate"
        ] >= SEVERE_LOST_SALES_RATE
    )

    # -----------------------------------------------------
    # Difference between true demand and observed sales.
    #
    # This quantifies how much a sales-only forecast could
    # understate demand in the recent period.
    # -----------------------------------------------------

    result[
        "13w_sales_capture_rate"
    ] = np.where(
        result[
            "13w_total_demand"
        ] > 0,

        result[
            "13w_units_sold"
        ]
        /
        result[
            "13w_total_demand"
        ],

        1.0
    )

    result[
        "13w_demand_hidden_by_stockout_pct"
    ] = (
        1
        -
        result[
            "13w_sales_capture_rate"
        ]
    )

    result[
        "13w_demand_hidden_by_stockout_pct"
    ] = (
        result[
            "13w_demand_hidden_by_stockout_pct"
        ]
        .clip(
            lower=0,
            upper=1
        )
    )

    return result


# =========================================================
# DEMAND MOMENTUM FLAGS
# =========================================================

def add_momentum_flags(
    df
):

    print(
        "Adding demand-momentum diagnostics..."
    )

    result = df.copy()

    result[
        "recent_demand_above_baseline"
    ] = (
        result[
            "13w_vs_52w_ratio"
        ] >= 1.10
    )

    result[
        "recent_demand_below_baseline"
    ] = (
        result[
            "13w_vs_52w_ratio"
        ] <= 0.90
    )

    result[
        "strong_acceleration_flag"
    ] = (
        result[
            "demand_acceleration"
        ] >= 0.25
    )

    result[
        "strong_deceleration_flag"
    ] = (
        result[
            "demand_acceleration"
        ] <= -0.25
    )

    return result


# =========================================================
# FORECAST STRATEGY
# =========================================================

def assign_forecast_strategy(
    df
):

    print(
        "Assigning forecast strategies..."
    )

    result = df.copy()

    strategies = []

    for _, row in result.iterrows():

        demand_class = (
            row[
                "final_demand_class"
            ]
        )

        history = (
            row[
                "history_weeks"
            ]
        )

        # -------------------------------------------------
        # NEW PRODUCT
        # -------------------------------------------------

        if (
            demand_class == "new_product"
            or history < MIN_STANDARD_HISTORY
        ):

            strategy = (
                "short_history"
            )

        # -------------------------------------------------
        # INTERMITTENT
        # -------------------------------------------------

        elif (
            row[
                "intermittent_model_required"
            ]
        ):

            strategy = (
                "intermittent"
            )

        # -------------------------------------------------
        # SEASONAL
        # -------------------------------------------------

        elif (
            demand_class == "seasonal"
            and row[
                "seasonal_model_eligible"
            ]
        ):

            strategy = (
                "seasonal"
            )

        # -------------------------------------------------
        # TREND
        # -------------------------------------------------

        elif (
            demand_class
            in [
                "growth",
                "declining",
            ]
            and row[
                "trend_model_eligible"
            ]
        ):

            strategy = (
                "trend"
            )

        # -------------------------------------------------
        # VOLATILE
        # -------------------------------------------------

        elif (
            demand_class == "volatile"
        ):

            strategy = (
                "volatile"
            )

        # -------------------------------------------------
        # STABLE
        # -------------------------------------------------

        else:

            strategy = (
                "level"
            )

        strategies.append(
            strategy
        )

    result[
        "forecast_strategy"
    ] = strategies

    return result


# =========================================================
# FORECAST DIFFICULTY
# =========================================================

def assign_forecast_difficulty(
    df
):

    print(
        "Scoring forecast difficulty..."
    )

    result = df.copy()

    scores = []

    for _, row in result.iterrows():

        score = 0

        # Limited history
        if row[
            "history_weeks"
        ] < 52:

            score += 2

        # Intermittency
        if row[
            "intermittent_model_required"
        ]:

            score += 2

        # High variability
        if row[
            "high_variability_flag"
        ]:

            score += 1

        # Extreme variability
        if row[
            "very_high_variability_flag"
        ]:

            score += 1

        # Lost-sales censoring
        if row[
            "lost_sales_bias_flag"
        ]:

            score += 1

        if row[
            "severe_lost_sales_bias_flag"
        ]:

            score += 1

        # Strong recent acceleration/deceleration
        if (
            row[
                "strong_acceleration_flag"
            ]
            or row[
                "strong_deceleration_flag"
            ]
        ):

            score += 1

        # Low classification confidence
        if (
            row[
                "classification_confidence"
            ] < 0.50
        ):

            score += 1

        scores.append(
            score
        )

    result[
        "forecast_difficulty_score"
    ] = scores

    difficulty_labels = []

    for score in scores:

        if score <= 1:

            label = "low"

        elif score <= 3:

            label = "medium"

        else:

            label = "high"

        difficulty_labels.append(
            label
        )

    result[
        "forecast_difficulty"
    ] = difficulty_labels

    return result


# =========================================================
# FORECAST READINESS
# =========================================================

def assign_forecast_readiness(
    df
):

    print(
        "Assigning forecast readiness..."
    )

    result = df.copy()

    readiness = []

    for _, row in result.iterrows():

        demand_class = (
            row[
                "final_demand_class"
            ]
        )

        history = (
            row[
                "history_weeks"
            ]
        )

        difficulty = (
            row[
                "forecast_difficulty"
            ]
        )

        # -------------------------------------------------
        # NEW PRODUCTS
        # -------------------------------------------------

        if (
            demand_class == "new_product"
            or history < MIN_STANDARD_HISTORY
        ):

            status = "limited"

        # -------------------------------------------------
        # INTERMITTENT
        # -------------------------------------------------

        elif (
            row[
                "intermittent_model_required"
            ]
        ):

            status = "specialized"

        # -------------------------------------------------
        # SEASONAL WITHOUT ENOUGH HISTORY
        # -------------------------------------------------

        elif (
            demand_class == "seasonal"
            and not row[
                "seasonal_model_eligible"
            ]
        ):

            status = "limited"

        # -------------------------------------------------
        # HIGH DIFFICULTY
        # -------------------------------------------------

        elif difficulty == "high":

            status = "caution"

        # -------------------------------------------------
        # OTHERWISE READY
        # -------------------------------------------------

        else:

            status = "ready"

        readiness.append(
            status
        )

    result[
        "forecast_readiness"
    ] = readiness

    return result


# =========================================================
# DATA QUALITY / MODEL WARNINGS
# =========================================================

def build_forecast_warning(
    row
):

    warnings = []

    if row[
        "short_history_flag"
    ]:

        warnings.append(
            "short history"
        )

    if row[
        "intermittent_model_required"
    ]:

        warnings.append(
            "intermittent demand"
        )

    if row[
        "very_high_variability_flag"
    ]:

        warnings.append(
            "very high variability"
        )

    elif row[
        "high_variability_flag"
    ]:

        warnings.append(
            "high variability"
        )

    if row[
        "severe_lost_sales_bias_flag"
    ]:

        warnings.append(
            "severe lost-sales bias"
        )

    elif row[
        "lost_sales_bias_flag"
    ]:

        warnings.append(
            "lost-sales bias"
        )

    if row[
        "strong_acceleration_flag"
    ]:

        warnings.append(
            "demand accelerating"
        )

    if row[
        "strong_deceleration_flag"
    ]:

        warnings.append(
            "demand decelerating"
        )

    if (
        row[
            "classification_confidence"
        ] < 0.50
    ):

        warnings.append(
            "low behavior confidence"
        )

    if not warnings:

        return "none"

    return "; ".join(
        warnings
    )


def add_forecast_warnings(
    df
):

    print(
        "Building forecast warnings..."
    )

    result = df.copy()

    result[
        "forecast_warning"
    ] = result.apply(
        build_forecast_warning,
        axis=1
    )

    return result


# =========================================================
# BUILD FINAL TABLE
# =========================================================

def build_forecast_readiness(
    demand_behavior,
    planning_master
):

    result = build_base_table(
        demand_behavior,
        planning_master
    )

    result = add_history_flags(
        result
    )

    result = add_variability_flags(
        result
    )

    result = add_intermittency_flags(
        result
    )

    result = add_lost_sales_flags(
        result
    )

    result = add_momentum_flags(
        result
    )

    result = assign_forecast_strategy(
        result
    )

    result = assign_forecast_difficulty(
        result
    )

    result = assign_forecast_readiness(
        result
    )

    result = add_forecast_warnings(
        result
    )

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate_output(
    result,
    demand_behavior
):

    print("\n" + "=" * 80)
    print("FORECAST READINESS VALIDATION")
    print("=" * 80)

    assert len(
        result
    ) == 200

    print(
        "✓ Exactly 200 SKU rows"
    )

    assert (
        result[
            "sku"
        ].is_unique
    )

    print(
        "✓ One row per SKU"
    )

    assert set(
        result["sku"]
    ) == set(
        demand_behavior["sku"]
    )

    print(
        "✓ All demand-behavior SKUs preserved"
    )

    # -----------------------------------------------------
    # STRATEGY VALUES
    # -----------------------------------------------------

    valid_strategies = {
        "short_history",
        "intermittent",
        "seasonal",
        "trend",
        "volatile",
        "level",
    }

    assert set(
        result[
            "forecast_strategy"
        ].unique()
    ).issubset(
        valid_strategies
    )

    print(
        "✓ Forecast strategies valid"
    )

    # -----------------------------------------------------
    # READINESS VALUES
    # -----------------------------------------------------

    valid_readiness = {
        "ready",
        "specialized",
        "limited",
        "caution",
    }

    assert set(
        result[
            "forecast_readiness"
        ].unique()
    ).issubset(
        valid_readiness
    )

    print(
        "✓ Forecast-readiness values valid"
    )

    # -----------------------------------------------------
    # DIFFICULTY VALUES
    # -----------------------------------------------------

    valid_difficulty = {
        "low",
        "medium",
        "high",
    }

    assert set(
        result[
            "forecast_difficulty"
        ].unique()
    ).issubset(
        valid_difficulty
    )

    print(
        "✓ Forecast-difficulty values valid"
    )

    # -----------------------------------------------------
    # SALES CAPTURE RATE
    # -----------------------------------------------------

    assert (
        (
            result[
                "13w_sales_capture_rate"
            ] >= 0
        )
        &
        (
            result[
                "13w_sales_capture_rate"
            ] <= 1
        )
    ).all()

    print(
        "✓ Sales-capture rates valid"
    )

    # -----------------------------------------------------
    # NEW PRODUCTS
    # -----------------------------------------------------

    new_products = result[
        result[
            "final_demand_class"
        ] == "new_product"
    ]

    assert (
        new_products[
            "forecast_strategy"
        ] == "short_history"
    ).all()

    print(
        "✓ New products routed to "
        "short-history strategy"
    )

    # -----------------------------------------------------
    # INTERMITTENT
    # -----------------------------------------------------

    intermittent = result[
        result[
            "final_demand_class"
        ] == "intermittent"
    ]

    assert (
        intermittent[
            "forecast_strategy"
        ] == "intermittent"
    ).all()

    print(
        "✓ Intermittent SKUs routed to "
        "specialized strategy"
    )

    # -----------------------------------------------------
    # SEASONAL
    # -----------------------------------------------------

    seasonal_eligible = result[
        (
            result[
                "final_demand_class"
            ] == "seasonal"
        )
        &
        (
            result[
                "seasonal_model_eligible"
            ]
        )
    ]

    assert (
        seasonal_eligible[
            "forecast_strategy"
        ] == "seasonal"
    ).all()

    print(
        "✓ Eligible seasonal SKUs routed "
        "to seasonal strategy"
    )

    # -----------------------------------------------------
    # NO NEGATIVE DIFFICULTY
    # -----------------------------------------------------

    assert (
        result[
            "forecast_difficulty_score"
        ] >= 0
    ).all()

    print(
        "✓ Forecast difficulty scores valid"
    )

    print(
        "\nAll Phase 4C validation "
        "checks passed."
    )


# =========================================================
# PRINT STRATEGY SUMMARY
# =========================================================

def print_strategy_summary(
    result
):

    print("\n" + "=" * 90)
    print("FORECAST STRATEGY SUMMARY")
    print("=" * 90)

    summary = pd.crosstab(
        result[
            "final_demand_class"
        ],
        result[
            "forecast_strategy"
        ]
    )

    print(
        summary.to_string()
    )


# =========================================================
# PRINT READINESS SUMMARY
# =========================================================

def print_readiness_summary(
    result
):

    print("\n" + "=" * 80)
    print("FORECAST READINESS SUMMARY")
    print("=" * 80)

    print(
        result[
            "forecast_readiness"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nForecast difficulty:"
    )

    print(
        result[
            "forecast_difficulty"
        ]
        .value_counts()
        .to_string()
    )


# =========================================================
# LOST SALES SUMMARY
# =========================================================

def print_lost_sales_summary(
    result
):

    print("\n" + "=" * 90)
    print("LOST SALES / DEMAND CENSORING SUMMARY")
    print("=" * 90)

    flagged = int(
        result[
            "lost_sales_bias_flag"
        ].sum()
    )

    severe = int(
        result[
            "severe_lost_sales_bias_flag"
        ].sum()
    )

    avg_capture = (
        result[
            "13w_sales_capture_rate"
        ].mean()
    )

    print(
        f"SKUs with lost-sales bias warning: "
        f"{flagged}"
    )

    print(
        f"SKUs with severe lost-sales bias: "
        f"{severe}"
    )

    print(
        f"Average 13W sales capture rate: "
        f"{avg_capture:.2%}"
    )

    if flagged > 0:

        print(
            "\nHighest recent lost-sales rates:"
        )

        columns = [
            "sku",
            "final_demand_class",
            "13w_avg_demand",
            "13w_lost_sales_rate",
            "13w_sales_capture_rate",
            "net_available",
            "current_weeks_of_supply",
        ]

        examples = (
            result
            .sort_values(
                "13w_lost_sales_rate",
                ascending=False
            )
            .head(10)
        )

        print(
            examples[
                columns
            ]
            .round(3)
            .to_string(
                index=False
            )
        )


# =========================================================
# EXAMPLES
# =========================================================

def print_examples(
    result
):

    print("\n" + "=" * 130)
    print("EXAMPLE FORECAST READINESS RECORDS")
    print("=" * 130)

    columns = [
        "sku",
        "final_demand_class",
        "classification_confidence",
        "history_weeks",
        "13w_avg_demand",
        "52w_cv",
        "zero_demand_rate",
        "adi",
        "13w_lost_sales_rate",
        "forecast_strategy",
        "forecast_difficulty",
        "forecast_readiness",
        "forecast_warning",
    ]

    examples = (
        result
        .groupby(
            "final_demand_class",
            group_keys=False
        )
        .head(2)
    )

    print(
        examples[
            columns
        ]
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

    print("=" * 80)

    print(
        "PROCUREMENT AI - "
        "FORECAST READINESS & DEMAND DIAGNOSTICS"
    )

    print("=" * 80)

    data = load_data()

    result = (
        build_forecast_readiness(
            data[
                "demand_behavior"
            ],
            data[
                "planning_master"
            ]
        )
    )

    validate_output(
        result,
        data[
            "demand_behavior"
        ]
    )

    save_output(
        result
    )

    print_strategy_summary(
        result
    )

    print_readiness_summary(
        result
    )

    print_lost_sales_summary(
        result
    )

    print_examples(
        result
    )

    print("\n" + "=" * 80)

    print(
        "PHASE 4C COMPLETE"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()