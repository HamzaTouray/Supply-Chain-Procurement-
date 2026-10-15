from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import (
    ExponentialSmoothing,
    Holt,
    SimpleExpSmoothing,
)


# =========================================================
# SETTINGS
# =========================================================

# A more complex model must beat naive by at least this
# amount before we prefer it over the simple baseline.
MIN_IMPROVEMENT_OVER_NAIVE = 0.03

EPSILON = 1e-9


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

FORECASTING_DATA_FILE = (
    PROCESSED_DATA_DIR
    / "forecasting_dataset.csv"
)

MODEL_REGISTRY_FILE = (
    PROCESSED_DATA_DIR
    / "forecast_model_registry.csv"
)

BACKTEST_RESULTS_FILE = (
    PROCESSED_DATA_DIR
    / "forecast_backtest_results.csv"
)

MODEL_SELECTION_FILE = (
    PROCESSED_DATA_DIR
    / "forecast_model_selection.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\nLoading Phase 6B inputs...")

    required_files = {
        "forecasting_dataset":
            FORECASTING_DATA_FILE,

        "model_registry":
            MODEL_REGISTRY_FILE,
    }

    data = {}

    for name, path in required_files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

        data[name] = pd.read_csv(path)

        print(
            f"✓ {name:<24}"
            f"{len(data[name]):>8,} rows"
        )

    return data


# =========================================================
# VALIDATE INPUTS
# =========================================================

def validate_inputs(data):

    print(
        "Validating Phase 6B input schema..."
    )

    history_required = [
        "week_date",
        "sku",
        "forecast_target",
        "forecast_eligible",
    ]

    registry_required = [
        "sku",
        "final_demand_class",
        "candidate_models",
        "backtest_eligible",
        "backtest_horizon_weeks",
        "backtest_step_weeks",
        "minimum_training_weeks",
        "estimated_backtest_windows",
    ]

    for column in history_required:

        if column not in data[
            "forecasting_dataset"
        ].columns:

            raise ValueError(
                "forecasting_dataset missing "
                f"required column: {column}"
            )

    for column in registry_required:

        if column not in data[
            "model_registry"
        ].columns:

            raise ValueError(
                "forecast_model_registry missing "
                f"required column: {column}"
            )

    print(
        "✓ Required Phase 6B columns present"
    )


# =========================================================
# BOOLEAN NORMALIZATION
# =========================================================

def normalize_boolean(series):

    if pd.api.types.is_bool_dtype(series):
        return series

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


# =========================================================
# PREPARE HISTORY
# =========================================================

def prepare_history(df):

    print(
        "Preparing SKU demand histories..."
    )

    result = df.copy()

    result[
        "week_date"
    ] = pd.to_datetime(
        result[
            "week_date"
        ]
    )

    result[
        "forecast_eligible"
    ] = normalize_boolean(
        result[
            "forecast_eligible"
        ]
    )

    result = result[
        result[
            "forecast_eligible"
        ]
    ].copy()

    result[
        "forecast_target"
    ] = pd.to_numeric(
        result[
            "forecast_target"
        ],
        errors="coerce"
    )

    result = result[
        result[
            "forecast_target"
        ].notna()
    ].copy()

    result = (
        result
        .sort_values(
            [
                "sku",
                "week_date",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"✓ Forecast observations: "
        f"{len(result):,}"
    )

    print(
        f"✓ Forecast SKUs: "
        f"{result['sku'].nunique():,}"
    )

    return result


# =========================================================
# FORECAST HELPERS
# =========================================================

def sanitize_forecast(values, horizon):

    values = np.asarray(
        values,
        dtype=float
    ).reshape(-1)

    if len(values) != horizon:

        raise ValueError(
            "Forecast length does not match horizon"
        )

    values = np.where(
        np.isfinite(values),
        values,
        0.0
    )

    # Demand cannot be negative.
    values = np.maximum(
        values,
        0.0
    )

    return values


def naive_forecast(train, horizon):

    last_value = float(
        train[-1]
    )

    return np.repeat(
        last_value,
        horizon
    )


def moving_average_forecast(
    train,
    horizon,
    window
):

    window = min(
        window,
        len(train)
    )

    level = float(
        np.mean(
            train[-window:]
        )
    )

    return np.repeat(
        level,
        horizon
    )


def recent_average_forecast(
    train,
    horizon,
    window
):

    return moving_average_forecast(
        train,
        horizon,
        window
    )


def weighted_recent_average_forecast(
    train,
    horizon
):

    window = min(
        8,
        len(train)
    )

    recent = np.asarray(
        train[-window:],
        dtype=float
    )

    weights = np.arange(
        1,
        window + 1,
        dtype=float
    )

    level = np.average(
        recent,
        weights=weights
    )

    return np.repeat(
        float(level),
        horizon
    )


def simple_trend_forecast(
    train,
    horizon
):

    window = min(
        12,
        len(train)
    )

    y = np.asarray(
        train[-window:],
        dtype=float
    )

    x = np.arange(
        window,
        dtype=float
    )

    if window < 2:

        return naive_forecast(
            train,
            horizon
        )

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    future_x = np.arange(
        window,
        window + horizon,
        dtype=float
    )

    forecast = (
        intercept
        +
        slope * future_x
    )

    return np.maximum(
        forecast,
        0
    )


# =========================================================
# EXPONENTIAL SMOOTHING MODELS
# =========================================================

def ses_forecast(
    train,
    horizon
):

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        model = SimpleExpSmoothing(
            np.asarray(
                train,
                dtype=float
            ),
            initialization_method="estimated"
        )

        fitted = model.fit(
            optimized=True
        )

    return np.asarray(
        fitted.forecast(
            horizon
        )
    )


def holt_forecast(
    train,
    horizon,
    damped=False
):

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        model = Holt(
            np.asarray(
                train,
                dtype=float
            ),
            damped_trend=damped,
            initialization_method="estimated"
        )

        fitted = model.fit(
            optimized=True
        )

    return np.asarray(
        fitted.forecast(
            horizon
        )
    )


def holt_winters_additive_forecast(
    train,
    horizon
):

    if len(train) < 104:

        raise ValueError(
            "Holt-Winters annual seasonality "
            "requires at least 104 training weeks"
        )

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore"
        )

        model = ExponentialSmoothing(
            np.asarray(
                train,
                dtype=float
            ),
            trend=None,
            seasonal="add",
            seasonal_periods=52,
            initialization_method="estimated"
        )

        fitted = model.fit(
            optimized=True
        )

    return np.asarray(
        fitted.forecast(
            horizon
        )
    )


# =========================================================
# SEASONAL NAIVE
# =========================================================

def seasonal_naive_forecast(
    train,
    horizon,
    seasonal_period=52
):

    if len(train) < seasonal_period:

        raise ValueError(
            "Not enough history for "
            "seasonal naive forecast"
        )

    seasonal_values = np.asarray(
        train[
            -seasonal_period:
        ],
        dtype=float
    )

    forecast = []

    for step in range(horizon):

        forecast.append(
            seasonal_values[
                step % seasonal_period
            ]
        )

    return np.asarray(
        forecast
    )


# =========================================================
# CROSTON
# =========================================================

def croston_forecast(
    train,
    horizon,
    alpha=0.10,
    sba=False
):

    demand = np.asarray(
        train,
        dtype=float
    )

    positive_indices = np.where(
        demand > 0
    )[0]

    if len(
        positive_indices
    ) == 0:

        return np.zeros(
            horizon
        )

    first_index = int(
        positive_indices[0]
    )

    demand_estimate = float(
        demand[first_index]
    )

    interval_estimate = float(
        first_index + 1
    )

    last_positive_index = (
        first_index
    )

    for index in positive_indices[1:]:

        quantity = float(
            demand[index]
        )

        interval = float(
            index
            -
            last_positive_index
        )

        demand_estimate = (
            demand_estimate
            +
            alpha
            * (
                quantity
                -
                demand_estimate
            )
        )

        interval_estimate = (
            interval_estimate
            +
            alpha
            * (
                interval
                -
                interval_estimate
            )
        )

        last_positive_index = int(
            index
        )

    if interval_estimate <= 0:

        rate = 0.0

    else:

        rate = (
            demand_estimate
            /
            interval_estimate
        )

    if sba:

        rate = (
            1
            -
            alpha / 2
        ) * rate

    return np.repeat(
        max(
            float(rate),
            0.0
        ),
        horizon
    )


# =========================================================
# MODEL DISPATCHER
# =========================================================

def forecast_model(
    model_name,
    train,
    horizon
):

    train = np.asarray(
        train,
        dtype=float
    )

    if len(train) == 0:

        raise ValueError(
            "Training data is empty"
        )

    if model_name == "naive":

        forecast = naive_forecast(
            train,
            horizon
        )

    elif model_name == "moving_average_4":

        forecast = moving_average_forecast(
            train,
            horizon,
            4
        )

    elif model_name == "moving_average_13":

        forecast = moving_average_forecast(
            train,
            horizon,
            13
        )

    elif model_name == "simple_exponential_smoothing":

        forecast = ses_forecast(
            train,
            horizon
        )

    elif model_name == "holt_linear":

        forecast = holt_forecast(
            train,
            horizon,
            damped=False
        )

    elif model_name == "holt_damped":

        forecast = holt_forecast(
            train,
            horizon,
            damped=True
        )

    elif model_name == "seasonal_naive_52":

        forecast = seasonal_naive_forecast(
            train,
            horizon,
            seasonal_period=52
        )

    elif model_name == "holt_winters_additive":

        forecast = holt_winters_additive_forecast(
            train,
            horizon
        )

    elif model_name == "croston_classic":

        forecast = croston_forecast(
            train,
            horizon,
            alpha=0.10,
            sba=False
        )

    elif model_name == "croston_sba":

        forecast = croston_forecast(
            train,
            horizon,
            alpha=0.10,
            sba=True
        )

    elif model_name == "recent_average_4":

        forecast = recent_average_forecast(
            train,
            horizon,
            4
        )

    elif model_name == "recent_average_8":

        forecast = recent_average_forecast(
            train,
            horizon,
            8
        )

    elif model_name == "weighted_recent_average":

        forecast = weighted_recent_average_forecast(
            train,
            horizon
        )

    elif model_name == "simple_trend":

        forecast = simple_trend_forecast(
            train,
            horizon
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return sanitize_forecast(
        forecast,
        horizon
    )


# =========================================================
# BACKTEST ORIGINS
# =========================================================

def build_backtest_origins(
    history_length,
    minimum_training,
    horizon,
    step
):

    origins = []

    origin = int(
        minimum_training
    )

    while (
        origin + horizon
        <= history_length
    ):

        origins.append(
            origin
        )

        origin += int(
            step
        )

    return origins


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    actual,
    forecast
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    forecast = np.asarray(
        forecast,
        dtype=float
    )

    error = (
        forecast
        -
        actual
    )

    absolute_error = np.abs(
        error
    )

    mae = float(
        np.mean(
            absolute_error
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    error
                )
            )
        )
    )

    total_actual = float(
        np.sum(
            actual
        )
    )

    total_forecast = float(
        np.sum(
            forecast
        )
    )

    absolute_error_sum = float(
        np.sum(
            absolute_error
        )
    )

    if total_actual > EPSILON:

        wape = (
            absolute_error_sum
            /
            total_actual
        )

        forecast_bias = (
            (
                total_forecast
                -
                total_actual
            )
            /
            total_actual
        )

    else:

        wape = np.nan

        forecast_bias = np.nan

    return {
        "mae":
            mae,

        "rmse":
            rmse,

        "wape":
            wape,

        "forecast_bias":
            forecast_bias,

        "actual_units":
            total_actual,

        "forecast_units":
            total_forecast,

        "absolute_error_units":
            absolute_error_sum,
    }


# =========================================================
# BACKTEST ONE SKU
# =========================================================

def backtest_sku(
    sku_history,
    registry_row
):

    sku = registry_row[
        "sku"
    ]

    demand_class = registry_row[
        "final_demand_class"
    ]

    candidate_models = (
        str(
            registry_row[
                "candidate_models"
            ]
        )
        .split("|")
    )

    minimum_training = int(
        registry_row[
            "minimum_training_weeks"
        ]
    )

    horizon = int(
        registry_row[
            "backtest_horizon_weeks"
        ]
    )

    step = int(
        registry_row[
            "backtest_step_weeks"
        ]
    )

    values = (
        sku_history[
            "forecast_target"
        ]
        .astype(float)
        .to_numpy()
    )

    dates = (
        sku_history[
            "week_date"
        ]
        .to_numpy()
    )

    origins = build_backtest_origins(
        len(values),
        minimum_training,
        horizon,
        step
    )

    rows = []

    for model_name in candidate_models:

        for window_number, origin in enumerate(
            origins,
            start=1
        ):

            train = values[
                :origin
            ]

            actual = values[
                origin:
                origin + horizon
            ]

            test_dates = dates[
                origin:
                origin + horizon
            ]

            try:

                forecast = forecast_model(
                    model_name,
                    train,
                    horizon
                )

                metrics = calculate_metrics(
                    actual,
                    forecast
                )

                success = True
                error_message = ""

            except Exception as exc:

                metrics = {
                    "mae":
                        np.nan,

                    "rmse":
                        np.nan,

                    "wape":
                        np.nan,

                    "forecast_bias":
                        np.nan,

                    "actual_units":
                        float(
                            np.sum(actual)
                        ),

                    "forecast_units":
                        np.nan,

                    "absolute_error_units":
                        np.nan,
                }

                success = False

                error_message = str(
                    exc
                )

            rows.append(
                {
                    "sku":
                        sku,

                    "final_demand_class":
                        demand_class,

                    "model":
                        model_name,

                    "window_number":
                        window_number,

                    "train_weeks":
                        len(train),

                    "test_horizon_weeks":
                        horizon,

                    "test_start_date":
                        test_dates[0],

                    "test_end_date":
                        test_dates[-1],

                    "success":
                        success,

                    "error_message":
                        error_message,

                    **metrics,
                }
            )

    return rows


# =========================================================
# RUN ALL BACKTESTS
# =========================================================

def run_backtests(
    history,
    registry
):

    print(
        "Running rolling-origin backtests..."
    )

    all_rows = []

    grouped_history = {
        sku: group.copy()
        for sku, group
        in history.groupby(
            "sku",
            sort=False
        )
    }

    total_skus = len(
        registry
    )

    for counter, (_, row) in enumerate(
        registry.iterrows(),
        start=1
    ):

        sku = row[
            "sku"
        ]

        eligible = str(
            row[
                "backtest_eligible"
            ]
        ).strip().lower() in [
            "true",
            "1",
            "yes",
        ]

        if not eligible:
            continue

        if sku not in grouped_history:
            continue

        rows = backtest_sku(
            grouped_history[
                sku
            ],
            row
        )

        all_rows.extend(
            rows
        )

        if (
            counter % 25 == 0
            or counter == total_skus
        ):

            print(
                f"  Processed "
                f"{counter}/{total_skus} SKUs"
            )

    results = pd.DataFrame(
        all_rows
    )

    if results.empty:

        raise ValueError(
            "No backtest results were generated"
        )

    print(
        f"✓ Backtest records generated: "
        f"{len(results):,}"
    )

    return results


# =========================================================
# AGGREGATE MODEL PERFORMANCE
# =========================================================

def aggregate_model_results(
    results
):

    print(
        "Aggregating model performance..."
    )

    successful = results[
        results[
            "success"
        ]
    ].copy()

    if successful.empty:

        raise ValueError(
            "All forecast backtests failed"
        )

    def aggregate_group(group):

        actual_units = group[
            "actual_units"
        ].sum()

        forecast_units = group[
            "forecast_units"
        ].sum()

        absolute_error = group[
            "absolute_error_units"
        ].sum()

        if actual_units > EPSILON:

            pooled_wape = (
                absolute_error
                /
                actual_units
            )

            pooled_bias = (
                (
                    forecast_units
                    -
                    actual_units
                )
                /
                actual_units
            )

        else:

            pooled_wape = np.nan
            pooled_bias = np.nan

        return pd.Series(
            {
                "successful_windows":
                    len(group),

                "actual_units":
                    actual_units,

                "forecast_units":
                    forecast_units,

                "wape":
                    pooled_wape,

                "mae":
                    group[
                        "mae"
                    ].mean(),

                "rmse":
                    group[
                        "rmse"
                    ].mean(),

                "forecast_bias":
                    pooled_bias,
            }
        )

    summary = (
        successful
        .groupby(
            [
                "sku",
                "final_demand_class",
                "model",
            ],
            as_index=False
        )
        .apply(
            aggregate_group,
            include_groups=False
        )
        .reset_index()
    )

    # Pandas versions can differ in how groupby.apply
    # returns the grouping keys. Normalize the output.
    if (
        "level_0"
        in summary.columns
    ):

        summary = summary.drop(
            columns=[
                "level_0"
            ]
        )

    return summary


# =========================================================
# MODEL COMPLEXITY
# =========================================================

def model_complexity_rank(
    model_name
):

    ranks = {
        "naive": 1,

        "moving_average_4": 2,
        "moving_average_13": 2,
        "recent_average_4": 2,
        "recent_average_8": 2,
        "weighted_recent_average": 2,

        "croston_classic": 3,
        "croston_sba": 3,
        "simple_exponential_smoothing": 3,
        "simple_trend": 3,

        "holt_linear": 4,
        "holt_damped": 4,
        "seasonal_naive_52": 4,

        "holt_winters_additive": 5,
    }

    return ranks.get(
        model_name,
        99
    )


# =========================================================
# SELECT MODEL FOR ONE SKU
# =========================================================

def select_model_for_sku(
    group
):

    group = group.copy()

    group[
        "model_complexity_rank"
    ] = group[
        "model"
    ].apply(
        model_complexity_rank
    )

    # -----------------------------------------------------
    # Primary metric: WAPE.
    #
    # If pooled actual demand is zero and WAPE is undefined,
    # fall back to MAE.
    # -----------------------------------------------------

    valid_wape = group[
        group[
            "wape"
        ].notna()
    ].copy()

    if not valid_wape.empty:

        best_row = (
            valid_wape
            .sort_values(
                [
                    "wape",
                    "mae",
                    "model_complexity_rank",
                ]
            )
            .iloc[0]
        )

        selection_metric = "wape"

    else:

        best_row = (
            group
            .sort_values(
                [
                    "mae",
                    "model_complexity_rank",
                ]
            )
            .iloc[0]
        )

        selection_metric = "mae"

    naive_rows = group[
        group[
            "model"
        ] == "naive"
    ]

    naive_wape = np.nan
    naive_mae = np.nan

    if not naive_rows.empty:

        naive_row = (
            naive_rows
            .iloc[0]
        )

        naive_wape = (
            naive_row[
                "wape"
            ]
        )

        naive_mae = (
            naive_row[
                "mae"
            ]
        )

    selected_row = best_row.copy()

    complexity_override = False
    improvement_vs_naive = np.nan

    # -----------------------------------------------------
    # Complexity safeguard.
    #
    # A complex model must improve WAPE by at least 3%
    # RELATIVE to naive. Otherwise we retain naive.
    # -----------------------------------------------------

    if (
        best_row[
            "model"
        ] != "naive"
        and pd.notna(
            naive_wape
        )
        and naive_wape > EPSILON
        and pd.notna(
            best_row[
                "wape"
            ]
        )
    ):

        improvement_vs_naive = (
            naive_wape
            -
            best_row[
                "wape"
            ]
        ) / naive_wape

        if (
            improvement_vs_naive
            <
            MIN_IMPROVEMENT_OVER_NAIVE
        ):

            selected_row = (
                naive_rows
                .iloc[0]
                .copy()
            )

            complexity_override = True

    elif (
        best_row[
            "model"
        ] == "naive"
    ):

        improvement_vs_naive = 0.0

    # -----------------------------------------------------
    # If WAPE was unavailable, compare MAE.
    # -----------------------------------------------------

    if (
        selection_metric == "mae"
        and best_row[
            "model"
        ] != "naive"
        and pd.notna(
            naive_mae
        )
        and naive_mae > EPSILON
    ):

        improvement_vs_naive = (
            naive_mae
            -
            best_row[
                "mae"
            ]
        ) / naive_mae

        if (
            improvement_vs_naive
            <
            MIN_IMPROVEMENT_OVER_NAIVE
        ):

            selected_row = (
                naive_rows
                .iloc[0]
                .copy()
            )

            complexity_override = True

    return pd.Series(
        {
            "selected_model":
                selected_row[
                    "model"
                ],

            "selection_metric":
                selection_metric,

            "selected_wape":
                selected_row[
                    "wape"
                ],

            "selected_mae":
                selected_row[
                    "mae"
                ],

            "selected_rmse":
                selected_row[
                    "rmse"
                ],

            "selected_bias":
                selected_row[
                    "forecast_bias"
                ],

            "successful_backtest_windows":
                int(
                    selected_row[
                        "successful_windows"
                    ]
                ),

            "naive_wape":
                naive_wape,

            "naive_mae":
                naive_mae,

            "improvement_vs_naive":
                improvement_vs_naive,

            "complexity_override_to_naive":
                complexity_override,
        }
    )


# =========================================================
# BUILD MODEL SELECTION
# =========================================================

def build_model_selection(
    model_summary,
    registry
):

    print(
        "Selecting winning model per SKU..."
    )

    selections = (
        model_summary
        .groupby(
            [
                "sku",
                "final_demand_class",
            ]
        )
        .apply(
            select_model_for_sku,
            include_groups=False
        )
        .reset_index()
    )

    registry_columns = [
        "sku",
        "forecast_readiness",
        "forecast_difficulty",
        "classification_confidence",
        "candidate_models",
        "estimated_backtest_windows",
    ]

    selections = selections.merge(
        registry[
            registry_columns
        ],
        on="sku",
        how="left",
        validate="one_to_one"
    )

    return selections


# =========================================================
# VALIDATION
# =========================================================

def validate_outputs(
    raw_results,
    model_summary,
    selections,
    registry
):

    print("\n" + "=" * 90)
    print("FORECAST BACKTEST VALIDATION")
    print("=" * 90)

    eligible_registry = registry[
        normalize_boolean(
            registry[
                "backtest_eligible"
            ]
        )
    ].copy()

    assert not raw_results.empty

    print(
        "✓ Backtest results generated"
    )

    assert (
        raw_results[
            "test_horizon_weeks"
        ] > 0
    ).all()

    print(
        "✓ Backtest horizons are positive"
    )

    assert (
        raw_results[
            "train_weeks"
        ] > 0
    ).all()

    print(
        "✓ Training windows are positive"
    )

    successful = raw_results[
        raw_results[
            "success"
        ]
    ]

    assert not successful.empty

    print(
        f"✓ Successful backtest records: "
        f"{len(successful):,}"
    )

    failed = raw_results[
        ~raw_results[
            "success"
        ]
    ]

    print(
        f"✓ Failed model-window attempts captured: "
        f"{len(failed):,}"
    )

    assert (
        successful[
            "mae"
        ] >= 0
    ).all()

    assert (
        successful[
            "rmse"
        ] >= 0
    ).all()

    print(
        "✓ Forecast error metrics are non-negative"
    )

    valid_wape = successful[
        "wape"
    ].dropna()

    assert (
        valid_wape >= 0
    ).all()

    print(
        "✓ WAPE values are valid"
    )

    assert not model_summary.empty

    print(
        "✓ Model performance summary generated"
    )

    assert selections[
        "sku"
    ].is_unique

    print(
        "✓ One selected model per backtested SKU"
    )

    assert set(
        selections[
            "sku"
        ]
    ).issubset(
        set(
            eligible_registry[
                "sku"
            ]
        )
    )

    print(
        "✓ Selected SKUs belong to "
        "backtest-eligible population"
    )

    # -----------------------------------------------------
    # Every selected model must have been one of the
    # candidates assigned in Phase 6A.
    # -----------------------------------------------------

    registry_lookup = (
        registry
        .set_index(
            "sku"
        )[
            "candidate_models"
        ]
        .to_dict()
    )

    for _, row in selections.iterrows():

        candidates = (
            str(
                registry_lookup[
                    row[
                        "sku"
                    ]
                ]
            )
            .split("|")
        )

        assert (
            row[
                "selected_model"
            ]
            in candidates
        )

    print(
        "✓ Every selected model came from "
        "its assigned candidate set"
    )

    # -----------------------------------------------------
    # Intermittent model validation
    # -----------------------------------------------------

    intermittent = selections[
        selections[
            "final_demand_class"
        ]
        == "intermittent"
    ]

    assert len(
        intermittent
    ) > 0

    print(
        f"✓ Intermittent SKUs evaluated: "
        f"{len(intermittent)}"
    )

    # -----------------------------------------------------
    # Seasonal model validation
    # -----------------------------------------------------

    seasonal_results = model_summary[
        model_summary[
            "final_demand_class"
        ]
        == "seasonal"
    ]

    assert (
        seasonal_results[
            "model"
        ]
        == "seasonal_naive_52"
    ).any()

    print(
        "✓ Seasonal-naive model evaluated"
    )

    # -----------------------------------------------------
    # Complexity safeguard validation
    # -----------------------------------------------------

    overrides = selections[
        selections[
            "complexity_override_to_naive"
        ]
    ]

    print(
        f"✓ Complexity safeguards triggered: "
        f"{len(overrides)}"
    )

    print(
        "\nAll Phase 6B validation "
        "checks passed."
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================
    # Show failed backtest attempts
    failed_attempts = raw_results[
        ~raw_results["success"]
    ].copy()

    print("\n" + "=" * 90)
    print("FAILED BACKTEST DIAGNOSTICS")
    print("=" * 90)

    if failed_attempts.empty:
        print("No failed backtest attempts.")
    else:
        print(
            failed_attempts[
                [
                    "final_demand_class",
                    "model",
                    "error_message",
                ]
            ]
            .value_counts()
            .reset_index(name="failure_count")
            .to_string(index=False)
        )
        
def save_outputs(
    model_summary,
    selections
):

    model_summary.to_csv(
        BACKTEST_RESULTS_FILE,
        index=False
    )

    selections.to_csv(
        MODEL_SELECTION_FILE,
        index=False
    )

    print(
        f"\nCreated: "
        f"{BACKTEST_RESULTS_FILE}"
    )

    print(
        f"Created: "
        f"{MODEL_SELECTION_FILE}"
    )


# =========================================================
# PRINT MODEL WINNERS
# =========================================================

def print_model_winner_summary(
    selections
):

    print("\n" + "=" * 110)
    print("SELECTED MODEL DISTRIBUTION")
    print("=" * 110)

    summary = (
        selections
        .groupby(
            [
                "final_demand_class",
                "selected_model",
            ]
        )
        .agg(
            sku_count=(
                "sku",
                "count"
            ),

            avg_wape=(
                "selected_wape",
                "mean"
            ),

            avg_mae=(
                "selected_mae",
                "mean"
            ),

            avg_bias=(
                "selected_bias",
                "mean"
            ),
        )
        .round(3)
    )

    print(summary)


# =========================================================
# PRINT OVERALL PERFORMANCE
# =========================================================

def print_overall_performance(
    selections
):

    print("\n" + "=" * 90)
    print("FORECAST MODEL SELECTION SUMMARY")
    print("=" * 90)

    print(
        f"SKUs with selected models: "
        f"{len(selections):,}"
    )

    print(
        f"Average selected WAPE: "
        f"{selections['selected_wape'].mean():.2%}"
    )

    print(
        f"Median selected WAPE: "
        f"{selections['selected_wape'].median():.2%}"
    )

    print(
        f"Average absolute bias: "
        f"{selections['selected_bias'].abs().mean():.2%}"
    )

    improved = selections[
        selections[
            "improvement_vs_naive"
        ] > 0
    ]

    print(
        f"SKUs where best candidate beat naive: "
        f"{len(improved):,}"
    )

    overrides = selections[
        selections[
            "complexity_override_to_naive"
        ]
    ]

    print(
        f"Complexity overrides back to naive: "
        f"{len(overrides):,}"
    )


# =========================================================
# PRINT BEST / HARDEST FORECASTS
# =========================================================

def print_examples(
    selections
):

    print("\n" + "=" * 130)
    print("EXAMPLE MODEL SELECTIONS")
    print("=" * 130)

    columns = [
        "sku",
        "final_demand_class",
        "selected_model",
        "selected_wape",
        "selected_mae",
        "selected_bias",
        "naive_wape",
        "improvement_vs_naive",
        "successful_backtest_windows",
    ]

    examples = (
        selections
        .sort_values(
            "selected_wape",
            na_position="last"
        )
    )

    print(
        examples[
            columns
        ]
        .head(15)
        .round(3)
        .to_string(
            index=False
        )
    )

    print("\n" + "-" * 130)
    print("HIGHEST FORECAST ERROR SKUs")
    print("-" * 130)

    print(
        examples[
            columns
        ]
        .tail(15)
        .sort_values(
            "selected_wape",
            ascending=False,
            na_position="last"
        )
        .round(3)
        .to_string(
            index=False
        )
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 90)

    print(
        "PROCUREMENT AI - "
        "ROLLING FORECAST BACKTESTING"
    )

    print("=" * 90)

    data = load_data()

    validate_inputs(
        data
    )

    history = prepare_history(
        data[
            "forecasting_dataset"
        ]
    )

    registry = data[
        "model_registry"
    ].copy()

    raw_results = run_backtests(
        history,
        registry
    )

    model_summary = (
        aggregate_model_results(
            raw_results
        )
    )

    selections = (
        build_model_selection(
            model_summary,
            registry
        )
    )

    validate_outputs(
        raw_results,
        model_summary,
        selections,
        registry
    )

    save_outputs(
        model_summary,
        selections
    )

    print_overall_performance(
        selections
    )

    print_model_winner_summary(
        selections
    )

    print_examples(
        selections
    )

    print("\n" + "=" * 90)

    print(
        "PHASE 6B COMPLETE"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()