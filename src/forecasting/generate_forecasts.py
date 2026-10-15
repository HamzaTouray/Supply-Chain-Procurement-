from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FORECASTING_DATASET_PATH = (
    PROCESSED_DIR / "forecasting_dataset.csv"
)

MODEL_REGISTRY_PATH = (
    PROCESSED_DIR / "forecast_model_registry.csv"
)

MODEL_SELECTION_PATH = (
    PROCESSED_DIR / "forecast_model_selection.csv"
)

WEEKLY_FORECAST_PATH = (
    PROCESSED_DIR / "production_forecasts.csv"
)

FORECAST_SUMMARY_PATH = (
    PROCESSED_DIR / "production_forecast_summary.csv"
)


# =========================================================
# IMPORT BACKTEST FORECAST ENGINE
# =========================================================

FORECASTING_SRC = PROJECT_ROOT / "src" / "forecasting"

if str(FORECASTING_SRC) not in sys.path:
    sys.path.insert(0, str(FORECASTING_SRC))

from backtest_models import forecast_model


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_FORECAST_HORIZON = 26

VALID_FORECAST_HORIZONS = {
    4,
    13,
    26,
    52,
}

FALLBACK_MODEL = "weighted_recent_average"


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
        "PROCUREMENT AI - PRODUCTION FORECAST GENERATOR"
    )

    print("\nLoading Phase 6C inputs...")

    forecasting_dataset = pd.read_csv(
        FORECASTING_DATASET_PATH
    )

    model_registry = pd.read_csv(
        MODEL_REGISTRY_PATH
    )

    model_selection = pd.read_csv(
        MODEL_SELECTION_PATH
    )

    check(
        f"forecasting_dataset     "
        f"{len(forecasting_dataset):>8,} rows"
    )

    check(
        f"forecast_model_registry "
        f"{len(model_registry):>8,} rows"
    )

    check(
        f"forecast_model_selection"
        f"{len(model_selection):>8,} rows"
    )

    return {
        "forecasting_dataset": forecasting_dataset,
        "model_registry": model_registry,
        "model_selection": model_selection,
    }


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_inputs(data):

    print("Validating production-forecast inputs...")

    forecasting_dataset = data[
        "forecasting_dataset"
    ]

    registry = data[
        "model_registry"
    ]

    selection = data[
        "model_selection"
    ]

    required_forecast_columns = {
        "week_date",
        "sku",
        "forecast_eligible",
        "forecast_target",
        "is_prelaunch",
    }

    required_registry_columns = {
        "sku",
        "product_name",
        "category",
        "supplier_id",
        "final_demand_class",
        "forecast_readiness",
        "forecast_difficulty",
        "classification_confidence",
        "history_weeks",
        "primary_forecast_horizon_weeks",
        "candidate_models",
        "backtest_eligible",
    }

    required_selection_columns = {
        "sku",
        "selected_model",
        "selected_wape",
        "selected_mae",
        "selected_rmse",
        "selected_bias",
        "successful_backtest_windows",
        "improvement_vs_naive",
    }

    missing_forecast = (
        required_forecast_columns
        - set(forecasting_dataset.columns)
    )

    missing_registry = (
        required_registry_columns
        - set(registry.columns)
    )

    missing_selection = (
        required_selection_columns
        - set(selection.columns)
    )

    if missing_forecast:
        raise ValueError(
            "forecasting_dataset.csv missing columns: "
            + ", ".join(sorted(missing_forecast))
        )

    if missing_registry:
        raise ValueError(
            "forecast_model_registry.csv missing columns: "
            + ", ".join(sorted(missing_registry))
        )

    if missing_selection:
        raise ValueError(
            "forecast_model_selection.csv missing columns: "
            + ", ".join(sorted(missing_selection))
        )

    if registry["sku"].duplicated().any():
        raise ValueError(
            "forecast_model_registry.csv contains duplicate SKUs"
        )

    if selection["sku"].duplicated().any():
        raise ValueError(
            "forecast_model_selection.csv contains duplicate SKUs"
        )

    if len(registry) != 200:
        raise ValueError(
            f"Expected 200 registry SKUs, found {len(registry)}"
        )

    check("Required production-forecast columns present")
    check("Forecast registry contains one row per SKU")
    check("Model selection contains one row per selected SKU")


# =========================================================
# BOOLEAN NORMALIZATION
# =========================================================

def normalize_boolean(series):

    if pd.api.types.is_bool_dtype(series):
        return series

    normalized = (
        series
        .astype(str)
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
    )

    return normalized.fillna(False).astype(bool)


# =========================================================
# PREPARE FORECAST HISTORY
# =========================================================

def prepare_history(forecasting_dataset):

    print("Preparing full production forecasting history...")

    df = forecasting_dataset.copy()

    df["week_date"] = pd.to_datetime(
        df["week_date"]
    )

    df["forecast_eligible"] = normalize_boolean(
        df["forecast_eligible"]
    )

    df["is_prelaunch"] = normalize_boolean(
        df["is_prelaunch"]
    )

    df["forecast_target"] = pd.to_numeric(
        df["forecast_target"],
        errors="coerce"
    )

    df = df.sort_values(
        ["sku", "week_date"]
    ).reset_index(drop=True)

    eligible = df[
        df["forecast_eligible"]
    ].copy()

    if eligible["forecast_target"].isna().any():

        bad_rows = eligible[
            eligible["forecast_target"].isna()
        ]

        raise ValueError(
            "Forecast-eligible rows contain missing "
            f"forecast targets: {len(bad_rows)} rows"
        )

    if (
        eligible["forecast_target"] < 0
    ).any():

        raise ValueError(
            "Negative forecast targets detected"
        )

    check(
        f"Usable demand observations: "
        f"{len(eligible):,}"
    )

    check(
        f"SKUs with usable history: "
        f"{eligible['sku'].nunique():,}"
    )

    return df


# =========================================================
# BUILD PRODUCTION MODEL REGISTRY
# =========================================================

def build_production_registry(
    registry,
    selection
):

    print("Building production model registry...")

    result = registry.copy()

    selection_columns = [
        "sku",
        "selected_model",
        "selected_wape",
        "selected_mae",
        "selected_rmse",
        "selected_bias",
        "successful_backtest_windows",
        "improvement_vs_naive",
    ]

    result = result.merge(
        selection[selection_columns],
        on="sku",
        how="left",
        validate="one_to_one"
    )

    result[
        "champion_model_available"
    ] = result[
        "selected_model"
    ].notna()

    result[
        "production_model"
    ] = result[
        "selected_model"
    ]

    result[
        "model_source"
    ] = np.where(
        result["champion_model_available"],
        "backtest_champion",
        "fallback"
    )

    # For SKUs that did not have enough history for a
    # valid rolling backtest, use a controlled fallback.
    fallback_mask = (
        result["production_model"].isna()
    )

    result.loc[
        fallback_mask,
        "production_model"
    ] = FALLBACK_MODEL

    result[
        "production_forecast_horizon_weeks"
    ] = DEFAULT_FORECAST_HORIZON

    result[
        "production_forecast_horizon_weeks"
    ] = pd.to_numeric(
        result[
            "production_forecast_horizon_weeks"
        ],
        errors="coerce"
    ).fillna(
        DEFAULT_FORECAST_HORIZON
    ).astype(int)

    fallback_count = int(
        fallback_mask.sum()
    )

    check(
        f"Champion models assigned: "
        f"{len(result) - fallback_count:,}"
    )

    check(
        f"Fallback models assigned: "
        f"{fallback_count:,}"
    )

    return result


# =========================================================
# FORECAST DATE GENERATION
# =========================================================

def build_future_dates(
    last_history_date,
    horizon
):

    last_history_date = pd.Timestamp(
        last_history_date
    )

    # Historical data is weekly and uses Monday dates.
    # Continue the exact weekly cadence.
    return pd.date_range(
        start=last_history_date
        + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="7D"
    )


# =========================================================
# GENERATE ONE SKU FORECAST
# =========================================================

def forecast_one_sku(
    sku,
    sku_history,
    registry_row
):

    history = (
        sku_history[
            sku_history["forecast_eligible"]
        ]
        .sort_values("week_date")
        .copy()
    )

    if history.empty:
        raise ValueError(
            f"{sku}: no forecast-eligible history"
        )

    train = (
        history["forecast_target"]
        .astype(float)
        .to_numpy()
    )

    model_name = registry_row[
        "production_model"
    ]

    horizon = int(
        registry_row[
            "production_forecast_horizon_weeks"
        ]
    )

    forecast_values = forecast_model(
        model_name,
        train,
        horizon
    )

    forecast_values = np.asarray(
        forecast_values,
        dtype=float
    )

    if len(forecast_values) != horizon:
        raise ValueError(
            f"{sku}: model {model_name} returned "
            f"{len(forecast_values)} periods; "
            f"expected {horizon}"
        )

    if not np.isfinite(
        forecast_values
    ).all():
        raise ValueError(
            f"{sku}: non-finite production forecast"
        )

    if (
        forecast_values < 0
    ).any():
        raise ValueError(
            f"{sku}: negative production forecast"
        )

    last_history_date = history[
        "week_date"
    ].max()

    future_dates = build_future_dates(
        last_history_date,
        horizon
    )

    forecast_df = pd.DataFrame(
        {
            "sku": sku,
            "forecast_week": future_dates,
            "forecast_horizon_week": np.arange(
                1,
                horizon + 1
            ),
            "forecast_demand": forecast_values,
        }
    )

    forecast_df[
        "production_model"
    ] = model_name

    forecast_df[
        "model_source"
    ] = registry_row[
        "model_source"
    ]

    forecast_df[
        "final_demand_class"
    ] = registry_row[
        "final_demand_class"
    ]

    forecast_df[
        "forecast_readiness"
    ] = registry_row[
        "forecast_readiness"
    ]

    forecast_df[
        "forecast_difficulty"
    ] = registry_row[
        "forecast_difficulty"
    ]

    forecast_df[
        "classification_confidence"
    ] = registry_row[
        "classification_confidence"
    ]

    forecast_df[
        "selected_wape"
    ] = registry_row.get(
        "selected_wape",
        np.nan
    )

    forecast_df[
        "selected_bias"
    ] = registry_row.get(
        "selected_bias",
        np.nan
    )

    return forecast_df


# =========================================================
# GENERATE ALL FORECASTS
# =========================================================

def generate_production_forecasts(
    forecasting_dataset,
    production_registry
):

    print("Generating forward production forecasts...")

    forecasts = []

    failures = []

    grouped_history = {
        sku: group.copy()
        for sku, group
        in forecasting_dataset.groupby("sku")
    }

    for _, row in production_registry.iterrows():

        sku = row["sku"]

        try:

            if sku not in grouped_history:
                raise ValueError(
                    "SKU missing from forecasting dataset"
                )

            sku_forecast = forecast_one_sku(
                sku,
                grouped_history[sku],
                row
            )

            forecasts.append(
                sku_forecast
            )

        except Exception as exc:

            failures.append(
                {
                    "sku": sku,
                    "production_model": row[
                        "production_model"
                    ],
                    "error": str(exc),
                }
            )

    if failures:

        failure_df = pd.DataFrame(
            failures
        )

        print("\nForecast failures:")
        print(
            failure_df.to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{len(failures)} SKU forecasts failed"
        )

    forecast_df = pd.concat(
        forecasts,
        ignore_index=True
    )

    check(
        f"Production forecasts generated: "
        f"{len(forecast_df):,} rows"
    )

    check(
        f"SKUs forecasted: "
        f"{forecast_df['sku'].nunique():,}"
    )

    return forecast_df


# =========================================================
# BUILD FORECAST SUMMARY
# =========================================================

def build_forecast_summary(
    forecasts,
    production_registry
):

    print("Building production forecast summary...")

    forecast_summary = (
        forecasts
        .groupby("sku", as_index=False)
        .agg(
            forecast_4w_units=(
                "forecast_demand",
                lambda x: float(
                    x.iloc[:4].sum()
                )
            ),
            forecast_13w_units=(
                "forecast_demand",
                lambda x: float(
                    x.iloc[:13].sum()
                )
            ),
            forecast_26w_units=(
                "forecast_demand",
                lambda x: float(
                    x.iloc[:26].sum()
                )
            ),
            avg_weekly_forecast=(
                "forecast_demand",
                "mean"
            ),
            min_weekly_forecast=(
                "forecast_demand",
                "min"
            ),
            max_weekly_forecast=(
                "forecast_demand",
                "max"
            ),
        )
    )

    metadata_columns = [
        "sku",
        "product_name",
        "category",
        "supplier_id",
        "final_demand_class",
        "classification_confidence",
        "forecast_readiness",
        "forecast_difficulty",
        "production_model",
        "model_source",
        "history_weeks",
        "selected_wape",
        "selected_mae",
        "selected_rmse",
        "selected_bias",
        "successful_backtest_windows",
        "improvement_vs_naive",
    ]

    summary = production_registry[
        metadata_columns
    ].merge(
        forecast_summary,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    return summary


# =========================================================
# VALIDATION
# =========================================================

def validate_outputs(
    forecasts,
    summary,
    production_registry
):

    section(
        "PRODUCTION FORECAST VALIDATION"
    )

    expected_skus = set(
        production_registry["sku"]
    )

    forecast_skus = set(
        forecasts["sku"]
    )

    summary_skus = set(
        summary["sku"]
    )

    assert len(
        production_registry
    ) == 200

    check(
        "Exactly 200 production registry rows"
    )

    assert (
        production_registry["sku"]
        .nunique()
        == 200
    )

    check(
        "One production registry row per SKU"
    )

    assert forecast_skus == expected_skus

    check(
        "All registry SKUs received forecasts"
    )

    assert summary_skus == expected_skus

    check(
        "All registry SKUs present in forecast summary"
    )

    assert (
        forecasts["forecast_demand"]
        .notna()
        .all()
    )

    check(
        "No missing forecast quantities"
    )

    assert (
        forecasts["forecast_demand"]
        >= 0
    ).all()

    check(
        "Forecast quantities are non-negative"
    )

    assert np.isfinite(
        forecasts["forecast_demand"]
    ).all()

    check(
        "Forecast quantities are finite"
    )

    expected_rows = int(
        production_registry[
            "production_forecast_horizon_weeks"
        ].sum()
    )

    assert len(
        forecasts
    ) == expected_rows

    check(
        f"Forecast row count correct: "
        f"{expected_rows:,}"
    )

    horizon_counts = (
        forecasts
        .groupby("sku")
        .size()
    )

    expected_horizons = (
        production_registry
        .set_index("sku")[
            "production_forecast_horizon_weeks"
        ]
    )

    horizon_counts = horizon_counts.reindex(
        expected_horizons.index
    )

    assert (
        horizon_counts.values
        == expected_horizons.values
    ).all()

    check(
        "Every SKU has the correct forecast horizon"
    )

    fallback_count = int(
        (
            production_registry[
                "model_source"
            ]
            == "fallback"
        ).sum()
    )

    champion_count = int(
        (
            production_registry[
                "model_source"
            ]
            == "backtest_champion"
        ).sum()
    )

    assert (
        fallback_count
        + champion_count
        == 200
    )

    check(
        f"Champion/fallback model routing valid: "
        f"{champion_count} champion, "
        f"{fallback_count} fallback"
    )

    assert (
        summary["forecast_4w_units"]
        <= summary["forecast_13w_units"]
        + 1e-9
    ).all()

    assert (
        summary["forecast_13w_units"]
        <= summary["forecast_26w_units"]
        + 1e-9
    ).all()

    check(
        "Forecast cumulative horizons reconcile"
    )

    print(
        "\nAll Phase 6C validation checks passed."
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================

def save_outputs(
    forecasts,
    summary
):

    section(
        "SAVING PRODUCTION FORECASTS"
    )

    output_forecasts = forecasts.copy()

    output_forecasts[
        "forecast_week"
    ] = pd.to_datetime(
        output_forecasts[
            "forecast_week"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
    )

    output_forecasts.to_csv(
        WEEKLY_FORECAST_PATH,
        index=False
    )

    summary.to_csv(
        FORECAST_SUMMARY_PATH,
        index=False
    )

    check(
        WEEKLY_FORECAST_PATH.name
    )

    check(
        FORECAST_SUMMARY_PATH.name
    )


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    forecasts,
    summary
):

    section(
        "PRODUCTION FORECAST SUMMARY"
    )

    print(
        f"\nSKUs forecasted: "
        f"{summary['sku'].nunique():,}"
    )

    print(
        f"Weekly forecast rows: "
        f"{len(forecasts):,}"
    )

    print(
        f"Total 4-week forecast: "
        f"{summary['forecast_4w_units'].sum():,.1f}"
    )

    print(
        f"Total 13-week forecast: "
        f"{summary['forecast_13w_units'].sum():,.1f}"
    )

    print(
        f"Total 26-week forecast: "
        f"{summary['forecast_26w_units'].sum():,.1f}"
    )

    print(
        "\nProduction model source:"
    )

    print(
        summary[
            "model_source"
        ].value_counts()
    )

    print(
        "\nProduction model distribution:"
    )

    print(
        summary[
            "production_model"
        ].value_counts()
    )

    section(
        "FORECAST BY DEMAND CLASS",
        width=110
    )

    class_summary = (
        summary
        .groupby(
            "final_demand_class"
        )
        .agg(
            sku_count=(
                "sku",
                "count"
            ),
            forecast_4w_units=(
                "forecast_4w_units",
                "sum"
            ),
            forecast_13w_units=(
                "forecast_13w_units",
                "sum"
            ),
            forecast_26w_units=(
                "forecast_26w_units",
                "sum"
            ),
            avg_weekly_forecast=(
                "avg_weekly_forecast",
                "mean"
            ),
        )
        .round(2)
    )

    print(
        class_summary.to_string()
    )

    section(
        "EXAMPLE PRODUCTION FORECASTS",
        width=130
    )

    example_columns = [
        "sku",
        "product_name",
        "final_demand_class",
        "production_model",
        "model_source",
        "history_weeks",
        "selected_wape",
        "forecast_4w_units",
        "forecast_13w_units",
        "forecast_26w_units",
    ]

    examples = (
        summary[
            example_columns
        ]
        .sort_values(
            [
                "final_demand_class",
                "sku",
            ]
        )
        .groupby(
            "final_demand_class",
            group_keys=False
        )
        .head(2)
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

    data = load_data()

    validate_inputs(
        data
    )

    forecasting_dataset = prepare_history(
        data["forecasting_dataset"]
    )

    production_registry = (
        build_production_registry(
            data["model_registry"],
            data["model_selection"]
        )
    )

    forecasts = (
        generate_production_forecasts(
            forecasting_dataset,
            production_registry
        )
    )

    summary = build_forecast_summary(
        forecasts,
        production_registry
    )

    validate_outputs(
        forecasts,
        summary,
        production_registry
    )

    save_outputs(
        forecasts,
        summary
    )

    print_summary(
        forecasts,
        summary
    )

    section(
        "PHASE 6C COMPLETE"
    )


if __name__ == "__main__":
    main()