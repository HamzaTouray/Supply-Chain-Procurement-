from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

SNAPSHOT_DATE = pd.Timestamp("2026-09-30")

# Weekly data: annual seasonality is approximately 52 weeks.
SEASONAL_LAG = 52

# Minimum history required to compare one annual cycle
# against another.
MIN_SEASONAL_HISTORY = 90


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

DEMAND_FEATURE_FILE = (
    PROCESSED_DATA_DIR
    / "demand_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "demand_behavior.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\nLoading Phase 4B inputs...")

    files = {
        "sales": SALES_FILE,
        "features": DEMAND_FEATURE_FILE,
    }

    data = {}

    for name, file_path in files.items():

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing required file: {file_path}"
            )

        data[name] = pd.read_csv(
            file_path
        )

        print(
            f"✓ {name:<20} "
            f"{len(data[name]):>7,} rows"
        )

    data["sales"]["week_date"] = (
        pd.to_datetime(
            data["sales"]["week_date"]
        )
    )

    return data


# =========================================================
# PREPARE DEMAND
# =========================================================

def prepare_demand(
    sales_df
):

    print("\nPreparing weekly demand...")

    sales = sales_df.copy()

    sales[
        "units_demanded"
    ] = pd.to_numeric(
        sales["units_demanded"],
        errors="coerce"
    )

    # Pre-launch NaNs are not zero demand.
    demand = sales[
        sales["units_demanded"].notna()
    ].copy()

    demand = demand.sort_values(
        [
            "sku",
            "week_date",
        ]
    )

    print(
        f"✓ Demand observations: "
        f"{len(demand):,}"
    )

    print(
        f"✓ SKUs: "
        f"{demand['sku'].nunique():,}"
    )

    return demand


# =========================================================
# SAFE CORRELATION
# =========================================================

def safe_correlation(
    x,
    y
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    if len(x) < 4 or len(y) < 4:

        return np.nan

    if (
        np.std(x) == 0
        or np.std(y) == 0
    ):

        return 0.0

    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


# =========================================================
# ANNUAL AUTOCORRELATION
# =========================================================

def calculate_annual_autocorrelation(
    values,
    lag=SEASONAL_LAG
):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) <= lag + 3:

        return np.nan

    earlier = values[:-lag]
    later = values[lag:]

    return safe_correlation(
        earlier,
        later
    )


# =========================================================
# HALF-CYCLE AUTOCORRELATION
# =========================================================

def calculate_half_cycle_autocorrelation(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    lag = 26

    if len(values) <= lag + 3:

        return np.nan

    return safe_correlation(
        values[:-lag],
        values[lag:]
    )


# =========================================================
# QUARTER-CYCLE AUTOCORRELATION
# =========================================================

def calculate_quarter_cycle_autocorrelation(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    lag = 13

    if len(values) <= lag + 3:

        return np.nan

    return safe_correlation(
        values[:-lag],
        values[lag:]
    )


# =========================================================
# SEASONAL STRENGTH
# =========================================================

def calculate_seasonal_strength(
    values
):

    """
    Compare week-to-week variation with variation
    after subtracting the value from 52 weeks ago.

    If the same annual pattern repeats, the
    year-over-year residual variation should be
    much smaller than the raw demand variation.

    Higher values indicate stronger annual
    repeating structure.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) <= SEASONAL_LAG:

        return np.nan

    raw_variance = np.var(
        values,
        ddof=1
    )

    if raw_variance <= 0:

        return 0.0

    seasonal_difference = (
        values[SEASONAL_LAG:]
        -
        values[:-SEASONAL_LAG]
    )

    residual_variance = np.var(
        seasonal_difference,
        ddof=1
    )

    strength = (
        1
        -
        (
            residual_variance
            /
            (2 * raw_variance)
        )
    )

    return float(
        np.clip(
            strength,
            0,
            1
        )
    )


# =========================================================
# SEASONAL AMPLITUDE
# =========================================================

def calculate_seasonal_amplitude(
    values
):

    """
    Measures how large the recurring movement is
    relative to average demand.

    We smooth demand using 4-week blocks to reduce
    random weekly noise.
    """

    series = pd.Series(
        values,
        dtype=float
    )

    if len(series) < 26:

        return np.nan

    rolling = (
        series
        .rolling(
            window=4,
            center=True,
            min_periods=2
        )
        .mean()
    )

    mean_demand = (
        series.mean()
    )

    if mean_demand <= 0:

        return np.nan

    high = rolling.quantile(
        0.90
    )

    low = rolling.quantile(
        0.10
    )

    amplitude = (
        high - low
    ) / mean_demand

    return float(
        max(
            amplitude,
            0
        )
    )


# =========================================================
# YEAR-OVER-YEAR SIMILARITY
# =========================================================

def calculate_yoy_similarity(
    values
):

    """
    Compare the most recent annual pattern with
    the previous annual pattern.

    This gives us another direct measure of whether
    peaks and valleys repeat at similar times.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 90:

        return np.nan

    if len(values) >= 104:

        previous = values[-104:-52]
        current = values[-52:]

    else:

        overlap = (
            len(values) - 52
        )

        previous = values[
            -(52 + overlap):-52
        ]

        current = values[
            -overlap:
        ]

    return safe_correlation(
        previous,
        current
    )


# =========================================================
# DETREND SERIES
# =========================================================

def detrend_series(
    values
):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) < 4:

        return values

    x = np.arange(
        len(values),
        dtype=float
    )

    slope, intercept = (
        np.polyfit(
            x,
            values,
            1
        )
    )

    trend = (
        slope * x
        + intercept
    )

    return (
        values
        -
        trend
        +
        np.mean(values)
    )


# =========================================================
# SEASONAL FEATURES
# =========================================================

def calculate_seasonality_features(
    demand_df
):

    print(
        "Calculating seasonality features..."
    )

    records = []

    for sku, group in (
        demand_df.groupby("sku")
    ):

        group = group.sort_values(
            "week_date"
        )

        values = (
            group[
                "units_demanded"
            ]
            .astype(float)
            .to_numpy()
        )

        history_weeks = len(
            values
        )

        # -------------------------------------------------
        # Raw seasonal measurements
        # -------------------------------------------------

        annual_autocorr = (
            calculate_annual_autocorrelation(
                values
            )
        )

        yoy_similarity = (
            calculate_yoy_similarity(
                values
            )
        )

        seasonal_strength = (
            calculate_seasonal_strength(
                values
            )
        )

        seasonal_amplitude = (
            calculate_seasonal_amplitude(
                values
            )
        )

        half_cycle_corr = (
            calculate_half_cycle_autocorrelation(
                values
            )
        )

        quarter_cycle_corr = (
            calculate_quarter_cycle_autocorrelation(
                values
            )
        )

        # -------------------------------------------------
        # Detrended measurements
        #
        # Important:
        # Growth/decline can artificially create
        # correlations. Removing linear trend helps
        # isolate recurring seasonal structure.
        # -------------------------------------------------

        detrended = (
            detrend_series(
                values
            )
        )

        detrended_annual_autocorr = (
            calculate_annual_autocorrelation(
                detrended
            )
        )

        detrended_yoy_similarity = (
            calculate_yoy_similarity(
                detrended
            )
        )

        records.append(
            {
                "sku": sku,

                "seasonal_history_weeks": (
                    history_weeks
                ),

                "annual_autocorrelation": (
                    annual_autocorr
                ),

                "detrended_annual_autocorrelation": (
                    detrended_annual_autocorr
                ),

                "yoy_pattern_similarity": (
                    yoy_similarity
                ),

                "detrended_yoy_similarity": (
                    detrended_yoy_similarity
                ),

                "seasonal_strength": (
                    seasonal_strength
                ),

                "seasonal_amplitude": (
                    seasonal_amplitude
                ),

                "half_cycle_autocorrelation": (
                    half_cycle_corr
                ),

                "quarter_cycle_autocorrelation": (
                    quarter_cycle_corr
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# SEASONAL SCORE
# =========================================================

def add_seasonal_score(
    features
):

    print(
        "Building seasonal evidence score..."
    )

    df = features.copy()

    # -----------------------------------------------------
    # We intentionally use multiple signals rather than
    # saying:
    #
    # autocorrelation > X = seasonal
    #
    # Each test contributes evidence.
    # -----------------------------------------------------

    df[
        "seasonal_evidence_score"
    ] = 0

    mature = (
        df[
            "seasonal_history_weeks"
        ]
        >= MIN_SEASONAL_HISTORY
    )

    # Strong annual autocorrelation
    df.loc[
        mature
        &
        (
            df[
                "detrended_annual_autocorrelation"
            ] >= 0.45
        ),
        "seasonal_evidence_score"
    ] += 2

    # Moderate annual autocorrelation
    df.loc[
        mature
        &
        (
            df[
                "detrended_annual_autocorrelation"
            ] >= 0.25
        )
        &
        (
            df[
                "detrended_annual_autocorrelation"
            ] < 0.45
        ),
        "seasonal_evidence_score"
    ] += 1

    # Similar annual shape
    df.loc[
        mature
        &
        (
            df[
                "detrended_yoy_similarity"
            ] >= 0.45
        ),
        "seasonal_evidence_score"
    ] += 2

    df.loc[
        mature
        &
        (
            df[
                "detrended_yoy_similarity"
            ] >= 0.25
        )
        &
        (
            df[
                "detrended_yoy_similarity"
            ] < 0.45
        ),
        "seasonal_evidence_score"
    ] += 1

    # Strong repeating structure
    df.loc[
        mature
        &
        (
            df[
                "seasonal_strength"
            ] >= 0.55
        ),
        "seasonal_evidence_score"
    ] += 2

    df.loc[
        mature
        &
        (
            df[
                "seasonal_strength"
            ] >= 0.35
        )
        &
        (
            df[
                "seasonal_strength"
            ] < 0.55
        ),
        "seasonal_evidence_score"
    ] += 1

    # Meaningful peak-to-trough amplitude
    df.loc[
        mature
        &
        (
            df[
                "seasonal_amplitude"
            ] >= 0.35
        ),
        "seasonal_evidence_score"
    ] += 1

    # -----------------------------------------------------
    # Final seasonal candidate
    # -----------------------------------------------------

    df[
        "seasonal_candidate"
    ] = (
        mature
        &
        (
            df[
                "seasonal_evidence_score"
            ] >= 4
        )
    )

    return df


# =========================================================
# FINAL HIERARCHICAL CLASSIFICATION
# =========================================================

def classify_final_behavior(
    features
):

    print(
        "Building final hierarchical "
        "demand classification..."
    )

    df = features.copy()

    final_classes = []

    for _, row in df.iterrows():

        history = (
            row[
                "history_weeks"
            ]
        )

        zero_rate = (
            row[
                "zero_demand_rate"
            ]
        )

        adi = row[
            "adi"
        ]

        seasonal = bool(
            row[
                "seasonal_candidate"
            ]
        )

        cv = row[
            "52w_cv"
        ]

        slope = row[
            "52w_normalized_slope"
        ]

        trend_strength = row[
            "52w_trend_r_squared"
        ]

        ratio_13_52 = row[
            "13w_vs_52w_ratio"
        ]

        # -------------------------------------------------
        # 1. NEW PRODUCT
        # -------------------------------------------------

        if history < 52:

            classification = (
                "new_product"
            )

        # -------------------------------------------------
        # 2. INTERMITTENT
        # -------------------------------------------------

        elif (
            zero_rate >= 0.30
            or (
                pd.notna(adi)
                and adi >= 1.32
            )
        ):

            classification = (
                "intermittent"
            )

        # -------------------------------------------------
        # 3. SEASONAL
        #
        # Seasonal is checked BEFORE trend.
        # This prevents the downward portion of a
        # seasonal cycle being called declining.
        # -------------------------------------------------

        elif seasonal:

            classification = (
                "seasonal"
            )

        # -------------------------------------------------
        # 4. STRUCTURAL GROWTH
        #
        # We use both regression evidence and recent
        # level evidence.
        # -------------------------------------------------

        elif (
            pd.notna(slope)
            and slope >= 0.003
            and trend_strength >= 0.15
            and pd.notna(ratio_13_52)
            and ratio_13_52 >= 1.03
        ):

            classification = (
                "growth"
            )

        # -------------------------------------------------
        # 5. STRUCTURAL DECLINE
        # -------------------------------------------------

        elif (
            pd.notna(slope)
            and slope <= -0.003
            and trend_strength >= 0.15
            and pd.notna(ratio_13_52)
            and ratio_13_52 <= 0.97
        ):

            classification = (
                "declining"
            )

        # -------------------------------------------------
        # 6. VOLATILE
        # -------------------------------------------------

        elif (
            pd.notna(cv)
            and cv >= 0.40
        ):

            classification = (
                "volatile"
            )

        # -------------------------------------------------
        # 7. STABLE
        # -------------------------------------------------

        else:

            classification = (
                "stable"
            )

        final_classes.append(
            classification
        )

    df[
        "final_demand_class"
    ] = final_classes

    return df


# =========================================================
# CLASSIFICATION CONFIDENCE
# =========================================================

def add_classification_confidence(
    features
):

    """
    This is not an ML probability.

    It is an interpretable evidence indicator that
    will later help the AI assistant explain whether
    a classification is strongly or weakly supported.
    """

    df = features.copy()

    confidence = []

    for _, row in df.iterrows():

        classification = (
            row[
                "final_demand_class"
            ]
        )

        if classification == "new_product":

            score = 1.0

        elif classification == "intermittent":

            score = min(
                1.0,
                max(
                    row[
                        "zero_demand_rate"
                    ] / 0.60,
                    (
                        row["adi"] / 2.5
                        if pd.notna(
                            row["adi"]
                        )
                        else 0
                    )
                )
            )

        elif classification == "seasonal":

            score = min(
                1.0,
                row[
                    "seasonal_evidence_score"
                ] / 7
            )

        elif classification in [
            "growth",
            "declining",
        ]:

            score = min(
                1.0,
                max(
                    float(
                        row[
                            "52w_trend_r_squared"
                        ]
                    ),
                    0
                )
            )

        elif classification == "volatile":

            score = min(
                1.0,
                (
                    row[
                        "52w_cv"
                    ]
                    / 0.80
                    if pd.notna(
                        row[
                            "52w_cv"
                        ]
                    )
                    else 0
                )
            )

        else:

            cv = (
                row[
                    "52w_cv"
                ]
            )

            if pd.isna(cv):

                score = 0.5

            else:

                score = max(
                    0,
                    min(
                        1.0,
                        1 - cv
                    )
                )

        confidence.append(
            round(
                float(score),
                3
            )
        )

    df[
        "classification_confidence"
    ] = confidence

    return df


# =========================================================
# BUILD OUTPUT
# =========================================================

def build_output(
    demand_features,
    seasonal_features
):

    result = (
        demand_features
        .merge(
            seasonal_features,
            on="sku",
            how="left",
            validate="one_to_one"
        )
    )

    result = (
        add_seasonal_score(
            result
        )
    )

    result = (
        classify_final_behavior(
            result
        )
    )

    result = (
        add_classification_confidence(
            result
        )
    )

    return result


# =========================================================
# VALIDATION
# =========================================================

def validate_output(
    result,
    demand_features
):

    print("\n" + "=" * 75)
    print("SEASONALITY & CLASSIFICATION VALIDATION")
    print("=" * 75)

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
        demand_features["sku"]
    )

    print(
        "✓ All demand-feature SKUs preserved"
    )

    allowed_classes = {
        "stable",
        "volatile",
        "intermittent",
        "seasonal",
        "growth",
        "declining",
        "new_product",
    }

    assert set(
        result[
            "final_demand_class"
        ].unique()
    ).issubset(
        allowed_classes
    )

    print(
        "✓ Final classifications valid"
    )

    assert (
        (
            result[
                "classification_confidence"
            ] >= 0
        )
        &
        (
            result[
                "classification_confidence"
            ] <= 1
        )
    ).all()

    print(
        "✓ Classification confidence "
        "is between 0 and 1"
    )

    # -----------------------------------------------------
    # NEW PRODUCTS
    # -----------------------------------------------------

    new_products = result[
        result[
            "demand_profile"
        ] == "new_product"
    ]

    assert (
        new_products[
            "final_demand_class"
        ] == "new_product"
    ).all()

    print(
        "✓ New-product logic preserved"
    )

    # -----------------------------------------------------
    # INTERMITTENT
    # -----------------------------------------------------

    intermittent = result[
        result[
            "demand_profile"
        ] == "intermittent"
    ]

    intermittent_detection_rate = (
        (
            intermittent[
                "final_demand_class"
            ]
            ==
            "intermittent"
        ).mean()
    )

    assert (
        intermittent_detection_rate
        >= 0.80
    )

    print(
        "✓ Intermittent detection "
        f"rate: "
        f"{intermittent_detection_rate:.1%}"
    )

    # -----------------------------------------------------
    # SEASONAL SIGNAL CHECK
    #
    # We are not demanding perfect classification.
    # But synthetic seasonal SKUs should show stronger
    # seasonal evidence than stable SKUs on average.
    # -----------------------------------------------------

    seasonal_score = (
        result[
            result[
                "demand_profile"
            ] == "seasonal"
        ][
            "seasonal_evidence_score"
        ].mean()
    )

    stable_score = (
        result[
            result[
                "demand_profile"
            ] == "stable"
        ][
            "seasonal_evidence_score"
        ].mean()
    )

    assert (
        seasonal_score
        >
        stable_score
    )

    print(
        "✓ Seasonal SKUs show stronger "
        "seasonal evidence than stable SKUs"
    )

    # -----------------------------------------------------
    # GROWTH VS DECLINE
    # -----------------------------------------------------

    growth_slope = (
        result[
            result[
                "demand_profile"
            ] == "growth"
        ][
            "52w_normalized_slope"
        ].mean()
    )

    declining_slope = (
        result[
            result[
                "demand_profile"
            ] == "declining"
        ][
            "52w_normalized_slope"
        ].mean()
    )

    assert (
        growth_slope
        >
        declining_slope
    )

    print(
        "✓ Growth and decline trend "
        "direction remains valid"
    )

    print(
        "\nAll Phase 4B validation "
        "checks passed."
    )


# =========================================================
# PROFILE SUMMARY
# =========================================================

def print_profile_summary(
    result
):

    print("\n" + "=" * 110)
    print("SEASONAL SIGNAL BY SYNTHETIC PROFILE")
    print("=" * 110)

    summary = (
        result
        .groupby(
            "demand_profile"
        )
        .agg(
            sku_count=(
                "sku",
                "count"
            ),

            avg_annual_corr=(
                "detrended_annual_autocorrelation",
                "mean"
            ),

            avg_yoy_similarity=(
                "detrended_yoy_similarity",
                "mean"
            ),

            avg_seasonal_strength=(
                "seasonal_strength",
                "mean"
            ),

            avg_seasonal_amplitude=(
                "seasonal_amplitude",
                "mean"
            ),

            avg_seasonal_score=(
                "seasonal_evidence_score",
                "mean"
            ),

            seasonal_candidates=(
                "seasonal_candidate",
                "sum"
            ),
        )
        .round(3)
    )

    print(
        summary.to_string()
    )


# =========================================================
# CONFUSION TABLE
# =========================================================

def print_classification_table(
    result
):

    print("\n" + "=" * 100)
    print("SYNTHETIC PROFILE VS FINAL ANALYTICAL CLASS")
    print("=" * 100)

    table = pd.crosstab(
        result[
            "demand_profile"
        ],
        result[
            "final_demand_class"
        ]
    )

    print(
        table.to_string()
    )


# =========================================================
# CLASSIFICATION SUMMARY
# =========================================================

def print_final_summary(
    result
):

    print("\n" + "=" * 80)
    print("FINAL DEMAND CLASS DISTRIBUTION")
    print("=" * 80)

    counts = (
        result[
            "final_demand_class"
        ]
        .value_counts()
    )

    print(
        counts.to_string()
    )

    print(
        "\nAverage classification confidence:"
    )

    confidence = (
        result
        .groupby(
            "final_demand_class"
        )[
            "classification_confidence"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        .round(3)
    )

    print(
        confidence.to_string()
    )


# =========================================================
# EXAMPLES
# =========================================================

def print_examples(
    result
):

    print("\n" + "=" * 130)
    print("EXAMPLE SEASONALITY ANALYSIS")
    print("=" * 130)

    columns = [
        "sku",
        "demand_profile",
        "final_demand_class",
        "classification_confidence",
        "52w_avg_demand",
        "52w_cv",
        "zero_demand_rate",
        "52w_normalized_slope",
        "52w_trend_r_squared",
        "detrended_annual_autocorrelation",
        "detrended_yoy_similarity",
        "seasonal_strength",
        "seasonal_amplitude",
        "seasonal_evidence_score",
    ]

    examples = (
        result
        .groupby(
            "demand_profile",
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

    print("=" * 75)

    print(
        "PROCUREMENT AI - "
        "SEASONALITY & DEMAND BEHAVIOR"
    )

    print("=" * 75)

    data = load_data()

    demand = prepare_demand(
        data["sales"]
    )

    seasonal_features = (
        calculate_seasonality_features(
            demand
        )
    )

    result = build_output(
        data["features"],
        seasonal_features
    )

    validate_output(
        result,
        data["features"]
    )

    save_output(
        result
    )

    print_profile_summary(
        result
    )

    print_classification_table(
        result
    )

    print_final_summary(
        result
    )

    print_examples(
        result
    )

    print("\n" + "=" * 75)

    print(
        "PHASE 4B COMPLETE"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()