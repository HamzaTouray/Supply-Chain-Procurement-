from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

SNAPSHOT_DATE = pd.Timestamp("2026-09-30")

WINDOWS = {
    "4w": 4,
    "13w": 13,
    "26w": 26,
    "52w": 52,
}


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

PRODUCT_FILE = (
    PROCESSED_DATA_DIR
    / "products_clean.csv"
)

PLANNING_MASTER_FILE = (
    PROCESSED_DATA_DIR
    / "planning_master.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "demand_features.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    print("\nLoading demand-analysis inputs...")

    required_files = {
        "sales": SALES_FILE,
        "products": PRODUCT_FILE,
        "planning_master": PLANNING_MASTER_FILE,
    }

    data = {}

    for name, file_path in required_files.items():

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing required file: "
                f"{file_path}"
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
# PREPARE VALID DEMAND HISTORY
# =========================================================

def prepare_demand_history(
    sales_df
):

    print(
        "\nPreparing demand history..."
    )

    sales = sales_df.copy()

    # -----------------------------------------------------
    # Preserve pre-launch nulls.
    #
    # NaN = product did not yet have demand history.
    # 0   = product existed but had zero demand.
    # -----------------------------------------------------

    sales[
        "units_demanded"
    ] = pd.to_numeric(
        sales[
            "units_demanded"
        ],
        errors="coerce"
    )

    sales[
        "units_sold"
    ] = pd.to_numeric(
        sales[
            "units_sold"
        ],
        errors="coerce"
    )

    sales[
        "lost_sales"
    ] = pd.to_numeric(
        sales[
            "lost_sales"
        ],
        errors="coerce"
    )

    valid = sales[
        sales[
            "units_demanded"
        ].notna()
    ].copy()

    valid = valid.sort_values(
        [
            "sku",
            "week_date"
        ]
    )

    print(
        f"✓ Valid demand observations: "
        f"{len(valid):,}"
    )

    print(
        f"✓ Pre-launch rows excluded: "
        f"{len(sales) - len(valid):,}"
    )

    return valid


# =========================================================
# BASIC HISTORY FEATURES
# =========================================================

def calculate_history_features(
    demand_df
):

    print(
        "Calculating lifetime demand features..."
    )

    features = (
        demand_df
        .groupby("sku")
        .agg(
            history_weeks=(
                "units_demanded",
                "count"
            ),

            first_demand_date=(
                "week_date",
                "min"
            ),

            last_demand_date=(
                "week_date",
                "max"
            ),

            lifetime_avg_demand=(
                "units_demanded",
                "mean"
            ),

            lifetime_median_demand=(
                "units_demanded",
                "median"
            ),

            lifetime_std_demand=(
                "units_demanded",
                "std"
            ),

            lifetime_min_demand=(
                "units_demanded",
                "min"
            ),

            lifetime_max_demand=(
                "units_demanded",
                "max"
            ),

            lifetime_total_demand=(
                "units_demanded",
                "sum"
            ),

            lifetime_units_sold=(
                "units_sold",
                "sum"
            ),

            lifetime_lost_sales=(
                "lost_sales",
                "sum"
            ),
        )
        .reset_index()
    )

    features[
        "history_age_weeks"
    ] = (
        (
            features[
                "last_demand_date"
            ]
            -
            features[
                "first_demand_date"
            ]
        ).dt.days
        / 7
        + 1
    )

    features[
        "lifetime_cv"
    ] = np.where(
        features[
            "lifetime_avg_demand"
        ] > 0,

        features[
            "lifetime_std_demand"
        ]
        /
        features[
            "lifetime_avg_demand"
        ],

        np.nan
    )

    features[
        "lifetime_lost_sales_rate"
    ] = np.where(
        features[
            "lifetime_total_demand"
        ] > 0,

        features[
            "lifetime_lost_sales"
        ]
        /
        features[
            "lifetime_total_demand"
        ],

        0
    )

    return features


# =========================================================
# WINDOW FEATURES
# =========================================================

def calculate_window_features(
    demand_df,
    weeks,
    prefix
):

    start_date = (
        SNAPSHOT_DATE
        - pd.Timedelta(
            weeks=weeks - 1
        )
    )

    window = demand_df[
        (
            demand_df[
                "week_date"
            ] >= start_date
        )
        &
        (
            demand_df[
                "week_date"
            ] <= SNAPSHOT_DATE
        )
    ].copy()

    features = (
        window
        .groupby("sku")
        .agg(
            **{
                f"{prefix}_observations": (
                    "units_demanded",
                    "count"
                ),

                f"{prefix}_avg_demand": (
                    "units_demanded",
                    "mean"
                ),

                f"{prefix}_median_demand": (
                    "units_demanded",
                    "median"
                ),

                f"{prefix}_std_demand": (
                    "units_demanded",
                    "std"
                ),

                f"{prefix}_min_demand": (
                    "units_demanded",
                    "min"
                ),

                f"{prefix}_max_demand": (
                    "units_demanded",
                    "max"
                ),

                f"{prefix}_total_demand": (
                    "units_demanded",
                    "sum"
                ),

                f"{prefix}_units_sold": (
                    "units_sold",
                    "sum"
                ),

                f"{prefix}_lost_sales": (
                    "lost_sales",
                    "sum"
                ),
            }
        )
        .reset_index()
    )

    features[
        f"{prefix}_cv"
    ] = np.where(
        features[
            f"{prefix}_avg_demand"
        ] > 0,

        features[
            f"{prefix}_std_demand"
        ]
        /
        features[
            f"{prefix}_avg_demand"
        ],

        np.nan
    )

    features[
        f"{prefix}_lost_sales_rate"
    ] = np.where(
        features[
            f"{prefix}_total_demand"
        ] > 0,

        features[
            f"{prefix}_lost_sales"
        ]
        /
        features[
            f"{prefix}_total_demand"
        ],

        0
    )

    return features


# =========================================================
# ZERO DEMAND / INTERMITTENCY
# =========================================================

def calculate_intermittency_features(
    demand_df
):

    print(
        "Calculating intermittency features..."
    )

    records = []

    for sku, group in (
        demand_df.groupby("sku")
    ):

        group = group.sort_values(
            "week_date"
        )

        demand = (
            group[
                "units_demanded"
            ]
            .astype(float)
        )

        zero_weeks = int(
            (demand == 0).sum()
        )

        positive_weeks = int(
            (demand > 0).sum()
        )

        total_weeks = len(
            demand
        )

        zero_rate = (
            zero_weeks
            / total_weeks
            if total_weeks > 0
            else np.nan
        )

        positive_rate = (
            positive_weeks
            / total_weeks
            if total_weeks > 0
            else np.nan
        )

        # -------------------------------------------------
        # ADI:
        # Average Demand Interval
        #
        # Average number of periods between
        # non-zero demand occurrences.
        # -------------------------------------------------

        positive_positions = np.where(
            demand.to_numpy() > 0
        )[0]

        if len(
            positive_positions
        ) >= 2:

            intervals = np.diff(
                positive_positions
            )

            adi = float(
                np.mean(
                    intervals
                )
            )

        elif len(
            positive_positions
        ) == 1:

            adi = float(
                total_weeks
            )

        else:

            adi = np.nan

        # -------------------------------------------------
        # CV² of positive demand sizes
        #
        # Used with ADI for intermittent-demand
        # classification.
        # -------------------------------------------------

        positive_demand = demand[
            demand > 0
        ]

        if (
            len(
                positive_demand
            ) >= 2
            and positive_demand.mean() > 0
        ):

            positive_cv = (
                positive_demand.std(
                    ddof=1
                )
                /
                positive_demand.mean()
            )

            cv_squared = (
                positive_cv ** 2
            )

        else:

            cv_squared = np.nan

        records.append(
            {
                "sku": sku,
                "zero_demand_weeks": (
                    zero_weeks
                ),
                "positive_demand_weeks": (
                    positive_weeks
                ),
                "zero_demand_rate": (
                    zero_rate
                ),
                "positive_demand_rate": (
                    positive_rate
                ),
                "adi": adi,
                "positive_demand_cv_squared": (
                    cv_squared
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# TREND FEATURES
# =========================================================

def calculate_trend_features(
    demand_df
):

    print(
        "Calculating demand trend features..."
    )

    records = []

    for sku, group in (
        demand_df.groupby("sku")
    ):

        group = group.sort_values(
            "week_date"
        ).copy()

        # Use up to the most recent 52 weeks.
        group = group.tail(
            52
        )

        y = (
            group[
                "units_demanded"
            ]
            .astype(float)
            .to_numpy()
        )

        n = len(y)

        if n >= 4:

            x = np.arange(
                n,
                dtype=float
            )

            slope, intercept = (
                np.polyfit(
                    x,
                    y,
                    1
                )
            )

            fitted = (
                slope * x
                + intercept
            )

            ss_res = np.sum(
                (
                    y
                    - fitted
                ) ** 2
            )

            ss_tot = np.sum(
                (
                    y
                    - np.mean(y)
                ) ** 2
            )

            if ss_tot > 0:

                r_squared = (
                    1
                    -
                    ss_res
                    /
                    ss_tot
                )

            else:

                r_squared = 0.0

            mean_demand = np.mean(
                y
            )

            if mean_demand > 0:

                normalized_slope = (
                    slope
                    /
                    mean_demand
                )

            else:

                normalized_slope = 0.0

        else:

            slope = np.nan
            normalized_slope = np.nan
            r_squared = np.nan

        records.append(
            {
                "sku": sku,
                "52w_trend_slope": (
                    slope
                ),
                "52w_normalized_slope": (
                    normalized_slope
                ),
                "52w_trend_r_squared": (
                    r_squared
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# =========================================================
# RECENT MOMENTUM
# =========================================================

def calculate_momentum_features(
    features
):

    df = features.copy()

    # -----------------------------------------------------
    # 4W vs 13W
    # -----------------------------------------------------

    df[
        "4w_vs_13w_ratio"
    ] = np.where(
        df[
            "13w_avg_demand"
        ] > 0,

        df[
            "4w_avg_demand"
        ]
        /
        df[
            "13w_avg_demand"
        ],

        np.nan
    )

    # -----------------------------------------------------
    # 13W vs 52W
    # -----------------------------------------------------

    df[
        "13w_vs_52w_ratio"
    ] = np.where(
        df[
            "52w_avg_demand"
        ] > 0,

        df[
            "13w_avg_demand"
        ]
        /
        df[
            "52w_avg_demand"
        ],

        np.nan
    )

    # -----------------------------------------------------
    # 4W vs 52W
    # -----------------------------------------------------

    df[
        "4w_vs_52w_ratio"
    ] = np.where(
        df[
            "52w_avg_demand"
        ] > 0,

        df[
            "4w_avg_demand"
        ]
        /
        df[
            "52w_avg_demand"
        ],

        np.nan
    )

    # -----------------------------------------------------
    # Demand acceleration
    #
    # > 0 means recent demand is strengthening
    # < 0 means recent demand is weakening
    # -----------------------------------------------------

    df[
        "demand_acceleration"
    ] = (
        df[
            "4w_vs_13w_ratio"
        ]
        -
        df[
            "13w_vs_52w_ratio"
        ]
    )

    return df


# =========================================================
# HISTORY MATURITY
# =========================================================

def add_history_maturity(
    features
):

    df = features.copy()

    conditions = [
        df[
            "history_weeks"
        ] < 13,

        (
            df[
                "history_weeks"
            ] >= 13
        )
        &
        (
            df[
                "history_weeks"
            ] < 26
        ),

        (
            df[
                "history_weeks"
            ] >= 26
        )
        &
        (
            df[
                "history_weeks"
            ] < 52
        ),

        df[
            "history_weeks"
        ] >= 52,
    ]

    labels = [
        "very_new",
        "new",
        "developing",
        "mature",
    ]

    df[
        "history_maturity"
    ] = np.select(
        conditions,
        labels,
        default="unknown"
    )

    return df


# =========================================================
# DEMAND BEHAVIOR CLASSIFICATION
# =========================================================

def classify_demand_behavior(
    features
):

    print(
        "Classifying analytical demand behavior..."
    )

    df = features.copy()

    classifications = []

    for _, row in df.iterrows():

        history_weeks = (
            row[
                "history_weeks"
            ]
        )

        zero_rate = (
            row[
                "zero_demand_rate"
            ]
        )

        adi = row["adi"]

        cv = row[
            "52w_cv"
        ]

        trend = row[
            "52w_normalized_slope"
        ]

        trend_strength = row[
            "52w_trend_r_squared"
        ]

        # -------------------------------------------------
        # NEW PRODUCT
        # -------------------------------------------------

        if history_weeks < 52:

            classification = (
                "new_product"
            )

        # -------------------------------------------------
        # INTERMITTENT
        #
        # High zero frequency or ADI above
        # common intermittent threshold.
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
        # GROWTH
        # -------------------------------------------------

        elif (
            pd.notna(trend)
            and trend >= 0.008
            and trend_strength >= 0.15
        ):

            classification = (
                "growth"
            )

        # -------------------------------------------------
        # DECLINING
        # -------------------------------------------------

        elif (
            pd.notna(trend)
            and trend <= -0.008
            and trend_strength >= 0.15
        ):

            classification = (
                "declining"
            )

        # -------------------------------------------------
        # VOLATILE
        # -------------------------------------------------

        elif (
            pd.notna(cv)
            and cv >= 0.45
        ):

            classification = (
                "volatile"
            )

        # -------------------------------------------------
        # STABLE / SEASONAL CANDIDATE
        #
        # We deliberately do not finalize
        # seasonality here.
        #
        # Seasonality requires its own
        # statistical test in Phase 4B.
        # -------------------------------------------------

        else:

            classification = (
                "stable_or_seasonal"
            )

        classifications.append(
            classification
        )

    df[
        "analytical_demand_class"
    ] = classifications

    return df


# =========================================================
# BUILD FEATURE TABLE
# =========================================================

def build_feature_table(
    data,
    demand_df
):

    history_features = (
        calculate_history_features(
            demand_df
        )
    )

    features = (
        data[
            "products"
        ][
            [
                "sku",
                "product_name",
                "category",
                "supplier_id",
                "demand_profile",
            ]
        ]
        .copy()
    )

    features = features.merge(
        history_features,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    # -----------------------------------------------------
    # WINDOW FEATURES
    # -----------------------------------------------------

    for prefix, weeks in (
        WINDOWS.items()
    ):

        window_features = (
            calculate_window_features(
                demand_df,
                weeks,
                prefix
            )
        )

        features = features.merge(
            window_features,
            on="sku",
            how="left",
            validate="one_to_one"
        )

    # -----------------------------------------------------
    # INTERMITTENCY
    # -----------------------------------------------------

    intermittency = (
        calculate_intermittency_features(
            demand_df
        )
    )

    features = features.merge(
        intermittency,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    trend_features = (
        calculate_trend_features(
            demand_df
        )
    )

    features = features.merge(
        trend_features,
        on="sku",
        how="left",
        validate="one_to_one"
    )

    # -----------------------------------------------------
    # MOMENTUM
    # -----------------------------------------------------

    features = (
        calculate_momentum_features(
            features
        )
    )

    # -----------------------------------------------------
    # MATURITY
    # -----------------------------------------------------

    features = (
        add_history_maturity(
            features
        )
    )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    features = (
        classify_demand_behavior(
            features
        )
    )

    return features


# =========================================================
# VALIDATION
# =========================================================

def validate_features(
    features,
    products_df
):

    print("\n" + "=" * 70)
    print("DEMAND FEATURE VALIDATION")
    print("=" * 70)

    assert len(
        features
    ) == len(
        products_df
    )

    print(
        f"✓ Exactly "
        f"{len(features)} SKU rows"
    )

    assert (
        features[
            "sku"
        ].is_unique
    )

    print(
        "✓ One row per SKU"
    )

    assert set(
        features["sku"]
    ) == set(
        products_df["sku"]
    )

    print(
        "✓ All products preserved"
    )

    assert (
        features[
            "history_weeks"
        ] > 0
    ).all()

    print(
        "✓ All SKUs have demand history"
    )

    assert (
        (
            features[
                "zero_demand_rate"
            ] >= 0
        )
        &
        (
            features[
                "zero_demand_rate"
            ] <= 1
        )
    ).all()

    print(
        "✓ Zero-demand rates valid"
    )

    assert (
        features[
            "lifetime_total_demand"
        ] >= 0
    ).all()

    print(
        "✓ Demand quantities non-negative"
    )

    assert (
        features[
            "lifetime_lost_sales"
        ] >= 0
    ).all()

    print(
        "✓ Lost-sales quantities non-negative"
    )

    # -----------------------------------------------------
    # NEW PRODUCT LOGIC
    # -----------------------------------------------------

    new_products = features[
        features[
            "demand_profile"
        ] == "new_product"
    ]

    assert (
        new_products[
            "history_weeks"
        ] < 52
    ).all()

    print(
        "✓ Synthetic new products have "
        "limited history"
    )

    # -----------------------------------------------------
    # INTERMITTENT SIGNAL
    # -----------------------------------------------------

    intermittent = features[
        features[
            "demand_profile"
        ] == "intermittent"
    ]

    assert (
        intermittent[
            "zero_demand_rate"
        ].mean()
        >
        features[
            features[
                "demand_profile"
            ] == "stable"
        ][
            "zero_demand_rate"
        ].mean()
    )

    print(
        "✓ Intermittent products show "
        "more zero-demand weeks than stable products"
    )

    # -----------------------------------------------------
    # GROWTH VS DECLINING TREND
    # -----------------------------------------------------

    growth_mean_slope = (
        features[
            features[
                "demand_profile"
            ] == "growth"
        ][
            "52w_normalized_slope"
        ].mean()
    )

    declining_mean_slope = (
        features[
            features[
                "demand_profile"
            ] == "declining"
        ][
            "52w_normalized_slope"
        ].mean()
    )

    assert (
        growth_mean_slope
        >
        declining_mean_slope
    )

    print(
        "✓ Growth products trend stronger "
        "than declining products"
    )

    print(
        "\nAll demand-feature validation "
        "checks passed."
    )


# =========================================================
# CLASSIFICATION COMPARISON
# =========================================================

def print_classification_comparison(
    features
):

    print("\n" + "=" * 90)
    print("SYNTHETIC PROFILE VS ANALYTICAL CLASSIFICATION")
    print("=" * 90)

    comparison = pd.crosstab(
        features[
            "demand_profile"
        ],
        features[
            "analytical_demand_class"
        ]
    )

    print(
        comparison.to_string()
    )


# =========================================================
# PROFILE STATISTICS
# =========================================================

def print_profile_statistics(
    features
):

    print("\n" + "=" * 110)
    print("DEMAND PROFILE STATISTICS")
    print("=" * 110)

    summary = (
        features
        .groupby(
            "demand_profile"
        )
        .agg(
            sku_count=(
                "sku",
                "count"
            ),

            avg_13w_demand=(
                "13w_avg_demand",
                "mean"
            ),

            avg_52w_cv=(
                "52w_cv",
                "mean"
            ),

            avg_zero_rate=(
                "zero_demand_rate",
                "mean"
            ),

            avg_adi=(
                "adi",
                "mean"
            ),

            avg_trend=(
                "52w_normalized_slope",
                "mean"
            ),

            avg_trend_strength=(
                "52w_trend_r_squared",
                "mean"
            ),

            avg_lost_sales_rate=(
                "13w_lost_sales_rate",
                "mean"
            ),
        )
        .round(3)
    )

    print(
        summary.to_string()
    )


# =========================================================
# EXAMPLE SKUS
# =========================================================

def print_examples(
    features
):

    print("\n" + "=" * 120)
    print("EXAMPLE DEMAND ANALYTICS")
    print("=" * 120)

    columns = [
        "sku",
        "demand_profile",
        "analytical_demand_class",
        "history_weeks",
        "4w_avg_demand",
        "13w_avg_demand",
        "52w_avg_demand",
        "52w_cv",
        "zero_demand_rate",
        "adi",
        "52w_normalized_slope",
        "52w_trend_r_squared",
        "demand_acceleration",
    ]

    examples = (
        features
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

def save_features(
    features
):

    features.to_csv(
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

    print("=" * 70)

    print(
        "PROCUREMENT AI - "
        "DEMAND ANALYTICS"
    )

    print("=" * 70)

    data = load_data()

    demand_df = (
        prepare_demand_history(
            data["sales"]
        )
    )

    features = (
        build_feature_table(
            data,
            demand_df
        )
    )

    validate_features(
        features,
        data["products"]
    )

    save_features(
        features
    )

    print_profile_statistics(
        features
    )

    print_classification_comparison(
        features
    )

    print_examples(
        features
    )

    print("\n" + "=" * 70)

    print(
        "PHASE 4A COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()