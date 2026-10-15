import random
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

RANDOM_SEED = 42
NUM_WEEKS = 104

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PRODUCT_FILE = RAW_DATA_DIR / "products.csv"
OUTPUT_FILE = RAW_DATA_DIR / "sales_history.csv"


# =========================================================
# DATE RANGE
# =========================================================

def create_week_dates():

    # Fixed end date keeps our synthetic data reproducible.
    end_date = pd.Timestamp("2026-09-28")

    return pd.date_range(
        end=end_date,
        periods=NUM_WEEKS,
        freq="W-MON"
    )


# =========================================================
# BASE DEMAND
# =========================================================

def get_base_demand(category):

    ranges = {
        "Equipment": (1, 8),
        "Accessories": (10, 45),
        "Consumables": (15, 70),
        "Spare Parts": (5, 35),
        "Electronics": (3, 20),
    }

    low, high = ranges[category]

    return np.random.uniform(low, high)


# =========================================================
# DEMAND PROFILE FUNCTIONS
# =========================================================

def stable_demand(base):

    demand = np.random.normal(
        loc=base,
        scale=max(1, base * 0.10)
    )

    return max(0, demand)


def volatile_demand(base):

    demand = np.random.normal(
        loc=base,
        scale=max(2, base * 0.45)
    )

    return max(0, demand)


def seasonal_demand(base, week):

    seasonal_factor = (
        1
        + 0.35
        * np.sin(
            2 * np.pi * week / 52
        )
    )

    noise = np.random.normal(
        0,
        max(1, base * 0.10)
    )

    demand = (
        base * seasonal_factor
        + noise
    )

    return max(0, demand)


def growth_demand(base, week):

    growth_factor = (
        1
        + 0.008 * week
    )

    noise = np.random.normal(
        0,
        max(1, base * 0.10)
    )

    demand = (
        base * growth_factor
        + noise
    )

    return max(0, demand)


def declining_demand(base, week):

    decline_factor = max(
        0.25,
        1 - 0.006 * week
    )

    noise = np.random.normal(
        0,
        max(1, base * 0.10)
    )

    demand = (
        base * decline_factor
        + noise
    )

    return max(0, demand)


def intermittent_demand(base):

    # Roughly 65% of weeks have no demand.
    if np.random.random() < 0.65:
        return 0

    demand = np.random.normal(
        loc=base,
        scale=max(1, base * 0.30)
    )

    return max(0, demand)


def new_product_demand(base, week, launch_week):

    if week < launch_week:
        return None

    weeks_since_launch = (
        week - launch_week
    )

    ramp_factor = min(
        1.0,
        0.20
        + 0.05 * weeks_since_launch
    )

    noise = np.random.normal(
        0,
        max(1, base * 0.10)
    )

    demand = (
        base * ramp_factor
        + noise
    )

    return max(0, demand)


# =========================================================
# GENERATE SALES HISTORY
# =========================================================

def generate_sales_history(products_df):

    week_dates = create_week_dates()

    records = []

    for _, product in products_df.iterrows():

        sku = product["sku"]
        category = product["category"]
        profile = product["demand_profile"]
        unit_price = product["unit_price"]

        base_demand = get_base_demand(
            category
        )

        # New products launch during the
        # final portion of history.
        launch_week = random.randint(
            65,
            90
        )

        for week_number, week_date in enumerate(
            week_dates
        ):

            # -------------------------------------
            # TRUE DEMAND
            # -------------------------------------

            if profile == "stable":

                demand = stable_demand(
                    base_demand
                )

            elif profile == "volatile":

                demand = volatile_demand(
                    base_demand
                )

            elif profile == "seasonal":

                demand = seasonal_demand(
                    base_demand,
                    week_number
                )

            elif profile == "growth":

                demand = growth_demand(
                    base_demand,
                    week_number
                )

            elif profile == "declining":

                demand = declining_demand(
                    base_demand,
                    week_number
                )

            elif profile == "intermittent":

                demand = intermittent_demand(
                    base_demand
                )

            elif profile == "new_product":

                demand = new_product_demand(
                    base_demand,
                    week_number,
                    launch_week
                )

            else:

                raise ValueError(
                    f"Unknown demand profile: "
                    f"{profile}"
                )

            # -------------------------------------
            # BEFORE PRODUCT LAUNCH
            # -------------------------------------

            if demand is None:

                units_demanded = np.nan
                units_sold = np.nan
                lost_sales = np.nan
                revenue = np.nan

            else:

                units_demanded = int(
                    round(demand)
                )

                # ---------------------------------
                # SIMULATED STOCKOUT
                # ---------------------------------

                # Most weeks fulfill all demand.
                # Occasionally inventory shortages
                # prevent full sales.
                stockout_probability = 0.08

                if (
                    units_demanded > 0
                    and np.random.random()
                    < stockout_probability
                ):

                    fulfillment_rate = (
                        np.random.uniform(
                            0.40,
                            0.90
                        )
                    )

                    units_sold = int(
                        round(
                            units_demanded
                            * fulfillment_rate
                        )
                    )

                else:

                    units_sold = (
                        units_demanded
                    )

                units_sold = min(
                    units_sold,
                    units_demanded
                )

                lost_sales = (
                    units_demanded
                    - units_sold
                )

                revenue = round(
                    units_sold
                    * unit_price,
                    2
                )

            # -------------------------------------
            # PROMOTION
            # -------------------------------------

            promotion_flag = (
                np.random.random()
                < 0.04
            )

            # -------------------------------------
            # RECORD
            # -------------------------------------

            records.append(
                {
                    "week_date": week_date,
                    "sku": sku,
                    "units_demanded": (
                        units_demanded
                    ),
                    "units_sold": units_sold,
                    "lost_sales": lost_sales,
                    "revenue": revenue,
                    "promotion_flag": (
                        promotion_flag
                    ),
                }
            )

    return pd.DataFrame(records)


# =========================================================
# VALIDATION
# =========================================================

def validate_sales_history(
    sales_df,
    products_df
):

    print("\n" + "=" * 60)
    print("SALES HISTORY VALIDATION")
    print("=" * 60)

    expected_max_rows = (
        len(products_df)
        * NUM_WEEKS
    )

    assert len(sales_df) == expected_max_rows

    assert (
        sales_df["units_sold"].dropna()
        <=
        sales_df["units_demanded"].dropna()
    ).all()

    assert (
        sales_df["lost_sales"].dropna()
        >= 0
    ).all()

    valid_rows = sales_df[
        sales_df["units_demanded"]
        .notna()
    ]

    assert (
        valid_rows["units_demanded"]
        ==
        valid_rows["units_sold"]
        + valid_rows["lost_sales"]
    ).all()

    assert (
        valid_rows["revenue"]
        >= 0
    ).all()

    print(
        f"✓ Row count correct: "
        f"{len(sales_df):,}"
    )

    print(
        "✓ Units sold never exceed demand"
    )

    print(
        "✓ Lost sales are non-negative"
    )

    print(
        "✓ Demand = sales + lost sales"
    )

    print(
        "✓ Revenue is non-negative"
    )

    print(
        "\nAll sales-history validation "
        "checks passed."
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary(
    sales_df,
    products_df
):

    merged = sales_df.merge(
        products_df[
            [
                "sku",
                "demand_profile"
            ]
        ],
        on="sku",
        how="left"
    )

    print("\n" + "=" * 60)
    print("DEMAND SUMMARY")
    print("=" * 60)

    print("\nTotal true demand:")

    print(
        f"{sales_df['units_demanded'].sum():,.0f}"
    )

    print("\nTotal units sold:")

    print(
        f"{sales_df['units_sold'].sum():,.0f}"
    )

    print("\nTotal lost sales:")

    print(
        f"{sales_df['lost_sales'].sum():,.0f}"
    )

    print("\nLost-sales rate:")

    total_demand = (
        sales_df["units_demanded"]
        .sum()
    )

    lost_sales = (
        sales_df["lost_sales"]
        .sum()
    )

    lost_sales_rate = (
        lost_sales / total_demand
        if total_demand > 0
        else 0
    )

    print(
        f"{lost_sales_rate:.2%}"
    )

    print(
        "\nAverage Weekly Demand "
        "by Profile:"
    )

    profile_summary = (
        merged
        .groupby("demand_profile")[
            "units_demanded"
        ]
        .mean()
        .round(2)
        .sort_values(
            ascending=False
        )
    )

    print(profile_summary)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print(
        "PROCUREMENT AI - "
        "DEMAND GENERATOR"
    )
    print("=" * 60)

    # ---------------------------------------------
    # LOAD PRODUCTS
    # ---------------------------------------------

    if not PRODUCT_FILE.exists():

        raise FileNotFoundError(
            "products.csv was not found. "
            "Run generate_data.py first."
        )

    products_df = pd.read_csv(
        PRODUCT_FILE
    )

    print(
        f"\nProducts loaded: "
        f"{len(products_df)}"
    )

    # ---------------------------------------------
    # GENERATE DEMAND
    # ---------------------------------------------

    sales_df = generate_sales_history(
        products_df
    )

    # ---------------------------------------------
    # VALIDATE
    # ---------------------------------------------

    validate_sales_history(
        sales_df,
        products_df
    )

    # ---------------------------------------------
    # SAVE
    # ---------------------------------------------

    sales_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nCreated: {OUTPUT_FILE}"
    )

    # ---------------------------------------------
    # SUMMARY
    # ---------------------------------------------

    print_summary(
        sales_df,
        products_df
    )

    print("\n" + "=" * 60)
    print("PHASE 2D COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()