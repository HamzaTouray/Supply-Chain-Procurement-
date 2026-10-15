from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PRODUCT_FILE = RAW_DATA_DIR / "products.csv"
SALES_FILE = RAW_DATA_DIR / "sales_history.csv"


# =========================================================
# LOAD DATA
# =========================================================

products = pd.read_csv(PRODUCT_FILE)

sales = pd.read_csv(
    SALES_FILE,
    parse_dates=["week_date"]
)


# =========================================================
# SELECT ONE SKU FROM EACH PROFILE
# =========================================================

profiles = [
    "stable",
    "volatile",
    "seasonal",
    "growth",
    "declining",
    "intermittent",
    "new_product",
]

selected_skus = {}

for profile in profiles:

    sku = (
        products.loc[
            products["demand_profile"] == profile,
            "sku"
        ]
        .iloc[0]
    )

    selected_skus[profile] = sku


# =========================================================
# ANALYZE PROFILE
# =========================================================

def analyze_sku(profile, sku):

    sku_data = (
        sales[
            sales["sku"] == sku
        ]
        .sort_values("week_date")
        .copy()
    )

    demand = sku_data["units_demanded"]

    valid_demand = demand.dropna()

    first_half = valid_demand.iloc[
        :len(valid_demand) // 2
    ]

    second_half = valid_demand.iloc[
        len(valid_demand) // 2:
    ]

    average = valid_demand.mean()

    std_dev = valid_demand.std()

    cv = (
        std_dev / average
        if average > 0
        else 0
    )

    zero_rate = (
        (valid_demand == 0).mean()
        if len(valid_demand) > 0
        else 0
    )

    first_half_avg = first_half.mean()

    second_half_avg = second_half.mean()

    change_pct = (
        (
            second_half_avg
            - first_half_avg
        )
        / first_half_avg
        if first_half_avg > 0
        else 0
    )

    first_valid_date = (
        sku_data.loc[
            sku_data["units_demanded"].notna(),
            "week_date"
        ]
        .min()
    )

    return {
        "profile": profile,
        "sku": sku,
        "weeks_with_history": len(valid_demand),
        "avg_demand": round(average, 2),
        "std_dev": round(std_dev, 2),
        "cv": round(cv, 2),
        "zero_demand_rate": round(zero_rate, 2),
        "first_half_avg": round(first_half_avg, 2),
        "second_half_avg": round(second_half_avg, 2),
        "change_pct": round(change_pct, 2),
        "first_demand_date": first_valid_date,
    }


# =========================================================
# RUN ANALYSIS
# =========================================================

results = []

for profile, sku in selected_skus.items():

    results.append(
        analyze_sku(
            profile,
            sku
        )
    )


results_df = pd.DataFrame(results)


# =========================================================
# PRINT RESULTS
# =========================================================

print("=" * 90)
print("DEMAND PROFILE BEHAVIOR VALIDATION")
print("=" * 90)

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# SIMPLE CHECKS
# =========================================================

print("\n" + "=" * 90)
print("PROFILE CHECKS")
print("=" * 90)


# Growth
growth_row = results_df[
    results_df["profile"] == "growth"
].iloc[0]

if growth_row["second_half_avg"] > growth_row["first_half_avg"]:
    print("✓ Growth SKU increases over time")
else:
    print("⚠ Growth SKU did not increase as expected")


# Declining
declining_row = results_df[
    results_df["profile"] == "declining"
].iloc[0]

if declining_row["second_half_avg"] < declining_row["first_half_avg"]:
    print("✓ Declining SKU decreases over time")
else:
    print("⚠ Declining SKU did not decrease as expected")


# Intermittent
intermittent_row = results_df[
    results_df["profile"] == "intermittent"
].iloc[0]

if intermittent_row["zero_demand_rate"] >= 0.50:
    print("✓ Intermittent SKU has frequent zero-demand weeks")
else:
    print("⚠ Intermittent SKU does not have enough zero-demand weeks")


# New product
new_product_row = results_df[
    results_df["profile"] == "new_product"
].iloc[0]

if new_product_row["weeks_with_history"] < 104:
    print("✓ New product has limited history")
else:
    print("⚠ New product incorrectly has full history")


# Volatile vs stable
stable_row = results_df[
    results_df["profile"] == "stable"
].iloc[0]

volatile_row = results_df[
    results_df["profile"] == "volatile"
].iloc[0]

if volatile_row["cv"] > stable_row["cv"]:
    print("✓ Volatile SKU varies more than stable SKU")
else:
    print("⚠ Volatile SKU is not sufficiently different from stable SKU")


print("\nValidation complete.")