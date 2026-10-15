import random
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# PROJECT SETTINGS
# =========================================================

RANDOM_SEED = 42
NUM_SUPPLIERS = 10
NUM_SKUS = 200
NUM_WEEKS = 104

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# FILE PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SUPPLIER GENERATION
# =========================================================

def generate_suppliers():

    suppliers = [
        {
            "supplier_id": "SUP-001",
            "supplier_name": "Apex Industrial",
            "country": "Belgium",
            "default_lead_time_weeks": 4,
            "default_transit_weeks": 2,
            "payment_terms": "NET30",
            "currency": "USD",
            "supplier_status": "Active",
            "behavior_type": "excellent",
        },
        {
            "supplier_id": "SUP-002",
            "supplier_name": "Nova Manufacturing",
            "country": "Germany",
            "default_lead_time_weeks": 5,
            "default_transit_weeks": 2,
            "payment_terms": "NET30",
            "currency": "USD",
            "supplier_status": "Active",
            "behavior_type": "excellent",
        },
        {
            "supplier_id": "SUP-003",
            "supplier_name": "Vertex Components",
            "country": "Netherlands",
            "default_lead_time_weeks": 5,
            "default_transit_weeks": 2,
            "payment_terms": "NET45",
            "currency": "USD",
            "supplier_status": "Active",
            "behavior_type": "reliable",
        },
        {
            "supplier_id": "SUP-004",
            "supplier_name": "PrimeTech Supply",
            "country": "United States",
            "default_lead_time_weeks": 3,
            "default_transit_weeks": 1,
            "payment_terms": "NET30",
            "currency": "USD",
            "supplier_status": "Active",
            "behavior_type": "reliable",
        },
        {
            "supplier_id": "SUP-005",
            "supplier_name": "Global Source Systems",
            "country": "China",
            "default_lead_time_weeks": 6,
            "default_transit_weeks": 4,
            "payment_terms": "NET45",
            "currency": "USD",
            "supplier_status": "Active",
            "behavior_type": "frequently_late",
        },
        {
            "supplier_id": "SUP-006",
            "supplier_name": "EastBridge Manufacturing",
            "country": "Vietnam",
            "default_lead_time_weeks": 6,
            "default_transit_weeks": 4,
            "payment_terms": "NET30",
            "currency": "USD",
            "supplier_status": "Active",
            "behavior_type": "frequently_late",
        },
        {
            "supplier_id": "SUP-007",
            "supplier_name": "Precision Parts Group",
            "country": "Mexico",
            "default_lead_time_weeks": 4,
            "default_transit_weeks": 1,
            "payment_terms": "NET30",
            "currency": "USD",
            "supplier_status": "Active",
            "behavior_type": "partial_shipments",
        },
        {
            "supplier_id": "SUP-008",
            "supplier_name": "Titan Industrial",
            "country": "China",
            "default_lead_time_weeks": 7,
            "default_transit_weeks": 4,
            "payment_terms": "NET45",
            "currency": "USD",
            "supplier_status": "Active",
            "behavior_type": "partial_and_late",
        },
        {
            "supplier_id": "SUP-009",
            "supplier_name": "Dynamic Supply Co",
            "country": "India",
            "default_lead_time_weeks": 6,
            "default_transit_weeks": 3,
            "payment_terms": "NET30",
            "currency": "USD",
            "supplier_status": "Active",
            "behavior_type": "unpredictable",
        },
        {
            "supplier_id": "SUP-010",
            "supplier_name": "Nordic Components",
            "country": "Sweden",
            "default_lead_time_weeks": 8,
            "default_transit_weeks": 2,
            "payment_terms": "NET45",
            "currency": "USD",
            "supplier_status": "Active",
            "behavior_type": "long_but_consistent",
        },
    ]

    return pd.DataFrame(suppliers)


# =========================================================
# PRODUCT GENERATION
# =========================================================

def generate_products(suppliers_df):

    categories = {
        "Equipment": (1500, 12000),
        "Accessories": (20, 500),
        "Consumables": (5, 150),
        "Spare Parts": (10, 750),
        "Electronics": (50, 1200),
    }

    demand_profiles = [
        "stable",
        "volatile",
        "seasonal",
        "growth",
        "declining",
        "intermittent",
        "new_product",
    ]

    demand_profile_weights = [
        0.30,
        0.15,
        0.15,
        0.10,
        0.10,
        0.15,
        0.05,
    ]

    service_levels = [
        0.90,
        0.95,
        0.97,
        0.99,
    ]

    service_level_weights = [
        0.10,
        0.55,
        0.25,
        0.10,
    ]

    moq_options = [
        1,
        5,
        10,
        20,
        25,
        50,
    ]

    case_pack_options = [
        1,
        5,
        10,
        20,
    ]

    products = []

    for i in range(1, NUM_SKUS + 1):

        sku = f"SKU-{i:03d}"

        # -----------------------------------------
        # CATEGORY
        # -----------------------------------------

        category = random.choice(
            list(categories.keys())
        )

        cost_low, cost_high = categories[category]

        # -----------------------------------------
        # COST
        # -----------------------------------------

        unit_cost = round(
            random.uniform(
                cost_low,
                cost_high
            ),
            2
        )

        # -----------------------------------------
        # SELLING PRICE
        # -----------------------------------------

        markup = random.uniform(
            1.25,
            1.80
        )

        unit_price = round(
            unit_cost * markup,
            2
        )

        # -----------------------------------------
        # SUPPLIER
        # -----------------------------------------

        supplier = suppliers_df.sample(
            n=1,
            random_state=RANDOM_SEED + i
        ).iloc[0]

        supplier_id = supplier["supplier_id"]

        # -----------------------------------------
        # LEAD TIME
        # -----------------------------------------

        lead_adjustment = random.choice(
            [-1, 0, 0, 0, 1]
        )

        quoted_lead_time = max(
            1,
            int(
                supplier[
                    "default_lead_time_weeks"
                ]
            )
            + lead_adjustment
        )

        transit_time = int(
            supplier[
                "default_transit_weeks"
            ]
        )

        # -----------------------------------------
        # PURCHASING RULES
        # -----------------------------------------

        case_pack = random.choice(
            case_pack_options
        )

        moq = random.choice(
            moq_options
        )

        # Make sure MOQ is compatible
        # with the case-pack size.
        if moq % case_pack != 0:

            moq = (
                (moq // case_pack) + 1
            ) * case_pack

        # -----------------------------------------
        # DEMAND PROFILE
        # -----------------------------------------

        demand_profile = random.choices(
            demand_profiles,
            weights=demand_profile_weights,
            k=1
        )[0]

        # -----------------------------------------
        # SERVICE LEVEL
        # -----------------------------------------

        service_level = random.choices(
            service_levels,
            weights=service_level_weights,
            k=1
        )[0]

        # -----------------------------------------
        # CREATE PRODUCT
        # -----------------------------------------

        product = {
            "sku": sku,
            "product_name": (
                f"{category} Product {i:03d}"
            ),
            "category": category,
            "supplier_id": supplier_id,
            "unit_cost": unit_cost,
            "unit_price": unit_price,
            "moq": moq,
            "case_pack": case_pack,
            "quoted_lead_time_weeks": (
                quoted_lead_time
            ),
            "transit_time_weeks": transit_time,
            "service_level_target": service_level,
            "demand_profile": demand_profile,
            "active": True,
        }

        products.append(product)

    return pd.DataFrame(products)


# =========================================================
# DATA VALIDATION
# =========================================================

def validate_products(products_df, suppliers_df):

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # Check row count
    assert len(products_df) == NUM_SKUS, (
        "Incorrect number of products."
    )

    # Check unique SKUs
    assert products_df["sku"].is_unique, (
        "Duplicate SKUs detected."
    )

    # Check suppliers
    valid_suppliers = set(
        suppliers_df["supplier_id"]
    )

    product_suppliers = set(
        products_df["supplier_id"]
    )

    assert product_suppliers.issubset(
        valid_suppliers
    ), "Invalid supplier detected."

    # Cost must be positive
    assert (
        products_df["unit_cost"] > 0
    ).all(), "Invalid product cost detected."

    # Selling price should exceed cost
    assert (
        products_df["unit_price"]
        > products_df["unit_cost"]
    ).all(), "Selling price below cost detected."

    # MOQ must work with case pack
    assert (
        products_df["moq"]
        % products_df["case_pack"]
        == 0
    ).all(), "MOQ / case-pack conflict detected."

    # Lead times must be positive
    assert (
        products_df["quoted_lead_time_weeks"]
        > 0
    ).all(), "Invalid lead time detected."

    print("✓ Product count correct")
    print("✓ SKUs are unique")
    print("✓ Supplier relationships valid")
    print("✓ Product costs valid")
    print("✓ Selling prices valid")
    print("✓ MOQ / case-pack rules valid")
    print("✓ Lead times valid")

    print("\nAll product validation checks passed.")


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("PROCUREMENT AI - SYNTHETIC DATA GENERATOR")
    print("=" * 60)

    # -----------------------------------------------------
    # GENERATE SUPPLIERS
    # -----------------------------------------------------

    suppliers_df = generate_suppliers()

    supplier_file = (
        RAW_DATA_DIR / "suppliers.csv"
    )

    suppliers_df.to_csv(
        supplier_file,
        index=False
    )

    print(
        f"\nCreated: {supplier_file}"
    )

    print(
        f"Suppliers generated: "
        f"{len(suppliers_df)}"
    )

    # -----------------------------------------------------
    # GENERATE PRODUCTS
    # -----------------------------------------------------

    products_df = generate_products(
        suppliers_df
    )

    product_file = (
        RAW_DATA_DIR / "products.csv"
    )

    products_df.to_csv(
        product_file,
        index=False
    )

    print(
        f"\nCreated: {product_file}"
    )

    print(
        f"Products generated: "
        f"{len(products_df)}"
    )

    # -----------------------------------------------------
    # VALIDATE
    # -----------------------------------------------------

    validate_products(
        products_df,
        suppliers_df
    )

    # -----------------------------------------------------
    # PREVIEW
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("PRODUCT PREVIEW")
    print("=" * 60)

    preview_columns = [
        "sku",
        "category",
        "supplier_id",
        "unit_cost",
        "moq",
        "case_pack",
        "quoted_lead_time_weeks",
        "demand_profile",
    ]

    print(
        products_df[
            preview_columns
        ].head(10)
    )

    # -----------------------------------------------------
    # DEMAND PROFILE DISTRIBUTION
    # -----------------------------------------------------

    print("\nDemand Profile Distribution:")

    print(
        products_df[
            "demand_profile"
        ].value_counts()
    )

    # -----------------------------------------------------
    # CATEGORY DISTRIBUTION
    # -----------------------------------------------------

    print("\nCategory Distribution:")

    print(
        products_df[
            "category"
        ].value_counts()
    )

    # -----------------------------------------------------
    # SUPPLIER PRODUCT COUNTS
    # -----------------------------------------------------

    print("\nProducts per Supplier:")

    print(
        products_df[
            "supplier_id"
        ].value_counts().sort_index()
    )

    print("\n" + "=" * 60)
    print("PHASE 2C COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()