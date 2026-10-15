import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

RANDOM_SEED = 42
SNAPSHOT_DATE = pd.Timestamp("2026-09-30")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PRODUCT_FILE = RAW_DATA_DIR / "products.csv"
SALES_FILE = RAW_DATA_DIR / "sales_history.csv"
SUPPLIER_FILE = RAW_DATA_DIR / "suppliers.csv"

INVENTORY_FILE = RAW_DATA_DIR / "inventory.csv"
OPEN_PO_FILE = RAW_DATA_DIR / "open_purchase_orders.csv"


# =========================================================
# HELPERS
# =========================================================

def round_to_case_pack(quantity, case_pack):

    if quantity <= 0:
        return 0

    return int(
        math.ceil(quantity / case_pack)
        * case_pack
    )


def calculate_recent_demand(sales_df):

    """
    Calculate recent weekly demand for every SKU.

    We use the most recent 13 weeks because this
    gives us a reasonable short-term planning signal.
    """

    sales = sales_df.copy()

    sales["week_date"] = pd.to_datetime(
        sales["week_date"]
    )

    recent_start = (
        sales["week_date"].max()
        - pd.Timedelta(weeks=12)
    )

    recent = sales[
        sales["week_date"] >= recent_start
    ].copy()

    recent_summary = (
        recent
        .groupby("sku")
        .agg(
            avg_weekly_demand=(
                "units_demanded",
                "mean"
            ),
            demand_std=(
                "units_demanded",
                "std"
            ),
            recent_demand=(
                "units_demanded",
                "sum"
            ),
        )
        .reset_index()
    )

    recent_summary[
        "avg_weekly_demand"
    ] = (
        recent_summary[
            "avg_weekly_demand"
        ]
        .fillna(0)
    )

    recent_summary[
        "demand_std"
    ] = (
        recent_summary[
            "demand_std"
        ]
        .fillna(0)
    )

    recent_summary[
        "recent_demand"
    ] = (
        recent_summary[
            "recent_demand"
        ]
        .fillna(0)
    )

    return recent_summary


# =========================================================
# ASSIGN INVENTORY SCENARIOS
# =========================================================

def assign_inventory_scenarios(products_df):

    """
    We intentionally create different planning
    situations so the future engine has problems
    to identify and solve.
    """

    scenarios = [
        "healthy",
        "low_stock",
        "critical",
        "overstock",
        "backordered",
    ]

    weights = [
        0.40,
        0.22,
        0.15,
        0.15,
        0.08,
    ]

    scenario_map = {}

    for sku in products_df["sku"]:

        scenario_map[sku] = random.choices(
            scenarios,
            weights=weights,
            k=1
        )[0]

    return scenario_map


# =========================================================
# GENERATE INVENTORY
# =========================================================

def generate_inventory(
    products_df,
    sales_df
):

    demand_summary = (
        calculate_recent_demand(
            sales_df
        )
    )

    planning_df = (
        products_df.merge(
            demand_summary,
            on="sku",
            how="left"
        )
    )

    planning_df[
        "avg_weekly_demand"
    ] = (
        planning_df[
            "avg_weekly_demand"
        ]
        .fillna(0)
    )

    scenario_map = (
        assign_inventory_scenarios(
            products_df
        )
    )

    records = []

    for _, product in planning_df.iterrows():

        sku = product["sku"]

        avg_weekly_demand = float(
            product["avg_weekly_demand"]
        )

        scenario = scenario_map[sku]

        # -----------------------------------------
        # TARGET WEEKS OF INVENTORY
        # -----------------------------------------

        if scenario == "healthy":

            weeks_of_supply = random.uniform(
                6,
                12
            )

        elif scenario == "low_stock":

            weeks_of_supply = random.uniform(
                2,
                5
            )

        elif scenario == "critical":

            weeks_of_supply = random.uniform(
                0.2,
                1.8
            )

        elif scenario == "overstock":

            weeks_of_supply = random.uniform(
                18,
                35
            )

        elif scenario == "backordered":

            weeks_of_supply = random.uniform(
                0,
                0.5
            )

        else:

            raise ValueError(
                f"Unknown scenario: {scenario}"
            )

        # -----------------------------------------
        # ON HAND
        # -----------------------------------------

        if avg_weekly_demand > 0:

            on_hand = int(
                round(
                    avg_weekly_demand
                    * weeks_of_supply
                )
            )

        else:

            on_hand = random.randint(
                0,
                20
            )

        on_hand = max(
            0,
            on_hand
        )

        # -----------------------------------------
        # ALLOCATED
        # -----------------------------------------

        if on_hand > 0:

            if scenario in [
                "critical",
                "backordered"
            ]:

                allocation_rate = (
                    random.uniform(
                        0.20,
                        0.70
                    )
                )

            else:

                allocation_rate = (
                    random.uniform(
                        0.00,
                        0.25
                    )
                )

            allocated = int(
                round(
                    on_hand
                    * allocation_rate
                )
            )

        else:

            allocated = 0

        allocated = min(
            allocated,
            on_hand
        )

        # -----------------------------------------
        # AVAILABLE
        # -----------------------------------------

        available = (
            on_hand
            - allocated
        )

        # -----------------------------------------
        # BACKORDER
        # -----------------------------------------

        if scenario == "backordered":

            backordered = max(
                1,
                int(
                    round(
                        max(
                            avg_weekly_demand,
                            1
                        )
                        * random.uniform(
                            0.5,
                            2.0
                        )
                    )
                )
            )

        else:

            backordered = 0

        # -----------------------------------------
        # INVENTORY VALUE
        # -----------------------------------------

        unit_cost = float(
            product["unit_cost"]
        )

        inventory_value = round(
            on_hand * unit_cost,
            2
        )

        # -----------------------------------------
        # RECORD
        # -----------------------------------------

        records.append(
            {
                "snapshot_date": SNAPSHOT_DATE,
                "sku": sku,
                "warehouse": "WH-01",
                "on_hand": on_hand,
                "allocated": allocated,
                "available": available,
                "backordered": backordered,
                "inventory_value": inventory_value,
                "scenario_label": scenario,
            }
        )

    return pd.DataFrame(records)


# =========================================================
# OPEN PO SCENARIOS
# =========================================================

def get_open_po_scenario():

    scenarios = [
        "normal",
        "late",
        "partial",
        "partial_late",
    ]

    weights = [
        0.48,
        0.20,
        0.18,
        0.14,
    ]

    return random.choices(
        scenarios,
        weights=weights,
        k=1
    )[0]


# =========================================================
# GENERATE CURRENT OPEN POS
# =========================================================

def generate_open_purchase_orders(
    products_df,
    inventory_df
):

    inventory_lookup = (
        inventory_df
        .set_index("sku")
        .to_dict("index")
    )

    records = []

    po_counter = 1

    for _, product in products_df.iterrows():

        sku = product["sku"]

        inventory = inventory_lookup[sku]

        scenario_label = (
            inventory["scenario_label"]
        )

        # -----------------------------------------
        # DECIDE WHETHER SKU HAS OPEN PO
        # -----------------------------------------

        if scenario_label in [
            "critical",
            "backordered",
            "low_stock",
        ]:

            open_po_probability = 0.75

        elif scenario_label == "healthy":

            open_po_probability = 0.40

        else:

            open_po_probability = 0.15

        if random.random() > open_po_probability:
            continue

        # Some SKUs can have more than one open PO.
        number_of_pos = (
            2
            if random.random() < 0.15
            else 1
        )

        for _ in range(number_of_pos):

            supplier_id = (
                product["supplier_id"]
            )

            case_pack = int(
                product["case_pack"]
            )

            moq = int(
                product["moq"]
            )

            avg_reference_qty = max(
                moq,
                case_pack
                * random.randint(
                    2,
                    10
                )
            )

            ordered_qty = (
                round_to_case_pack(
                    avg_reference_qty,
                    case_pack
                )
            )

            po_scenario = (
                get_open_po_scenario()
            )

            lead_time = int(
                product[
                    "quoted_lead_time_weeks"
                ]
            )

            transit_time = int(
                product[
                    "transit_time_weeks"
                ]
            )

            total_weeks = (
                lead_time
                + transit_time
            )

            # -------------------------------------
            # ORDER DATE
            # -------------------------------------

            if po_scenario in [
                "late",
                "partial_late",
            ]:

                weeks_ago = (
                    total_weeks
                    + random.randint(
                        1,
                        5
                    )
                )

            else:

                weeks_ago = random.randint(
                    1,
                    max(
                        1,
                        total_weeks - 1
                    )
                )

            order_date = (
                SNAPSHOT_DATE
                - pd.Timedelta(
                    weeks=weeks_ago
                )
            )

            promised_date = (
                order_date
                + pd.Timedelta(
                    weeks=total_weeks
                )
            )

            # -------------------------------------
            # RECEIVED QUANTITY
            # -------------------------------------

            if po_scenario in [
                "partial",
                "partial_late",
            ]:

                received_fraction = (
                    random.uniform(
                        0.15,
                        0.75
                    )
                )

                received_qty = int(
                    round(
                        ordered_qty
                        * received_fraction
                    )
                )

                received_qty = min(
                    received_qty,
                    ordered_qty - 1
                )

                received_qty = max(
                    1,
                    received_qty
                )

            else:

                received_qty = 0

            # -------------------------------------
            # OPEN QUANTITY
            # -------------------------------------

            open_qty = (
                ordered_qty
                - received_qty
            )

            # -------------------------------------
            # EXPECTED DATE
            # -------------------------------------

            if po_scenario == "normal":

                expected_date = (
                    promised_date
                )

            elif po_scenario == "late":

                expected_date = (
                    SNAPSHOT_DATE
                    + pd.Timedelta(
                        days=random.randint(
                            5,
                            28
                        )
                    )
                )

            elif po_scenario == "partial":

                expected_date = (
                    promised_date
                    + pd.Timedelta(
                        days=random.randint(
                            3,
                            14
                        )
                    )
                )

            elif po_scenario == "partial_late":

                expected_date = (
                    SNAPSHOT_DATE
                    + pd.Timedelta(
                        days=random.randint(
                            7,
                            35
                        )
                    )
                )

            else:

                raise ValueError(
                    f"Unknown PO scenario: "
                    f"{po_scenario}"
                )

            # -------------------------------------
            # STATUS
            # -------------------------------------

            if received_qty > 0:

                status = (
                    "Partially Received"
                )

            else:

                status = "Open"

            # -------------------------------------
            # DAYS LATE
            # -------------------------------------

            days_late = max(
                0,
                (
                    SNAPSHOT_DATE
                    - promised_date
                ).days
            )

            # -------------------------------------
            # RECORD
            # -------------------------------------

            records.append(
                {
                    "po_number": (
                        f"OPEN-PO-"
                        f"{po_counter:04d}"
                    ),
                    "po_line": 1,
                    "supplier_id": (
                        supplier_id
                    ),
                    "sku": sku,
                    "order_date": (
                        order_date
                    ),
                    "ordered_qty": (
                        ordered_qty
                    ),
                    "received_qty": (
                        received_qty
                    ),
                    "open_qty": (
                        open_qty
                    ),
                    "promised_date": (
                        promised_date
                    ),
                    "expected_date": (
                        expected_date
                    ),
                    "status": status,
                    "days_late": (
                        days_late
                    ),
                    "open_po_scenario": (
                        po_scenario
                    ),
                }
            )

            po_counter += 1

    return pd.DataFrame(records)


# =========================================================
# VALIDATE INVENTORY
# =========================================================

def validate_inventory(
    inventory_df,
    products_df
):

    print("\n" + "=" * 60)
    print("INVENTORY VALIDATION")
    print("=" * 60)

    assert (
        len(inventory_df)
        ==
        len(products_df)
    )

    assert (
        inventory_df["sku"].is_unique
    )

    assert (
        inventory_df["on_hand"] >= 0
    ).all()

    assert (
        inventory_df["allocated"] >= 0
    ).all()

    assert (
        inventory_df["available"] >= 0
    ).all()

    assert (
        inventory_df["backordered"] >= 0
    ).all()

    assert (
        inventory_df["available"]
        ==
        inventory_df["on_hand"]
        -
        inventory_df["allocated"]
    ).all()

    print(
        f"✓ Inventory rows correct: "
        f"{len(inventory_df)}"
    )

    print(
        "✓ One inventory record per SKU"
    )

    print(
        "✓ Inventory quantities are non-negative"
    )

    print(
        "✓ Available = On Hand - Allocated"
    )

    print(
        "\nAll inventory validation "
        "checks passed."
    )


# =========================================================
# VALIDATE OPEN POS
# =========================================================

def validate_open_pos(
    open_po_df
):

    print("\n" + "=" * 60)
    print("OPEN PO VALIDATION")
    print("=" * 60)

    assert (
        open_po_df[
            "po_number"
        ].is_unique
    )

    assert (
        open_po_df[
            "ordered_qty"
        ] > 0
    ).all()

    assert (
        open_po_df[
            "received_qty"
        ] >= 0
    ).all()

    assert (
        open_po_df[
            "open_qty"
        ] > 0
    ).all()

    assert (
        open_po_df[
            "received_qty"
        ]
        +
        open_po_df[
            "open_qty"
        ]
        ==
        open_po_df[
            "ordered_qty"
        ]
    ).all()

    assert (
        open_po_df[
            "days_late"
        ] >= 0
    ).all()

    print(
        f"✓ Open PO records: "
        f"{len(open_po_df)}"
    )

    print(
        "✓ Open PO numbers are unique"
    )

    print(
        "✓ Open quantities are positive"
    )

    print(
        "✓ Received + Open = Ordered"
    )

    print(
        "✓ Days late are valid"
    )

    print(
        "\nAll open-PO validation "
        "checks passed."
    )


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(
    inventory_df,
    open_po_df
):

    print("\n" + "=" * 80)
    print("CURRENT PLANNING SNAPSHOT")
    print("=" * 80)

    print(
        f"\nSnapshot Date: "
        f"{SNAPSHOT_DATE.date()}"
    )

    print("\nInventory Scenario Distribution:")

    print(
        inventory_df[
            "scenario_label"
        ]
        .value_counts()
    )

    print("\nOpen PO Scenario Distribution:")

    print(
        open_po_df[
            "open_po_scenario"
        ]
        .value_counts()
    )

    print("\nOpen PO Status:")

    print(
        open_po_df[
            "status"
        ]
        .value_counts()
    )

    total_inventory_value = (
        inventory_df[
            "inventory_value"
        ]
        .sum()
    )

    total_open_units = (
        open_po_df[
            "open_qty"
        ]
        .sum()
    )

    late_pos = (
        open_po_df[
            open_po_df[
                "days_late"
            ] > 0
        ]
    )

    print(
        f"\nTotal Inventory Value: "
        f"${total_inventory_value:,.2f}"
    )

    print(
        f"Total Open PO Units: "
        f"{total_open_units:,.0f}"
    )

    print(
        f"Late Open POs: "
        f"{len(late_pos)}"
    )


# =========================================================
# PRINT RISK EXAMPLES
# =========================================================

def print_examples(
    inventory_df,
    open_po_df
):

    print("\n" + "=" * 80)
    print("EXAMPLE CURRENT RISKS")
    print("=" * 80)

    risky_inventory = (
        inventory_df[
            inventory_df[
                "scenario_label"
            ].isin(
                [
                    "critical",
                    "backordered"
                ]
            )
        ]
        .head(5)
    )

    for _, row in (
        risky_inventory.iterrows()
    ):

        sku = row["sku"]

        print(
            f"\nSKU: {sku}"
        )

        print(
            f"Scenario: "
            f"{row['scenario_label']}"
        )

        print(
            f"On Hand: "
            f"{row['on_hand']}"
        )

        print(
            f"Allocated: "
            f"{row['allocated']}"
        )

        print(
            f"Available: "
            f"{row['available']}"
        )

        print(
            f"Backordered: "
            f"{row['backordered']}"
        )

        sku_pos = (
            open_po_df[
                open_po_df[
                    "sku"
                ] == sku
            ]
        )

        if sku_pos.empty:

            print(
                "Open PO: None"
            )

        else:

            for _, po in (
                sku_pos.iterrows()
            ):

                print(
                    f"Open PO: "
                    f"{po['po_number']} | "
                    f"Open Qty: "
                    f"{po['open_qty']} | "
                    f"Promised: "
                    f"{po['promised_date']} | "
                    f"Expected: "
                    f"{po['expected_date']} | "
                    f"Days Late: "
                    f"{po['days_late']}"
                )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)

    print(
        "PROCUREMENT AI - "
        "CURRENT SNAPSHOT GENERATOR"
    )

    print("=" * 60)

    # -----------------------------------------
    # CHECK INPUTS
    # -----------------------------------------

    required_files = [
        PRODUCT_FILE,
        SALES_FILE,
        SUPPLIER_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing required file: "
                f"{file_path}"
            )

    # -----------------------------------------
    # LOAD DATA
    # -----------------------------------------

    products_df = pd.read_csv(
        PRODUCT_FILE
    )

    sales_df = pd.read_csv(
        SALES_FILE
    )

    suppliers_df = pd.read_csv(
        SUPPLIER_FILE
    )

    print(
        f"\nProducts loaded: "
        f"{len(products_df)}"
    )

    print(
        f"Sales rows loaded: "
        f"{len(sales_df):,}"
    )

    print(
        f"Suppliers loaded: "
        f"{len(suppliers_df)}"
    )

    # -----------------------------------------
    # INVENTORY
    # -----------------------------------------

    inventory_df = (
        generate_inventory(
            products_df,
            sales_df
        )
    )

    # -----------------------------------------
    # OPEN PURCHASE ORDERS
    # -----------------------------------------

    open_po_df = (
        generate_open_purchase_orders(
            products_df,
            inventory_df
        )
    )

    # -----------------------------------------
    # VALIDATION
    # -----------------------------------------

    validate_inventory(
        inventory_df,
        products_df
    )

    validate_open_pos(
        open_po_df
    )

    # -----------------------------------------
    # SAVE
    # -----------------------------------------

    inventory_df.to_csv(
        INVENTORY_FILE,
        index=False
    )

    open_po_df.to_csv(
        OPEN_PO_FILE,
        index=False
    )

    print(
        f"\nCreated: "
        f"{INVENTORY_FILE}"
    )

    print(
        f"Created: "
        f"{OPEN_PO_FILE}"
    )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    print_summary(
        inventory_df,
        open_po_df
    )

    # -----------------------------------------
    # EXAMPLES
    # -----------------------------------------

    print_examples(
        inventory_df,
        open_po_df
    )

    # -----------------------------------------
    # COMPLETE
    # -----------------------------------------

    print("\n" + "=" * 60)

    print(
        "PHASE 2F COMPLETE"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()