import random
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# SETTINGS
# =========================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

HISTORY_START = pd.Timestamp("2024-10-01")
HISTORY_END = pd.Timestamp("2026-09-30")

TARGET_PO_LINES = 2000


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PRODUCT_FILE = RAW_DATA_DIR / "products.csv"
SUPPLIER_FILE = RAW_DATA_DIR / "suppliers.csv"

PO_FILE = RAW_DATA_DIR / "purchase_orders.csv"
RECEIPT_FILE = RAW_DATA_DIR / "po_receipts.csv"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def round_to_case_pack(quantity, case_pack):

    quantity = max(1, quantity)

    return int(
        math.ceil(quantity / case_pack)
        * case_pack
    )


# =========================================================
# SUPPLIER DELIVERY BEHAVIOR
# =========================================================

def get_delivery_behavior(behavior_type):

    behavior_rules = {

        "excellent": {
            "delay_mean": 0,
            "delay_std": 2,
            "partial_probability": 0.05,
            "max_shipments": 2,
        },

        "reliable": {
            "delay_mean": 1,
            "delay_std": 3,
            "partial_probability": 0.12,
            "max_shipments": 2,
        },

        "frequently_late": {
            "delay_mean": 10,
            "delay_std": 6,
            "partial_probability": 0.20,
            "max_shipments": 3,
        },

        "partial_shipments": {
            "delay_mean": 2,
            "delay_std": 4,
            "partial_probability": 0.70,
            "max_shipments": 4,
        },

        "partial_and_late": {
            "delay_mean": 12,
            "delay_std": 8,
            "partial_probability": 0.80,
            "max_shipments": 4,
        },

        "unpredictable": {
            "delay_mean": 7,
            "delay_std": 14,
            "partial_probability": 0.55,
            "max_shipments": 5,
        },

        "long_but_consistent": {
            "delay_mean": 1,
            "delay_std": 2,
            "partial_probability": 0.08,
            "max_shipments": 2,
        },
    }

    if behavior_type not in behavior_rules:

        raise ValueError(
            f"Unknown supplier behavior: "
            f"{behavior_type}"
        )

    return behavior_rules[
        behavior_type
    ]


# =========================================================
# GENERATE PURCHASE ORDERS
# =========================================================

def generate_purchase_orders(
    products_df,
    suppliers_df
):

    supplier_behavior = (
        suppliers_df
        .set_index("supplier_id")[
            "behavior_type"
        ]
        .to_dict()
    )

    po_records = []

    date_range_days = (
        HISTORY_END - HISTORY_START
    ).days

    for i in range(
        1,
        TARGET_PO_LINES + 1
    ):

        # -----------------------------------------
        # SELECT PRODUCT
        # -----------------------------------------

        product = products_df.sample(
            n=1,
            random_state=RANDOM_SEED + i
        ).iloc[0]

        sku = product["sku"]

        supplier_id = (
            product["supplier_id"]
        )

        case_pack = int(
            product["case_pack"]
        )

        moq = int(
            product["moq"]
        )

        unit_cost = float(
            product["unit_cost"]
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

        # -----------------------------------------
        # ORDER DATE
        # -----------------------------------------

        random_days = random.randint(
            0,
            date_range_days
        )

        order_date = (
            HISTORY_START
            + pd.Timedelta(
                days=random_days
            )
        )

        # -----------------------------------------
        # ORDER QUANTITY
        # -----------------------------------------

        multiplier = random.randint(
            1,
            8
        )

        ordered_qty = max(
            moq,
            case_pack * multiplier
        )

        ordered_qty = (
            round_to_case_pack(
                ordered_qty,
                case_pack
            )
        )

        # -----------------------------------------
        # PROMISED DATE
        # -----------------------------------------

        total_planned_weeks = (
            lead_time
            + transit_time
        )

        promised_date = (
            order_date
            + pd.Timedelta(
                weeks=total_planned_weeks
            )
        )

        # -----------------------------------------
        # SUPPLIER BEHAVIOR
        # -----------------------------------------

        behavior_type = (
            supplier_behavior[
                supplier_id
            ]
        )

        # -----------------------------------------
        # PO RECORD
        # -----------------------------------------

        po_records.append(
            {
                "po_number": (
                    f"PO-{i:05d}"
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
                "unit_cost": (
                    unit_cost
                ),
                "po_value": round(
                    ordered_qty
                    * unit_cost,
                    2
                ),
                "promised_date": (
                    promised_date
                ),
                "behavior_type": (
                    behavior_type
                ),
            }
        )

    return pd.DataFrame(
        po_records
    )


# =========================================================
# SAFE QUANTITY SPLITTING
# =========================================================

def split_quantity(
    total_qty,
    shipment_count
):

    total_qty = int(
        total_qty
    )

    shipment_count = int(
        shipment_count
    )

    # -----------------------------------------
    # VALIDATE INPUT
    # -----------------------------------------

    if total_qty <= 0:

        raise ValueError(
            "Total quantity must "
            "be greater than zero."
        )

    # Cannot create more positive shipments
    # than units available.
    shipment_count = min(
        shipment_count,
        total_qty
    )

    # -----------------------------------------
    # SINGLE SHIPMENT
    # -----------------------------------------

    if shipment_count <= 1:

        return [
            total_qty
        ]

    # -----------------------------------------
    # RANDOM SAFE SPLIT
    # -----------------------------------------

    # Example:
    #
    # total_qty = 100
    # shipment_count = 4
    #
    # Random cut points:
    #
    # 0 -- 20 -- 55 -- 80 -- 100
    #
    # Shipments:
    #
    # 20, 35, 25, 20

    cut_points = sorted(
        random.sample(
            range(
                1,
                total_qty
            ),
            shipment_count - 1
        )
    )

    boundaries = (
        [0]
        + cut_points
        + [total_qty]
    )

    shipments = []

    for i in range(
        len(boundaries) - 1
    ):

        qty = (
            boundaries[i + 1]
            - boundaries[i]
        )

        shipments.append(
            qty
        )

    # -----------------------------------------
    # INTERNAL CHECK
    # -----------------------------------------

    assert sum(shipments) == total_qty

    assert all(
        qty > 0
        for qty in shipments
    )

    return shipments


# =========================================================
# GENERATE RECEIPTS
# =========================================================

def generate_receipts(
    purchase_orders_df
):

    receipt_records = []

    receipt_counter = 1

    for _, po in (
        purchase_orders_df.iterrows()
    ):

        behavior = (
            get_delivery_behavior(
                po["behavior_type"]
            )
        )

        ordered_qty = int(
            po["ordered_qty"]
        )

        promised_date = (
            pd.Timestamp(
                po["promised_date"]
            )
        )

        # -----------------------------------------
        # FIRST DELIVERY DELAY
        # -----------------------------------------

        delay_days = int(
            round(
                np.random.normal(
                    behavior[
                        "delay_mean"
                    ],
                    behavior[
                        "delay_std"
                    ]
                )
            )
        )

        # Suppliers can occasionally
        # arrive early, but no more
        # than one week early.
        delay_days = max(
            -7,
            delay_days
        )

        first_receipt_date = (
            promised_date
            + pd.Timedelta(
                days=delay_days
            )
        )

        # -----------------------------------------
        # PARTIAL SHIPMENT DECISION
        # -----------------------------------------

        is_partial = (
            random.random()
            <
            behavior[
                "partial_probability"
            ]
        )

        # -----------------------------------------
        # NUMBER OF SHIPMENTS
        # -----------------------------------------

        if (
            is_partial
            and ordered_qty >= 2
        ):

            max_possible_shipments = min(
                int(
                    behavior[
                        "max_shipments"
                    ]
                ),
                ordered_qty
            )

            shipment_count = (
                random.randint(
                    2,
                    max_possible_shipments
                )
            )

        else:

            shipment_count = 1

        # -----------------------------------------
        # SPLIT ORDER QUANTITY
        # -----------------------------------------

        shipment_quantities = (
            split_quantity(
                ordered_qty,
                shipment_count
            )
        )

        # -----------------------------------------
        # CREATE RECEIPTS
        # -----------------------------------------

        current_receipt_date = (
            first_receipt_date
        )

        for (
            shipment_number,
            qty
        ) in enumerate(
            shipment_quantities,
            start=1
        ):

            # Later partial shipments
            # arrive after the first one.
            if shipment_number > 1:

                gap_days = (
                    random.randint(
                        5,
                        18
                    )
                )

                current_receipt_date = (
                    current_receipt_date
                    + pd.Timedelta(
                        days=gap_days
                    )
                )

            receipt_records.append(
                {
                    "receipt_id": (
                        f"REC-"
                        f"{receipt_counter:06d}"
                    ),
                    "po_number": (
                        po["po_number"]
                    ),
                    "po_line": (
                        po["po_line"]
                    ),
                    "supplier_id": (
                        po["supplier_id"]
                    ),
                    "sku": (
                        po["sku"]
                    ),
                    "receipt_number": (
                        shipment_number
                    ),
                    "receipt_date": (
                        current_receipt_date
                    ),
                    "receipt_qty": (
                        qty
                    ),
                }
            )

            receipt_counter += 1

    return pd.DataFrame(
        receipt_records
    )


# =========================================================
# ADD PO STATUS AND RECEIPT INFORMATION
# =========================================================

def add_po_status(
    purchase_orders_df,
    receipts_df
):

    receipt_summary = (
        receipts_df
        .groupby(
            [
                "po_number",
                "po_line"
            ]
        )
        .agg(
            received_qty=(
                "receipt_qty",
                "sum"
            ),
            first_receipt_date=(
                "receipt_date",
                "min"
            ),
            final_receipt_date=(
                "receipt_date",
                "max"
            ),
            shipment_count=(
                "receipt_id",
                "count"
            ),
        )
        .reset_index()
    )

    po_df = (
        purchase_orders_df.merge(
            receipt_summary,
            on=[
                "po_number",
                "po_line"
            ],
            how="left"
        )
    )

    # -----------------------------------------
    # RECEIVED QUANTITY
    # -----------------------------------------

    po_df[
        "received_qty"
    ] = (
        po_df[
            "received_qty"
        ]
        .fillna(0)
        .astype(int)
    )

    # -----------------------------------------
    # OPEN QUANTITY
    # -----------------------------------------

    po_df[
        "open_qty"
    ] = (
        po_df[
            "ordered_qty"
        ]
        -
        po_df[
            "received_qty"
        ]
    )

    # -----------------------------------------
    # STATUS
    # -----------------------------------------

    po_df[
        "status"
    ] = np.where(

        po_df[
            "open_qty"
        ] == 0,

        "Closed",

        np.where(

            po_df[
                "received_qty"
            ] > 0,

            "Partially Received",

            "Open"
        )
    )

    return po_df


# =========================================================
# PROCUREMENT VALIDATION
# =========================================================

def validate_procurement(
    po_df,
    receipts_df
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PROCUREMENT VALIDATION"
    )

    print(
        "=" * 60
    )

    # -----------------------------------------
    # PO COUNT
    # -----------------------------------------

    assert (
        len(po_df)
        ==
        TARGET_PO_LINES
    ), "Incorrect PO count."

    # -----------------------------------------
    # UNIQUE PO NUMBERS
    # -----------------------------------------

    assert (
        po_df[
            "po_number"
        ].is_unique
    ), "Duplicate PO numbers detected."

    # -----------------------------------------
    # POSITIVE ORDER QUANTITY
    # -----------------------------------------

    assert (
        po_df[
            "ordered_qty"
        ] > 0
    ).all(), (
        "Invalid PO quantity detected."
    )

    # -----------------------------------------
    # POSITIVE RECEIPT QUANTITY
    # -----------------------------------------

    assert (
        receipts_df[
            "receipt_qty"
        ] > 0
    ).all(), (
        "Invalid receipt quantity detected."
    )

    # -----------------------------------------
    # RECEIPT RECONCILIATION
    # -----------------------------------------

    received_by_po = (
        receipts_df
        .groupby(
            "po_number"
        )[
            "receipt_qty"
        ]
        .sum()
    )

    ordered_by_po = (
        po_df
        .set_index(
            "po_number"
        )[
            "ordered_qty"
        ]
    )

    comparison = pd.concat(
        [
            ordered_by_po,
            received_by_po
        ],
        axis=1
    )

    comparison.columns = [
        "ordered",
        "received"
    ]

    assert (
        comparison[
            "ordered"
        ]
        ==
        comparison[
            "received"
        ]
    ).all(), (
        "PO/receipt quantity mismatch."
    )

    # -----------------------------------------
    # OPEN QUANTITY
    # -----------------------------------------

    assert (
        po_df[
            "open_qty"
        ] >= 0
    ).all(), (
        "Negative open quantity detected."
    )

    # -----------------------------------------
    # SHIPMENT COUNT
    # -----------------------------------------

    assert (
        po_df[
            "shipment_count"
        ] >= 1
    ).all(), (
        "Invalid shipment count detected."
    )

    # -----------------------------------------
    # RECEIPT DATES
    # -----------------------------------------

    assert (
        pd.to_datetime(
            po_df[
                "final_receipt_date"
            ]
        )
        >=
        pd.to_datetime(
            po_df[
                "first_receipt_date"
            ]
        )
    ).all(), (
        "Final receipt occurs before "
        "first receipt."
    )

    print(
        f"✓ PO lines correct: "
        f"{len(po_df):,}"
    )

    print(
        "✓ PO numbers are unique"
    )

    print(
        "✓ All order quantities are positive"
    )

    print(
        "✓ All receipt quantities are positive"
    )

    print(
        "✓ Receipt quantities reconcile "
        "to ordered quantities"
    )

    print(
        "✓ No negative open quantities"
    )

    print(
        "✓ Shipment counts are valid"
    )

    print(
        "✓ Receipt dates are valid"
    )

    print(
        "\nAll procurement validation "
        "checks passed."
    )


# =========================================================
# SUPPLIER BEHAVIOR SUMMARY
# =========================================================

def print_supplier_summary(
    po_df
):

    temp = po_df.copy()

    # -----------------------------------------
    # DATE CONVERSION
    # -----------------------------------------

    temp[
        "promised_date"
    ] = pd.to_datetime(
        temp[
            "promised_date"
        ]
    )

    temp[
        "first_receipt_date"
    ] = pd.to_datetime(
        temp[
            "first_receipt_date"
        ]
    )

    temp[
        "final_receipt_date"
    ] = pd.to_datetime(
        temp[
            "final_receipt_date"
        ]
    )

    # -----------------------------------------
    # FIRST RECEIPT DELAY
    # -----------------------------------------

    temp[
        "first_delay_days"
    ] = (
        temp[
            "first_receipt_date"
        ]
        -
        temp[
            "promised_date"
        ]
    ).dt.days

    # -----------------------------------------
    # FULL FULFILLMENT DELAY
    # -----------------------------------------

    temp[
        "full_delay_days"
    ] = (
        temp[
            "final_receipt_date"
        ]
        -
        temp[
            "promised_date"
        ]
    ).dt.days

    # -----------------------------------------
    # PARTIAL SHIPMENT FLAG
    # -----------------------------------------

    temp[
        "partial_flag"
    ] = (
        temp[
            "shipment_count"
        ] > 1
    )

    # -----------------------------------------
    # ON-TIME FIRST RECEIPT
    # -----------------------------------------

    temp[
        "first_receipt_on_time"
    ] = (
        temp[
            "first_receipt_date"
        ]
        <=
        temp[
            "promised_date"
        ]
    )

    # -----------------------------------------
    # FULL PO ON TIME
    # -----------------------------------------

    temp[
        "full_po_on_time"
    ] = (
        temp[
            "final_receipt_date"
        ]
        <=
        temp[
            "promised_date"
        ]
    )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    behavior_summary = (
        temp
        .groupby(
            [
                "supplier_id",
                "behavior_type"
            ]
        )
        .agg(
            po_count=(
                "po_number",
                "count"
            ),

            avg_first_delay_days=(
                "first_delay_days",
                "mean"
            ),

            avg_full_delay_days=(
                "full_delay_days",
                "mean"
            ),

            partial_shipment_rate=(
                "partial_flag",
                "mean"
            ),

            avg_shipments_per_po=(
                "shipment_count",
                "mean"
            ),

            first_receipt_on_time_rate=(
                "first_receipt_on_time",
                "mean"
            ),

            full_po_on_time_rate=(
                "full_po_on_time",
                "mean"
            ),
        )
        .round(2)
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SUPPLIER BEHAVIOR SUMMARY"
    )

    print(
        "=" * 100
    )

    print(
        behavior_summary.to_string()
    )


# =========================================================
# EXAMPLE PARTIAL SHIPMENTS
# =========================================================

def print_partial_examples(
    po_df,
    receipts_df
):

    partial_pos = (
        po_df[
            po_df[
                "shipment_count"
            ] > 1
        ]
        .sort_values(
            "shipment_count",
            ascending=False
        )
        .head(3)
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "EXAMPLE PARTIAL SHIPMENTS"
    )

    print(
        "=" * 80
    )

    if partial_pos.empty:

        print(
            "No partial shipments generated."
        )

        return

    for _, po in (
        partial_pos.iterrows()
    ):

        po_number = (
            po["po_number"]
        )

        print(
            f"\n{po_number}"
        )

        print(
            f"Supplier: "
            f"{po['supplier_id']}"
        )

        print(
            f"SKU: "
            f"{po['sku']}"
        )

        print(
            f"Ordered quantity: "
            f"{po['ordered_qty']}"
        )

        print(
            f"Promised date: "
            f"{po['promised_date']}"
        )

        po_receipts = (
            receipts_df[
                receipts_df[
                    "po_number"
                ] == po_number
            ]
            [
                [
                    "receipt_number",
                    "receipt_date",
                    "receipt_qty"
                ]
            ]
        )

        print(
            po_receipts.to_string(
                index=False
            )
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "=" * 60
    )

    print(
        "PROCUREMENT AI - "
        "PROCUREMENT GENERATOR"
    )

    print(
        "=" * 60
    )

    # -----------------------------------------
    # CHECK INPUT FILES
    # -----------------------------------------

    if not PRODUCT_FILE.exists():

        raise FileNotFoundError(
            "products.csv was not found. "
            "Run generate_data.py first."
        )

    if not SUPPLIER_FILE.exists():

        raise FileNotFoundError(
            "suppliers.csv was not found. "
            "Run generate_data.py first."
        )

    # -----------------------------------------
    # LOAD DATA
    # -----------------------------------------

    products_df = pd.read_csv(
        PRODUCT_FILE
    )

    suppliers_df = pd.read_csv(
        SUPPLIER_FILE
    )

    print(
        f"\nProducts loaded: "
        f"{len(products_df)}"
    )

    print(
        f"Suppliers loaded: "
        f"{len(suppliers_df)}"
    )

    # -----------------------------------------
    # GENERATE PURCHASE ORDERS
    # -----------------------------------------

    purchase_orders_df = (
        generate_purchase_orders(
            products_df,
            suppliers_df
        )
    )

    print(
        f"\nPurchase orders generated: "
        f"{len(purchase_orders_df):,}"
    )

    # -----------------------------------------
    # GENERATE RECEIPTS
    # -----------------------------------------

    receipts_df = (
        generate_receipts(
            purchase_orders_df
        )
    )

    print(
        f"Receipt transactions generated: "
        f"{len(receipts_df):,}"
    )

    # -----------------------------------------
    # ADD PO STATUS
    # -----------------------------------------

    purchase_orders_df = (
        add_po_status(
            purchase_orders_df,
            receipts_df
        )
    )

    # -----------------------------------------
    # VALIDATE
    # -----------------------------------------

    validate_procurement(
        purchase_orders_df,
        receipts_df
    )

    # -----------------------------------------
    # SAVE
    # -----------------------------------------

    purchase_orders_df.to_csv(
        PO_FILE,
        index=False
    )

    receipts_df.to_csv(
        RECEIPT_FILE,
        index=False
    )

    print(
        f"\nCreated: "
        f"{PO_FILE}"
    )

    print(
        f"Created: "
        f"{RECEIPT_FILE}"
    )

    # -----------------------------------------
    # SUPPLIER SUMMARY
    # -----------------------------------------

    print_supplier_summary(
        purchase_orders_df
    )

    # -----------------------------------------
    # PARTIAL SHIPMENT EXAMPLES
    # -----------------------------------------

    print_partial_examples(
        purchase_orders_df,
        receipts_df
    )

    # -----------------------------------------
    # COMPLETE
    # -----------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PHASE 2E COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()