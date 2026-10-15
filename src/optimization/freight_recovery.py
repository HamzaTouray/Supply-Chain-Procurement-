
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# ENHANCEMENT 1B - FREIGHT RECOVERY SCENARIO ANALYSIS
# =========================================================
# Uses validated inventory policy lead times for all SKUs.
#
# This is an additive, demonstration-only analysis.
# It does not change purchase proposals, planning outputs,
# approval records, or the SQLite database.
#
# Air freight prices and transit times are assumptions.
# Supplier readiness and actual delivery are unconfirmed.
# =========================================================


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

PLANNING_FILE = DATA_DIR / "planning_master.csv"
POLICY_FILE = DATA_DIR / "inventory_policy.csv"
BRIDGE_FILE = DATA_DIR / "bridge_quantity_analysis.csv"
OPEN_PO_FILE = DATA_DIR / "open_purchase_orders_clean.csv"

OUTPUT_FILE = DATA_DIR / "freight_recovery_scenarios.csv"


# =========================================================
# CONFIGURABLE DEMONSTRATION ASSUMPTIONS
# =========================================================

EXPEDITED_TRANSIT_WEEKS = 1.0

STANDARD_FREIGHT_COST_PER_UNIT = 2.00
EXPEDITED_FREIGHT_COST_PER_UNIT = 12.00

DAYS_PER_WEEK = 7


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    files = {
        "planning_master": PLANNING_FILE,
        "inventory_policy": POLICY_FILE,
        "bridge_quantity_analysis": BRIDGE_FILE,
        "open_purchase_orders_clean": OPEN_PO_FILE,
    }

    data = {}

    for name, path in files.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

        data[name] = pd.read_csv(path)

        print(
            f"Loaded {path.name}: "
            f"{len(data[name]):,} rows"
        )

    return data


# =========================================================
# INPUT VALIDATION
# =========================================================

def validate_inputs(data):

    required = {
        "planning_master": [
            "sku",
            "supplier_id",
            "snapshot_date",
            "quoted_lead_time_weeks",
            "transit_time_weeks",
            "planned_replenishment_weeks",
        ],
        "inventory_policy": [
            "sku",
            "planning_lead_time_weeks",
        ],
        "bridge_quantity_analysis": [
            "sku",
            "first_feasible_normal_receipt",
            "first_pre_receipt_stockout_week",
            "minimum_stockout_bridge_units",
            "minimum_safety_bridge_units",
            "bridge_status",
            "priority",
        ],
        "open_purchase_orders_clean": [
            "sku",
            "open_qty",
            "expected_date",
        ],
    }

    for name, columns in required.items():

        missing = [
            column
            for column in columns
            if column not in data[name].columns
        ]

        if missing:
            raise ValueError(
                f"{name} missing columns: {missing}"
            )

    planning = data["planning_master"]
    policy = data["inventory_policy"]
    bridge = data["bridge_quantity_analysis"]
    open_pos = data["open_purchase_orders_clean"]

    for name, frame in [
        ("planning_master", planning),
        ("inventory_policy", policy),
        ("bridge_quantity_analysis", bridge),
    ]:

        if frame["sku"].isna().any():
            raise ValueError(
                f"Missing SKUs in {name}"
            )

        if frame["sku"].duplicated().any():
            raise ValueError(
                f"Duplicate SKUs in {name}"
            )

    planning_skus = set(planning["sku"])

    if planning_skus != set(policy["sku"]):
        raise ValueError(
            "Inventory policy SKU coverage mismatch"
        )

    if planning_skus != set(bridge["sku"]):
        raise ValueError(
            "Bridge analysis SKU coverage mismatch"
        )

    if not set(open_pos["sku"]).issubset(planning_skus):
        raise ValueError(
            "Open POs contain unknown SKUs"
        )

    if (
        EXPEDITED_TRANSIT_WEEKS <= 0
        or STANDARD_FREIGHT_COST_PER_UNIT < 0
        or EXPEDITED_FREIGHT_COST_PER_UNIT
        < STANDARD_FREIGHT_COST_PER_UNIT
    ):
        raise ValueError(
            "Invalid freight assumptions"
        )

    lead_columns = [
        "quoted_lead_time_weeks",
        "transit_time_weeks",
        "planned_replenishment_weeks",
    ]

    for column in lead_columns:

        values = pd.to_numeric(
            planning[column],
            errors="coerce"
        )

        if values.isna().any() or (values < 0).any():
            raise ValueError(
                f"Invalid planning values: {column}"
            )

    policy_lead = pd.to_numeric(
        policy["planning_lead_time_weeks"],
        errors="coerce"
    )

    if (
        policy_lead.isna().any()
        or (policy_lead <= 0).any()
    ):
        raise ValueError(
            "Invalid inventory policy lead times"
        )

    baseline = (
        planning["quoted_lead_time_weeks"]
        + planning["transit_time_weeks"]
    )

    if not np.allclose(
        baseline,
        planning["planned_replenishment_weeks"],
        rtol=0,
        atol=0.000001,
    ):
        raise ValueError(
            "Production + transit baseline mismatch"
        )

    snapshots = pd.to_datetime(
        planning["snapshot_date"],
        errors="coerce"
    )

    if snapshots.isna().any() or snapshots.nunique() != 1:
        raise ValueError(
            "Invalid or inconsistent snapshot dates"
        )

    for column in [
        "minimum_stockout_bridge_units",
        "minimum_safety_bridge_units",
    ]:

        values = pd.to_numeric(
            bridge[column],
            errors="coerce"
        )

        if values.isna().any() or (values < 0).any():
            raise ValueError(
                f"Invalid bridge quantities: {column}"
            )

    if (
        bridge["minimum_safety_bridge_units"].to_numpy()
        < bridge["minimum_stockout_bridge_units"].to_numpy()
    ).any():
        raise ValueError(
            "Safety bridge below stockout bridge"
        )

    open_qty = pd.to_numeric(
        open_pos["open_qty"],
        errors="coerce"
    )

    if open_qty.isna().any() or (open_qty < 0).any():
        raise ValueError(
            "Invalid open PO quantities"
        )

    print("PASS: Required columns validated")
    print("PASS: All 200 SKU relationships validated")
    print("PASS: Baseline lead times reconcile")
    print("PASS: Inventory policy lead times validated")
    print("PASS: Bridge and open PO quantities validated")


# =========================================================
# SUMMARIZE OPEN PURCHASE ORDERS
# =========================================================

def summarize_open_pos(open_pos):

    po = open_pos.copy()

    po["open_qty"] = pd.to_numeric(
        po["open_qty"]
    )

    po = po.loc[
        po["open_qty"] > 0
    ].copy()

    if po.empty:
        return pd.DataFrame(
            columns=[
                "sku",
                "open_po_lines",
                "open_po_units",
                "earliest_open_po_expected_date",
            ]
        )

    po["expected_date"] = pd.to_datetime(
        po["expected_date"],
        errors="coerce"
    )

    summary = (
        po.groupby("sku", as_index=False)
        .agg(
            open_po_lines=("open_qty", "size"),
            open_po_units=("open_qty", "sum"),
            earliest_open_po_expected_date=(
                "expected_date",
                "min"
            ),
        )
    )

    return summary


# =========================================================
# BUILD SCENARIOS
# =========================================================

def calculate_scenarios(data):

    planning = data["planning_master"].copy()
    policy = data["inventory_policy"].copy()
    bridge = data["bridge_quantity_analysis"].copy()

    open_po_summary = summarize_open_pos(
        data["open_purchase_orders_clean"]
    )

    # Inventory policy is the authoritative source
    # of planning lead time for all 200 SKUs.
    result = planning.merge(
        policy[
            ["sku", "planning_lead_time_weeks"]
        ],
        on="sku",
        how="left",
        validate="one_to_one",
    )

    result = result.merge(
        bridge,
        on="sku",
        how="left",
        validate="one_to_one",
        suffixes=("", "_bridge"),
    )

    result = result.merge(
        open_po_summary,
        on="sku",
        how="left",
        validate="one_to_one",
    )

    result["open_po_lines"] = (
        result["open_po_lines"]
        .fillna(0)
        .astype(int)
    )

    result["open_po_units"] = (
        result["open_po_units"]
        .fillna(0)
    )

    date_columns = [
        "snapshot_date",
        "first_feasible_normal_receipt",
        "first_pre_receipt_stockout_week",
        "earliest_open_po_expected_date",
    ]

    for column in date_columns:

        result[column] = pd.to_datetime(
            result[column],
            errors="coerce"
        )

    if result["snapshot_date"].isna().any():
        raise ValueError(
            "Missing scenario snapshot date"
        )

    for column in [
        "minimum_stockout_bridge_units",
        "minimum_safety_bridge_units",
    ]:

        result[column] = np.ceil(
            pd.to_numeric(result[column])
        ).astype(int)

    # -----------------------------------------------------
    # PRODUCTION AND TRANSPORTATION BASELINE
    # -----------------------------------------------------

    result["baseline_lead_time_weeks"] = (
        result["quoted_lead_time_weeks"]
        + result["transit_time_weeks"]
    )

    # Historical effective lead time may be above or
    # below the quoted baseline. Only positive excess
    # is treated as a risk allowance in this scenario.
    #
    # This is an approximation, not a causal
    # decomposition of historical delays.

    result["supplier_risk_allowance_weeks"] = (
        result["planning_lead_time_weeks"]
        - result["baseline_lead_time_weeks"]
    ).clip(lower=0)

    result["potential_transit_savings_weeks"] = (
        result["transit_time_weeks"]
        - EXPEDITED_TRANSIT_WEEKS
    ).clip(lower=0)

    # If normal transit is already shorter than the
    # assumed air transit, do not make it slower.
    result["scenario_transit_weeks"] = np.minimum(
        result["transit_time_weeks"],
        EXPEDITED_TRANSIT_WEEKS
    )

    result["expedited_scenario_lead_weeks"] = (
        result["quoted_lead_time_weeks"]
        + result["scenario_transit_weeks"]
        + result["supplier_risk_allowance_weeks"]
    )

    expedited_days = np.ceil(
        result["expedited_scenario_lead_weeks"]
        * DAYS_PER_WEEK
    ).astype(int)

    result["hypothetical_new_order_air_arrival"] = (
        result["snapshot_date"]
        + pd.to_timedelta(
            expedited_days,
            unit="D"
        )
    )

    # -----------------------------------------------------
    # STOCKOUT TIMING
    # -----------------------------------------------------

    has_stockout = (
        result["minimum_stockout_bridge_units"] > 0
    )

    has_stockout_date = (
        result[
            "first_pre_receipt_stockout_week"
        ].notna()
    )

    if (has_stockout & ~has_stockout_date).any():
        raise ValueError(
            "Stockout bridge exists without stockout date"
        )

    # A weekly stockout forecast does not identify
    # the precise day of shortage.
    #
    # Require the hypothetical arrival to be strictly
    # before the start of the stockout week.
    #
    # This is a timing screen, not proof of full
    # shortage prevention.

    result["new_order_air_before_stockout"] = (
        has_stockout
        & (
            result[
                "hypothetical_new_order_air_arrival"
            ]
            < result[
                "first_pre_receipt_stockout_week"
            ]
        )
    )

    # -----------------------------------------------------
    # EXISTING OPEN PO OPPORTUNITY
    # -----------------------------------------------------

    # Open PO availability is conditional because
    # production/shipment readiness is unknown.

    result["existing_po_supplier_confirmation"] = (
        np.where(
            result["open_po_units"] > 0,
            "REQUIRED",
            "NO OPEN PO"
        )
    )

    result["existing_po_potential_bridge_units"] = (
        np.minimum(
            result["open_po_units"],
            result["minimum_stockout_bridge_units"]
        )
        .astype(int)
    )

    # -----------------------------------------------------
    # ILLUSTRATIVE FREIGHT COSTS
    # -----------------------------------------------------

    # These costs apply to the entire stockout bridge
    # quantity. They are NOT a calculated minimum
    # shipment size for the expedited arrival date.

    result["standard_bridge_freight_cost"] = (
        result["minimum_stockout_bridge_units"]
        * STANDARD_FREIGHT_COST_PER_UNIT
    )

    result["air_bridge_freight_cost"] = (
        result["minimum_stockout_bridge_units"]
        * EXPEDITED_FREIGHT_COST_PER_UNIT
    )

    result["incremental_air_freight_cost"] = (
        result["air_bridge_freight_cost"]
        - result["standard_bridge_freight_cost"]
    )

    # -----------------------------------------------------
    # SCENARIO CLASSIFICATION
    # -----------------------------------------------------

    result["scenario_status"] = np.select(
        [
            ~has_stockout,
            result["new_order_air_before_stockout"],
            has_stockout
            & ~result["new_order_air_before_stockout"],
        ],
        [
            "NO STOCKOUT BRIDGE REQUIRED",
            "AIR ARRIVAL BEFORE FIRST STOCKOUT - CONDITIONAL",
            "NEW ORDER AIR TOO LATE FOR FIRST STOCKOUT",
        ],
        default="REVIEW REQUIRED"
    )

    result["scenario_limitation"] = (
        "Synthetic freight rates; supplier readiness "
        "unconfirmed; weekly forecast timing; "
        "shortage prevention not independently simulated"
    )

    columns = [
        "sku",
        "supplier_id",
        "snapshot_date",
        "priority",
        "bridge_status",
        "minimum_stockout_bridge_units",
        "minimum_safety_bridge_units",
        "first_pre_receipt_stockout_week",
        "first_feasible_normal_receipt",
        "quoted_lead_time_weeks",
        "transit_time_weeks",
        "baseline_lead_time_weeks",
        "planning_lead_time_weeks",
        "supplier_risk_allowance_weeks",
        "potential_transit_savings_weeks",
        "scenario_transit_weeks",
        "expedited_scenario_lead_weeks",
        "hypothetical_new_order_air_arrival",
        "new_order_air_before_stockout",
        "open_po_lines",
        "open_po_units",
        "earliest_open_po_expected_date",
        "existing_po_potential_bridge_units",
        "existing_po_supplier_confirmation",
        "standard_bridge_freight_cost",
        "air_bridge_freight_cost",
        "incremental_air_freight_cost",
        "scenario_status",
        "scenario_limitation",
    ]

    return result[columns].copy()


# =========================================================
# VALIDATE RESULTS
# =========================================================

def validate_output(result, data):

    planning = data["planning_master"]
    bridge = data["bridge_quantity_analysis"]

    has_stockout = (
        result["minimum_stockout_bridge_units"] > 0
    )

    early = result["new_order_air_before_stockout"]

    checks = {
        "One scenario per planning SKU":
            len(result) == len(planning),

        "No duplicate SKU records":
            not result["sku"].duplicated().any(),

        "All planning SKUs preserved":
            set(result["sku"]) == set(planning["sku"]),

        "All bridge SKUs preserved":
            set(result["sku"]) == set(bridge["sku"]),

        "No missing planning lead times":
            result["planning_lead_time_weeks"]
            .notna().all(),

        "Positive expedited lead times":
            (
                result["expedited_scenario_lead_weeks"]
                > 0
            ).all(),

        "Nonnegative stockout bridge":
            (
                result["minimum_stockout_bridge_units"]
                >= 0
            ).all(),

        "Safety bridge covers stockout bridge":
            (
                result["minimum_safety_bridge_units"]
                >= result["minimum_stockout_bridge_units"]
            ).all(),

        "No negative open PO units":
            (
                result["open_po_units"] >= 0
            ).all(),

        "Existing PO bridge respects limits":
            (
                result["existing_po_potential_bridge_units"]
                <= result["open_po_units"]
            ).all()
            and (
                result["existing_po_potential_bridge_units"]
                <= result["minimum_stockout_bridge_units"]
            ).all(),

        "Freight premium reconciles":
            np.allclose(
                result["incremental_air_freight_cost"],
                result["air_bridge_freight_cost"]
                - result["standard_bridge_freight_cost"]
            ),

        "No negative freight premiums":
            (
                result["incremental_air_freight_cost"]
                >= 0
            ).all(),

        "Early arrivals have stockout exposure":
            (~early | has_stockout).all(),

        "Early arrivals precede stockout week":
            (
                result.loc[
                    early,
                    "hypothetical_new_order_air_arrival"
                ].to_numpy()
                <
                result.loc[
                    early,
                    "first_pre_receipt_stockout_week"
                ].to_numpy()
            ).all(),

        "One snapshot date":
            result["snapshot_date"].nunique() == 1,

        "No missing scenario statuses":
            result["scenario_status"].notna().all(),
    }

    failures = []

    for name, passed in checks.items():

        if passed:
            print(f"PASS: {name}")
        else:
            print(f"FAIL: {name}")
            failures.append(name)

    if failures:
        raise ValueError(
            "Freight scenario validation failed: "
            + ", ".join(failures)
        )

    print(
        "\nAll freight scenario validations passed."
    )


# =========================================================
# REPORT
# =========================================================

def print_summary(result):

    print("\n" + "=" * 60)
    print("FREIGHT RECOVERY SCENARIO SUMMARY")
    print("=" * 60)

    print(
        "Snapshot:",
        result["snapshot_date"].iloc[0].date()
    )

    print(
        "SKUs analyzed:",
        len(result)
    )

    print(
        "SKUs requiring stockout bridge:",
        (
            result["minimum_stockout_bridge_units"] > 0
        ).sum()
    )

    print(
        "New-order air before first stockout:",
        result["new_order_air_before_stockout"].sum()
    )

    print(
        "New-order air too late:",
        (
            result["scenario_status"]
            == "NEW ORDER AIR TOO LATE FOR FIRST STOCKOUT"
        ).sum()
    )

    print(
        "SKUs with existing open POs:",
        (
            result["open_po_units"] > 0
        ).sum()
    )

    print(
        "Total stockout bridge units:",
        f"{result['minimum_stockout_bridge_units'].sum():,.0f}"
    )

    print(
        "Illustrative premium for all bridge units:",
        f"${result['incremental_air_freight_cost'].sum():,.2f}"
    )

    print(
        "\nScenario statuses:"
    )

    print(
        result["scenario_status"]
        .value_counts()
        .to_string()
    )

    print(
        "\nFirst 10 stockout bridge scenarios:"
    )

    sample = result.loc[
        result["minimum_stockout_bridge_units"] > 0,
        [
            "sku",
            "priority",
            "minimum_stockout_bridge_units",
            "first_pre_receipt_stockout_week",
            "hypothetical_new_order_air_arrival",
            "scenario_status",
        ]
    ].head(10)

    print(
        sample.to_string(index=False)
    )

    print(
        "\nIMPORTANT: Arrival feasibility is conditional."
    )

    print(
        "This model does not confirm supplier readiness, "
        "actual air freight rates, or full shortage prevention."
    )

    print(
        "The illustrative premium is NOT an optimized "
        "freight budget."
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\nRunning Enhancement 1B: "
        "Freight Recovery Scenario Analysis\n"
    )

    data = load_data()

    validate_inputs(data)

    result = calculate_scenarios(data)

    validate_output(result, data)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d"
    )

    print(
        f"\nSaved: {OUTPUT_FILE}"
    )

    print_summary(result)


if __name__ == "__main__":
    main()
