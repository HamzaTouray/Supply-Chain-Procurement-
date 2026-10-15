
from pathlib import Path
import math
import numpy as np
import pandas as pd

# ==========================================================
# PHASE 7F - FEASIBLE SUPPLY & EXCEPTION MANAGEMENT
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

ORDERS_FILE = DATA / "consolidated_sku_orders.csv"
SUMMARY_FILE = DATA / "time_phased_planning_summary.csv"
WEEKLY_FILE = DATA / "time_phased_inventory_plan.csv"
OPEN_PO_FILE = DATA / "open_purchase_orders_clean.csv"
MASTER_FILE = DATA / "planning_master.csv"

OUTPUT_ACTIONS = DATA / "feasible_supply_actions.csv"
OUTPUT_EXCEPTIONS = DATA / "supply_exceptions.csv"
OUTPUT_SUPPLIERS = DATA / "supplier_execution_summary.csv"
OUTPUT_PROJECTION = DATA / "feasible_inventory_projection.csv"

TOL = 1e-7


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("✓", message)


def load_data():
    section("PHASE 7F - FEASIBLE SUPPLY & EXCEPTIONS")

    files = {
        "orders": ORDERS_FILE,
        "summary": SUMMARY_FILE,
        "weekly": WEEKLY_FILE,
        "open_pos": OPEN_PO_FILE,
        "master": MASTER_FILE,
    }

    data = {}
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(path)
        data[name] = pd.read_csv(path)
        print(f"✓ {name}: {len(data[name]):,} rows")

    return data


def validate_inputs(data):
    required = {
        "orders": {
            "sku", "supplier_id", "required_receipt_week",
            "planned_release_date", "planning_lead_time_weeks",
            "recommended_order_qty", "recommended_order_value",
            "unit_cost", "supplier_risk_class", "release_status",
        },
        "summary": {
            "sku", "net_available", "safety_stock",
        },
        "weekly": {
            "sku", "forecast_week", "forecast_horizon_week",
            "forecast_demand", "existing_po_receipt_qty",
        },
        "open_pos": {
            "po_number", "sku", "open_qty", "expected_date",
            "overdue_flag", "partial_receipt_flag",
            "expected_after_promise_flag",
        },
        "master": {"sku", "snapshot_date"},
    }

    for name, columns in required.items():
        missing = columns - set(data[name].columns)
        check(not missing, f"{name} required columns present")

    check(data["summary"]["sku"].is_unique,
          "One summary row per SKU")
    check(data["weekly"]["sku"].nunique() == 200,
          "200 SKUs in weekly plan")
    check(len(data["weekly"]) == 5200,
          "5,200 weekly forecast records")

    check(
        (data["orders"]["recommended_order_qty"] > 0).all(),
        "Positive consolidated order quantities"
    )


def to_bool(series):
    return (
        series.astype(str).str.strip().str.lower()
        .isin(["true", "1", "yes"])
    )


def monday(series):
    dates = pd.to_datetime(series, errors="coerce").dt.normalize()
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def prepare(data):
    orders = data["orders"].copy()
    summary = data["summary"].copy()
    weekly = data["weekly"].copy()
    open_pos = data["open_pos"].copy()
    master = data["master"].copy()

    snapshot = pd.to_datetime(
        master["snapshot_date"], errors="coerce"
    ).dropna().max().normalize()

    if pd.isna(snapshot):
        raise ValueError("Missing snapshot date")

    first_week = pd.to_datetime(
        weekly["forecast_week"]
    ).min().normalize()

    # Planning baseline must agree with the forecast horizon.
    if abs((first_week - snapshot).days) > 7:
        raise ValueError(
            f"Snapshot {snapshot.date()} and forecast "
            f"start {first_week.date()} are inconsistent"
        )

    orders["required_receipt_week"] = pd.to_datetime(
        orders["required_receipt_week"]
    ).dt.normalize()

    orders["planned_release_date"] = pd.to_datetime(
        orders["planned_release_date"]
    ).dt.normalize()

    weekly["forecast_week"] = pd.to_datetime(
        weekly["forecast_week"]
    ).dt.normalize()

    open_pos["expected_date"] = pd.to_datetime(
        open_pos["expected_date"], errors="coerce"
    )

    for flag in [
        "overdue_flag",
        "partial_receipt_flag",
        "expected_after_promise_flag",
    ]:
        open_pos[flag] = to_bool(open_pos[flag])

    open_pos["at_risk"] = open_pos[
        ["overdue_flag", "partial_receipt_flag",
         "expected_after_promise_flag"]
    ].any(axis=1)

    check(orders["required_receipt_week"].notna().all(),
          "All required receipt dates valid")
    check(open_pos["expected_date"].notna().all(),
          "All open PO expected dates valid")

    return orders, summary, weekly, open_pos, snapshot, first_week


def classify_actions(orders, snapshot):
    section("FEASIBLE ORDER ARRIVAL ANALYSIS")

    df = orders.copy()

    # Preserve original release dates and quantities.
    df["original_release_date"] = df["planned_release_date"]

    # Earliest executable release cannot precede snapshot.
    df["executable_release_date"] = df[
        "planned_release_date"
    ].where(df["planned_release_date"] > snapshot, snapshot)

    lead_days = np.ceil(
        df["planning_lead_time_weeks"].astype(float) * 7
    ).astype(int)

    df["planning_lead_time_days"] = lead_days

    df["earliest_feasible_receipt"] = (
        df["executable_release_date"]
        + pd.to_timedelta(lead_days, unit="D")
    )

    df["receipt_delay_days"] = (
        df["earliest_feasible_receipt"]
        - df["required_receipt_week"]
    ).dt.days.clip(lower=0)

    df["receipt_feasible_on_time"] = (
        df["receipt_delay_days"] == 0
    )

    df["release_days_from_snapshot"] = (
        df["planned_release_date"] - snapshot
    ).dt.days

    conditions = [
        (df["release_days_from_snapshot"] <= 0)
        & (df["receipt_delay_days"] > 0),

        (df["release_days_from_snapshot"] <= 0)
        & (df["receipt_delay_days"] == 0),

        (df["release_days_from_snapshot"] <= 7),

        (df["release_days_from_snapshot"] <= 28),
    ]

    choices = [
        "ORDER NOW - LATE / EXPEDITE",
        "ORDER NOW - ON TIME",
        "ORDER THIS WEEK",
        "ORDER SOON",
    ]

    df["execution_action"] = np.select(
        conditions, choices, default="PLANNED ORDER"
    )

    df["expedite_required"] = (
        df["receipt_delay_days"] > 0
    )

    df["urgency_priority"] = np.select(
        [
            df["receipt_delay_days"] > 28,
            df["receipt_delay_days"] > 0,
            df["release_days_from_snapshot"] <= 7,
        ],
        ["CRITICAL", "HIGH", "MEDIUM"],
        default="PLANNED"
    )

    df["execution_reason"] = np.where(
        df["receipt_delay_days"] > 0,
        "Earliest feasible receipt misses required date by "
        + df["receipt_delay_days"].astype(str) + " days",
        "Earliest feasible receipt meets required date"
    )

    return df


def build_open_po_risk(open_pos, first_week, snapshot):
    section("EXISTING OPEN PO RISK ANALYSIS")

    po = open_pos.copy()

    # Overdue open POs are not assumed confirmed arrivals.
    po["receipt_week"] = monday(po["expected_date"])

    po["historical_expected_date"] = (
        po["expected_date"] < snapshot
    )

    po["uncertain_receipt"] = (
        po["at_risk"] | po["historical_expected_date"]
    )

    # Preserve the Phase 7D convention for comparison:
    # pre-horizon dates appear in week 1, but are flagged.
    po.loc[
        po["receipt_week"] < first_week, "receipt_week"
    ] = first_week

    po["risk_action"] = np.where(
        po["uncertain_receipt"],
        "EXPEDITE / CONFIRM EXISTING PO",
        "MONITOR EXISTING PO"
    )

    print(
        f"Open PO units: {po['open_qty'].sum():,.0f}"
    )
    print(
        f"Uncertain PO units: "
        f"{po.loc[po['uncertain_receipt'], 'open_qty'].sum():,.0f}"
    )

    return po


def simulate_feasible_inventory(
    weekly, summary, actions, open_pos
):
    section("FEASIBLE 26-WEEK INVENTORY PROJECTION")

    weeks = weekly.copy()

    # New orders arrive no earlier than their feasible date.
    actions = actions.copy()
    actions["feasible_receipt_week"] = monday(
        actions["earliest_feasible_receipt"]
    )

    # Week buckets are planning approximations, not
    # promises that goods arrive at the week's start.
    new_receipts = actions.groupby(
        ["sku", "feasible_receipt_week"]
    )["recommended_order_qty"].sum().to_dict()

    # Only non-risky existing POs are counted as confirmed
    # in the conservative projection.
    confirmed_pos = open_pos[
        ~open_pos["uncertain_receipt"]
    ]

    confirmed_receipts = confirmed_pos.groupby(
        ["sku", "receipt_week"]
    )["open_qty"].sum().to_dict()

    summary_lookup = summary.set_index("sku")
    rows = []

    for sku, group in weeks.groupby("sku"):
        group = group.sort_values("forecast_horizon_week")
        info = summary_lookup.loc[sku]

        inventory = float(info["net_available"])
        safety = float(info["safety_stock"])

        for _, week in group.iterrows():
            date = pd.Timestamp(week["forecast_week"])

            confirmed = float(
                confirmed_receipts.get((sku, date), 0)
            )
            new = float(new_receipts.get((sku, date), 0))
            demand = float(week["forecast_demand"])

            start = inventory
            inventory = start + confirmed + new - demand

            rows.append({
                "sku": sku,
                "forecast_week": date,
                "beginning_inventory": start,
                "confirmed_existing_receipts": confirmed,
                "feasible_new_receipts": new,
                "forecast_demand": demand,
                "ending_inventory": inventory,
                "safety_stock": safety,
                "below_safety_stock": inventory < safety - TOL,
                "stockout": inventory < -TOL,
            })

    return pd.DataFrame(rows)


def build_exceptions(actions, po, projection):
    section("BUILDING SUPPLY EXCEPTIONS")

    new_exceptions = actions[
        actions["receipt_delay_days"] > 0
    ].copy()

    new_exceptions["exception_type"] = (
        "NEW ORDER ARRIVAL TOO LATE"
    )
    new_exceptions["reference"] = new_exceptions["sku"]
    new_exceptions["exception_units"] = (
        new_exceptions["recommended_order_qty"]
    )
    new_exceptions["exception_date"] = (
        new_exceptions["required_receipt_week"]
    )

    new_exceptions = new_exceptions[[
        "sku", "supplier_id", "exception_type",
        "reference", "exception_units",
        "exception_date", "receipt_delay_days",
        "execution_action"
    ]]

    po_exceptions = po[po["uncertain_receipt"]].copy()
    po_exceptions["exception_type"] = (
        "EXISTING PO RECEIPT UNCONFIRMED"
    )
    po_exceptions["reference"] = po_exceptions["po_number"]
    po_exceptions["exception_units"] = po_exceptions["open_qty"]
    po_exceptions["exception_date"] = po_exceptions["expected_date"]
    po_exceptions["receipt_delay_days"] = np.nan
    po_exceptions["execution_action"] = (
        "EXPEDITE / CONFIRM EXISTING PO"
    )

    po_exceptions = po_exceptions[[
        "sku", "supplier_id", "exception_type",
        "reference", "exception_units",
        "exception_date", "receipt_delay_days",
        "execution_action"
    ]]

    breaches = projection[
        projection["below_safety_stock"]
    ].copy()

    breaches["exception_type"] = (
        "FEASIBLE INVENTORY BELOW SAFETY STOCK"
    )
    breaches["supplier_id"] = breaches["sku"].map(
        actions.drop_duplicates("sku").set_index("sku")[
            "supplier_id"
        ]
    )
    breaches["reference"] = breaches["sku"]
    breaches["exception_units"] = (
        breaches["safety_stock"] - breaches["ending_inventory"]
    ).clip(lower=0)
    breaches["exception_date"] = breaches["forecast_week"]
    breaches["receipt_delay_days"] = np.nan
    breaches["execution_action"] = (
        "REVIEW EXPEDITE / ALTERNATE SUPPLY"
    )

    breaches = breaches[[
        "sku", "supplier_id", "exception_type",
        "reference", "exception_units",
        "exception_date", "receipt_delay_days",
        "execution_action"
    ]]

    result = pd.concat(
        [new_exceptions, po_exceptions, breaches],
        ignore_index=True
    )

    return result


def build_supplier_summary(actions, po, projection):
    order_summary = actions.groupby(
        "supplier_id", as_index=False
    ).agg(
        new_order_lines=("sku", "size"),
        new_order_units=("recommended_order_qty", "sum"),
        new_order_value=("recommended_order_value", "sum"),
        late_order_lines=("expedite_required", "sum"),
    )

    po_summary = po.groupby(
        "supplier_id", as_index=False
    ).agg(
        existing_open_po_units=("open_qty", "sum"),
        uncertain_open_po_units=(
            "open_qty",
            lambda s: s[po.loc[s.index, "uncertain_receipt"]].sum()
        ),
    )

    supplier_lookup = (
        actions[["sku", "supplier_id"]]
        .drop_duplicates("sku")
    )

    breaches = projection[
        projection["below_safety_stock"]
    ].merge(supplier_lookup, on="sku", how="left")

    breach_summary = breaches.groupby(
        "supplier_id", as_index=False
    ).agg(
        at_risk_sku_count=("sku", "nunique"),
        at_risk_sku_weeks=("sku", "size")
    )

    result = (
        order_summary
        .merge(po_summary, on="supplier_id", how="outer")
        .merge(breach_summary, on="supplier_id", how="left")
        .fillna(0)
    )

    return result.sort_values(
        "new_order_value", ascending=False
    )


def validate_outputs(actions, po, projection, exceptions, suppliers):
    section("PHASE 7F VALIDATION")

    check(len(projection) == 5200,
          "5,200 feasible projection rows")

    check(
        np.isclose(
            actions["recommended_order_qty"].sum(),
            actions.groupby("sku")[
                "recommended_order_qty"
            ].sum().sum()
        ),
        "Consolidated order quantities reconcile"
    )

    check(
        np.isclose(
            po["open_qty"].sum(),
            po.groupby("sku")["open_qty"].sum().sum()
        ),
        "Open PO quantities reconcile"
    )

    check(
        np.isclose(
            suppliers["new_order_value"].sum(),
            actions["recommended_order_value"].sum()
        ),
        "Supplier purchasing value reconciles"
    )

    check(actions["earliest_feasible_receipt"].notna().all(),
          "All feasible arrival dates present")

    check(
        (actions["earliest_feasible_receipt"]
         >= actions["executable_release_date"]).all(),
        "Feasible receipts do not precede executable releases"
    )

    print("\nFeasible inventory results:")
    print(
        "Safety-stock breach SKU-weeks:",
        int(projection["below_safety_stock"].sum())
    )
    print(
        "Stockout SKU-weeks:",
        int(projection["stockout"].sum())
    )

    # Breaches here are exceptions, not validation failures.
    # Unlike 7E, 7F does not assume past-due orders arrive
    # on their historical required receipt dates.
    print("\nAll Phase 7F structural checks passed.")


def save_outputs(actions, exceptions, suppliers, projection):
    section("SAVING PHASE 7F OUTPUTS")

    outputs = [
        (actions, OUTPUT_ACTIONS),
        (exceptions, OUTPUT_EXCEPTIONS),
        (suppliers, OUTPUT_SUPPLIERS),
        (projection, OUTPUT_PROJECTION),
    ]

    for df, path in outputs:
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)
        print("✓", path.name)


def print_summary(actions, po, projection, exceptions, suppliers):
    section("PHASE 7F EXECUTION SUMMARY")

    print(f"Consolidated order lines: {len(actions):,}")
    print(
        f"New purchasing value: "
        f"${actions['recommended_order_value'].sum():,.2f}"
    )

    print("\nExecution actions:")
    print(actions["execution_action"].value_counts().to_string())

    print("\nLate-to-need new order lines:",
          int(actions["expedite_required"].sum()))

    print("\nExisting PO actions:")
    print(po["risk_action"].value_counts().to_string())

    print("\nFeasible inventory:")
    print(
        f"SKU-weeks below safety stock: "
        f"{projection['below_safety_stock'].sum():,}"
    )
    print(
        f"SKU-weeks stocked out: "
        f"{projection['stockout'].sum():,}"
    )

    print("\nException types:")
    print(exceptions["exception_type"].value_counts().to_string())

    section("SUPPLIER EXECUTION SUMMARY")
    print(suppliers.head(15).to_string(index=False))

    section("TOP LATE-TO-NEED NEW ORDERS")
    columns = [
        "sku", "supplier_id", "required_receipt_week",
        "earliest_feasible_receipt", "receipt_delay_days",
        "recommended_order_qty", "recommended_order_value",
        "execution_action",
    ]
    print(
        actions.sort_values(
            ["receipt_delay_days", "recommended_order_value"],
            ascending=False
        )[columns].head(20).to_string(index=False)
    )


def main():
    data = load_data()
    validate_inputs(data)

    (
        orders, summary, weekly,
        open_pos, snapshot, first_week
    ) = prepare(data)

    actions = classify_actions(orders, snapshot)
    po = build_open_po_risk(open_pos, first_week, snapshot)

    projection = simulate_feasible_inventory(
        weekly, summary, actions, po
    )

    exceptions = build_exceptions(actions, po, projection)
    suppliers = build_supplier_summary(actions, po, projection)

    validate_outputs(
        actions, po, projection, exceptions, suppliers
    )

    save_outputs(actions, exceptions, suppliers, projection)
    print_summary(actions, po, projection, exceptions, suppliers)

    section("PHASE 7F COMPLETE")


if __name__ == "__main__":
    main()
