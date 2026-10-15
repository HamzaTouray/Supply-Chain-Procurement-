"""

PROCUREMENT AI COMMAND CENTER

Phase 8A - Executive Overview

Phase 8B - Inventory Health, Forecasting, SKU Drilldown,

           ABC Analysis, Data Quality

Phase 8C - Supplier Intelligence

Phase 8D - Procurement Order Management
Phase 10C - AI Procurement Assistant

"""

from pathlib import Path

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import streamlit as st
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from supplier_intelligence_page import show_supplier_intelligence

from procurement_orders_page import show_procurement_orders

from procurement_ai_page import show_procurement_ai
from planning_analytics_page import show_planning_analytics

# ============================================================

# 1. CONFIGURATION

# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed"

st.set_page_config(

    page_title="Procurement AI | Command Center",

    page_icon="ðŸ“¦",

    layout="wide",

    initial_sidebar_state="expanded",

)

FILES = {

    "master": "planning_master.csv",

    "actions": "feasible_supply_actions.csv",

    "projection": "feasible_inventory_projection.csv",

    "recovery": "recovery_action_plan.csv",

    "suppliers": "supplier_recovery_priorities.csv",

    "forecasts": "production_forecasts.csv",

    "policy": "inventory_policy.csv",

}

# ============================================================

# 2. STYLE

# ============================================================

st.markdown(

    """

    <style>

    .block-container {

        padding-top: 1.5rem;

        padding-bottom: 2rem;

    }

    [data-testid="stMetric"] {

        background: rgba(128,128,128,0.07);

        padding: 16px;

        border-radius: 10px;

        border: 1px solid rgba(128,128,128,0.12);

    }

    h1, h2, h3 {

        letter-spacing: -0.025em;

    }

    </style>

    """,

    unsafe_allow_html=True,

)

# ============================================================

# 3. DATA LOADING AND VALIDATION

# ============================================================

@st.cache_data(show_spinner=False)

def read_csv(path, modified_time):

    return pd.read_csv(path)

def load_data():

    missing = [

        filename

        for filename in FILES.values()

        if not (DATA / filename).exists()

    ]

    if missing:

        st.error("Missing required processed data files:")

        st.code("\n".join(missing))

        st.stop()

    result = {}

    for key, filename in FILES.items():

        path = DATA / filename

        result[key] = read_csv(

            str(path),

            path.stat().st_mtime,

        )

    return result

def validate(data):

    required = {

        "master": {

            "sku", "product_name", "category",

            "supplier_id", "on_hand", "allocated",

            "available", "net_available",

            "inventory_value", "current_weeks_of_supply",

            "currently_stocked_out",

        },

        "actions": {

            "sku", "supplier_id",

            "recommended_order_qty",

            "recommended_order_value",

            "execution_action", "receipt_delay_days",

        },

        "projection": {

            "sku", "forecast_week",

            "stockout", "below_safety_stock",

            "ending_inventory",

        },

        "recovery": {

            "sku", "priority", "stockout_weeks",

            "peak_shortage_exposure_value",

        },

        "suppliers": {

            "supplier_id", "critical_skus",

            "total_stockout_sku_weeks",

        },

        "forecasts": {

            "sku", "forecast_week",

            "forecast_horizon_week",

            "forecast_demand",

        },

        "policy": {

            "sku", "safety_stock", "reorder_point",

        },

    }

    errors = []

    for key, columns in required.items():

        missing = columns - set(data[key].columns)

        if missing:

            errors.append(

                f"{FILES[key]}: {sorted(missing)}"

            )

    if errors:

        st.error("Data schema validation failed:")

        st.code("\n".join(errors))

        st.stop()

    for name in ["master", "recovery", "policy"]:

        if data[name]["sku"].duplicated().any():

            st.error(f"Duplicate SKUs in {FILES[name]}")

            st.stop()

    master_skus = set(data["master"]["sku"])

    for name in ["recovery", "policy"]:

        if set(data[name]["sku"]) != master_skus:

            st.error(

                f"SKU mismatch between master and {FILES[name]}"

            )

            st.stop()

    if set(data["forecasts"]["sku"]) != master_skus:

        st.error("Forecast SKU set does not match master.")

        st.stop()

    if data["forecasts"].duplicated(

        ["sku", "forecast_week"]

    ).any():

        st.error("Duplicate SKU/forecast-week records.")

        st.stop()

def as_bool(series):

    return (

        series.astype(str)

        .str.strip()

        .str.lower()

        .isin(["true", "1", "yes"])

    )

data = load_data()

validate(data)

master = data["master"].copy()

actions = data["actions"].copy()

projection = data["projection"].copy()

recovery = data["recovery"].copy()

suppliers = data["suppliers"].copy()

forecasts = data["forecasts"].copy()

policy = data["policy"].copy()

master["currently_stocked_out"] = as_bool(

    master["currently_stocked_out"]

)

for column in ["stockout", "below_safety_stock"]:

    projection[column] = as_bool(projection[column])

for df in [projection, forecasts]:

    df["forecast_week"] = pd.to_datetime(

        df["forecast_week"],

        errors="coerce",

    )

if (

    projection["forecast_week"].isna().any()

    or forecasts["forecast_week"].isna().any()

):

    st.error("Invalid forecast dates detected.")

    st.stop()

# ============================================================

# 4. INVENTORY INTELLIGENCE MASTER

# ============================================================

inventory = master.merge(

    policy[["sku", "safety_stock", "reorder_point"]],

    on="sku",

    how="left",

    validate="one_to_one",

)

inventory = inventory.merge(

    recovery[

        [

            "sku",

            "priority",

            "stockout_weeks",

            "peak_shortage_exposure_value",

        ]

    ],

    on="sku",

    how="left",

    validate="one_to_one",

)

inventory["below_safety_stock_now"] = (

    inventory["net_available"]

    < inventory["safety_stock"]

)

inventory["below_reorder_point"] = (

    inventory["net_available"]

    <= inventory["reorder_point"]

)

inventory["inventory_health"] = np.select(

    [

        inventory["currently_stocked_out"],

        inventory["below_safety_stock_now"],

        inventory["below_reorder_point"],

    ],

    [

        "STOCKOUT",

        "BELOW SAFETY STOCK",

        "REORDER REQUIRED",

    ],

    default="HEALTHY",

)

# Inventory-value ABC classification

inventory = inventory.sort_values(

    "inventory_value",

    ascending=False,

).copy()

total_inventory_value = (

    inventory["inventory_value"].clip(lower=0).sum()

)

if total_inventory_value > 0:

    inventory["cumulative_value_pct"] = (

        inventory["inventory_value"]

        .clip(lower=0)

        .cumsum()

        / total_inventory_value

        * 100

    )

    prior_pct = (

        inventory["cumulative_value_pct"]

        - inventory["inventory_value"].clip(lower=0)

        / total_inventory_value

        * 100

    )

    inventory["abc_class"] = np.select(

        [

            prior_pct < 80,

            prior_pct < 95,

        ],

        ["A", "B"],

        default="C",

    )

else:

    inventory["cumulative_value_pct"] = 0.0

    inventory["abc_class"] = "C"

# ============================================================

# 5. SIDEBAR NAVIGATION

# ============================================================

st.sidebar.title("ðŸ“¦ Procurement AI")

st.sidebar.caption(

    "Intelligent Supply Planning Command Center"

)

page = st.sidebar.radio(

    "Workspace",

    [

        "Executive Overview",

        "Inventory Health",

        "Demand Forecasting",

        "SKU Drilldown",

        "ABC Analysis",

        "Supplier Intelligence",

        "Procurement Orders",

        "AI Procurement Assistant",
        "Planning Analytics",
        "Data Quality & Sources",

    ],

)

st.sidebar.divider()

supplier_options = sorted(

    master["supplier_id"].dropna().unique().tolist()

)

selected_suppliers = st.sidebar.multiselect(

    "Suppliers",

    supplier_options,

    default=supplier_options,

)

category_options = sorted(

    master["category"].dropna().unique().tolist()

)

selected_categories = st.sidebar.multiselect(

    "Product Categories",

    category_options,

    default=category_options,

)

st.sidebar.divider()

st.sidebar.caption(

    "Synthetic data | 26-week planning horizon"

)

# ============================================================

# 6. GLOBAL FILTERS

# ============================================================

filtered_inventory = inventory[

    inventory["supplier_id"].isin(selected_suppliers)

    & inventory["category"].isin(selected_categories)

].copy()

selected_skus = set(filtered_inventory["sku"])

fm = master[

    master["sku"].isin(selected_skus)

].copy()

fa = actions[

    actions["sku"].isin(selected_skus)

].copy()

fp = projection[

    projection["sku"].isin(selected_skus)

].copy()

fr = recovery[

    recovery["sku"].isin(selected_skus)

].copy()

ff = forecasts[

    forecasts["sku"].isin(selected_skus)

].copy()

supplier_risk = (

    fp.merge(

        fm[["sku", "supplier_id"]],

        on="sku",

        how="left",

        validate="many_to_one",

    )

    .groupby("supplier_id", as_index=False)

    .agg(

        stockout_sku_weeks=("stockout", "sum"),

        safety_breach_sku_weeks=(

            "below_safety_stock", "sum"

        ),

    )

)

if filtered_inventory.empty:

    st.warning(

        "No SKUs match the current filters. "

        "Select a supplier and category."

    )

    st.stop()

# ============================================================

# 7. SHARED HEADER

# ============================================================

st.title("Procurement AI Command Center")

st.caption(

    "Inventory Intelligence | Demand Forecasting | "

    "Supplier Risk | Purchasing Optimization"

)

# ============================================================

# 8. EXECUTIVE OVERVIEW

# ============================================================

if page == "Executive Overview":

    st.subheader("Executive Performance Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("SKUs in Scope", f"{len(fm):,}")

    c2.metric(

        "Inventory Value",

        f"${fm['inventory_value'].sum():,.0f}",

    )

    c3.metric(

        "Recommended Purchasing",

        f"${fa['recommended_order_value'].sum():,.0f}",

    )

    c4.metric(

        "Critical Recovery SKUs",

        f"{(fr['priority'] == 'CRITICAL').sum():,}",

    )

    st.divider()

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(

        "Consolidated Order Lines",

        f"{len(fa):,}",

    )

    c6.metric(

        "Late-to-Need Orders",

        f"{(fa['receipt_delay_days'] > 0).sum():,}",

    )

    c7.metric(

        "Projected Stockout SKU-Weeks",

        f"{fp['stockout'].sum():,}",

    )

    c8.metric(

        "Safety Stock Breach SKU-Weeks",

        f"{fp['below_safety_stock'].sum():,}",

    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Recovery Priorities")

        priority_order = [

            "CRITICAL",

            "HIGH",

            "MEDIUM",

            "WATCH",

            "NO EXCEPTION",

        ]

        counts = (

            fr["priority"]

            .value_counts()

            .reindex(priority_order, fill_value=0)

            .reset_index()

        )

        counts.columns = ["Priority", "SKUs"]

        fig = px.bar(

            counts,

            x="Priority",

            y="SKUs",

            text="SKUs",

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

    with right:

        st.subheader("Supplier Shortage Exposure")

        chart = supplier_risk.sort_values(

            "stockout_sku_weeks",

            ascending=False,

        )

        fig = px.bar(

            chart,

            x="supplier_id",

            y="stockout_sku_weeks",

            labels={

                "supplier_id": "Supplier",

                "stockout_sku_weeks": "Stockout SKU-Weeks",

            },

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

    st.divider()

    st.subheader("Weekly Projected Inventory Risk")

    weekly = (

        fp.groupby("forecast_week", as_index=False)

        .agg(

            stockout_skus=("stockout", "sum"),

            safety_breach_skus=(

                "below_safety_stock", "sum"

            ),

        )

    )

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=weekly["forecast_week"],

            y=weekly["stockout_skus"],

            name="Projected Stockouts",

            mode="lines+markers",

        )

    )

    fig.add_trace(

        go.Scatter(

            x=weekly["forecast_week"],

            y=weekly["safety_breach_skus"],

            name="Below Safety Stock",

            mode="lines+markers",

        )

    )

    fig.update_layout(

        xaxis_title="Forecast Week",

        yaxis_title="Number of SKUs",

        hovermode="x unified",

    )

    st.plotly_chart(

        fig,

        use_container_width=True,

    )

    st.divider()

    st.subheader("Top 20 Recovery Priorities")

    st.dataframe(

        fr.sort_values(

            "peak_shortage_exposure_value",

            ascending=False,

        ).head(20),

        use_container_width=True,

        hide_index=True,

    )

# ============================================================

# 9. INVENTORY HEALTH

# ============================================================

elif page == "Inventory Health":

    st.header("Inventory Health")

    st.caption(

        "Current inventory position compared with "

        "planning safety stock and reorder points."

    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "On-Hand Units",

        f"{filtered_inventory['on_hand'].sum():,.0f}",

    )

    c2.metric(

        "Allocated Units",

        f"{filtered_inventory['allocated'].sum():,.0f}",

    )

    c3.metric(

        "Net Available Units",

        f"{filtered_inventory['net_available'].sum():,.0f}",

    )

    c4.metric(

        "Currently Stocked-Out SKUs",

        f"{filtered_inventory['currently_stocked_out'].sum():,}",

    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Inventory Health Distribution")

        counts = (

            filtered_inventory["inventory_health"]

            .value_counts()

            .reset_index()

        )

        counts.columns = ["Status", "SKUs"]

        fig = px.pie(

            counts,

            names="Status",

            values="SKUs",

            hole=0.55,

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

    with right:

        st.subheader("Inventory Value by Category")

        category_value = (

            filtered_inventory

            .groupby("category", as_index=False)[

                "inventory_value"

            ]

            .sum()

            .sort_values(

                "inventory_value",

                ascending=False,

            )

        )

        fig = px.bar(

            category_value,

            x="category",

            y="inventory_value",

            labels={

                "category": "Category",

                "inventory_value": "Inventory Value ($)",

            },

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

    st.divider()

    st.subheader("Inventory Health Detail")

    health_options = [

        "ALL",

        "STOCKOUT",

        "BELOW SAFETY STOCK",

        "REORDER REQUIRED",

        "HEALTHY",

    ]

    health_filter = st.selectbox(

        "Inventory Status",

        health_options,

    )

    detail = filtered_inventory.copy()

    if health_filter != "ALL":

        detail = detail[

            detail["inventory_health"] == health_filter

        ]

    columns = [

        "sku",

        "product_name",

        "category",

        "supplier_id",

        "on_hand",

        "allocated",

        "net_available",

        "safety_stock",

        "reorder_point",

        "current_weeks_of_supply",

        "inventory_health",

        "priority",

    ]

    st.dataframe(

        detail[columns],

        use_container_width=True,

        hide_index=True,

    )

    st.download_button(

        "Export Inventory Health CSV",

        detail[columns].to_csv(index=False),

        file_name="inventory_health_report.csv",

        mime="text/csv",

    )

# ============================================================

# 10. DEMAND FORECASTING

# ============================================================

elif page == "Demand Forecasting":

    st.header("Demand Forecasting")

    st.caption(

        "Production demand forecasts across "

        "the 26-week planning horizon."

    )

    total_forecast = ff["forecast_demand"].sum()

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "26-Week Forecast Units",

        f"{total_forecast:,.0f}",

    )

    c2.metric(

        "Forecasted SKUs",

        f"{ff['sku'].nunique():,}",

    )

    c3.metric(

        "Forecast Weeks",

        f"{ff['forecast_week'].nunique():,}",

    )

    st.divider()

    st.subheader("Aggregate Weekly Demand Forecast")

    weekly_demand = (

        ff.groupby("forecast_week", as_index=False)[

            "forecast_demand"

        ]

        .sum()

        .sort_values("forecast_week")

    )

    fig = px.line(

        weekly_demand,

        x="forecast_week",

        y="forecast_demand",

        markers=True,

        labels={

            "forecast_week": "Forecast Week",

            "forecast_demand": "Forecast Units",

        },

    )

    st.plotly_chart(

        fig,

        use_container_width=True,

    )

    st.divider()

    st.subheader("Forecast Demand by Category")

    forecast_categories = ff.merge(

        fm[["sku", "category"]],

        on="sku",

        how="left",

        validate="many_to_one",

    )

    category_forecast = (

        forecast_categories

        .groupby("category", as_index=False)[

            "forecast_demand"

        ]

        .sum()

        .sort_values(

            "forecast_demand",

            ascending=False,

        )

    )

    fig = px.bar(

        category_forecast,

        x="category",

        y="forecast_demand",

        labels={

            "category": "Product Category",

            "forecast_demand": "26-Week Demand Units",

        },

    )

    st.plotly_chart(

        fig,

        use_container_width=True,

    )

    st.divider()

    st.subheader("Individual SKU Forecast")

    sku_choice = st.selectbox(

        "Select SKU",

        sorted(ff["sku"].unique()),

        key="forecast_sku",

    )

    sku_forecast = ff[

        ff["sku"] == sku_choice

    ].sort_values("forecast_week")

    fig = px.line(

        sku_forecast,

        x="forecast_week",

        y="forecast_demand",

        markers=True,

        title=f"26-Week Demand Forecast â€” {sku_choice}",

    )

    st.plotly_chart(

        fig,

        use_container_width=True,

    )

    st.dataframe(

        sku_forecast,

        use_container_width=True,

        hide_index=True,

    )

# ============================================================

# 11. SKU DRILLDOWN

# ============================================================

elif page == "SKU Drilldown":

    st.header("SKU Intelligence")

    sku = st.selectbox(

        "Select a Product",

        sorted(filtered_inventory["sku"].unique()),

        key="drilldown_sku",

    )

    item = filtered_inventory[

        filtered_inventory["sku"] == sku

    ].iloc[0]

    st.subheader(

        f"{sku} â€” {item['product_name']}"

    )

    st.caption(

        f"Category: {item['category']} | "

        f"Supplier: {item['supplier_id']} | "

        f"Recovery Priority: {item['priority']}"

    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "On-Hand",

        f"{item['on_hand']:,.0f}",

    )

    c2.metric(

        "Net Available",

        f"{item['net_available']:,.0f}",

    )

    c3.metric(

        "Safety Stock",

        f"{item['safety_stock']:,.0f}",

    )

    c4.metric(

        "Reorder Point",

        f"{item['reorder_point']:,.0f}",

    )

    st.divider()

    sku_forecast = ff[

        ff["sku"] == sku

    ].sort_values("forecast_week")

    sku_projection = fp[

        fp["sku"] == sku

    ].sort_values("forecast_week")

    st.subheader("Demand Forecast & Inventory Projection")

    fig = go.Figure()

    fig.add_trace(

        go.Scatter(

            x=sku_forecast["forecast_week"],

            y=sku_forecast["forecast_demand"],

            name="Weekly Demand Forecast",

            mode="lines+markers",

            yaxis="y2",

        )

    )

    fig.add_trace(

        go.Scatter(

            x=sku_projection["forecast_week"],

            y=sku_projection["ending_inventory"],

            name="Projected Ending Inventory",

            mode="lines+markers",

        )

    )

    fig.add_hline(

        y=float(item["safety_stock"]),

        line_dash="dash",

        annotation_text="Safety Stock",

    )

    fig.add_hline(

        y=0,

        line_dash="dot",

        annotation_text="Zero Inventory",

    )

    fig.update_layout(

        xaxis_title="Forecast Week",

        yaxis=dict(

            title="Projected Inventory Units",

        ),

        yaxis2=dict(

            title="Weekly Demand Units",

            overlaying="y",

            side="right",

        ),

        hovermode="x unified",

    )

    st.plotly_chart(

        fig,

        use_container_width=True,

    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "Projected Stockout Weeks",

        f"{sku_projection['stockout'].sum():,}",

    )

    c2.metric(

        "Safety Stock Breach Weeks",

        f"{sku_projection['below_safety_stock'].sum():,}",

    )

    c3.metric(

        "26-Week Forecast Demand",

        f"{sku_forecast['forecast_demand'].sum():,.0f}",

    )

    st.subheader("SKU Purchase Recommendations")

    sku_actions = fa[

        fa["sku"] == sku

    ].copy()

    if sku_actions.empty:

        st.info("No new purchasing recommendations.")

    else:

        st.dataframe(

            sku_actions,

            use_container_width=True,

            hide_index=True,

        )

    st.subheader("Projected Inventory by Week")

    st.dataframe(

        sku_projection,

        use_container_width=True,

        hide_index=True,

    )

# ============================================================

# 12. ABC INVENTORY ANALYSIS

# ============================================================

elif page == "ABC Analysis":

    st.header("ABC Inventory Analysis")

    st.caption(

        "ABC classification is based on current inventory "

        "value across all 200 SKUs. Global classifications "

        "remain unchanged when filters are applied."

    )

    abc_summary = (

        filtered_inventory

        .groupby("abc_class", as_index=False)

        .agg(

            sku_count=("sku", "nunique"),

            inventory_value=("inventory_value", "sum"),

        )

    )

    abc_summary["abc_class"] = pd.Categorical(

        abc_summary["abc_class"],

        categories=["A", "B", "C"],

        ordered=True,

    )

    abc_summary = abc_summary.sort_values("abc_class")

    c1, c2, c3 = st.columns(3)

    for col, abc in zip(

        [c1, c2, c3],

        ["A", "B", "C"],

    ):

        count = int(

            (

                filtered_inventory["abc_class"] == abc

            ).sum()

        )

        col.metric(

            f"Class {abc} SKUs",

            f"{count:,}",

        )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Inventory Value by ABC Class")

        fig = px.bar(

            abc_summary,

            x="abc_class",

            y="inventory_value",

            text="sku_count",

            labels={

                "abc_class": "ABC Class",

                "inventory_value": "Inventory Value ($)",

            },

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

    with right:

        st.subheader("SKU Distribution")

        fig = px.pie(

            abc_summary,

            names="abc_class",

            values="sku_count",

            hole=0.55,

        )

        st.plotly_chart(

            fig,

            use_container_width=True,

        )

    st.divider()

    st.subheader("ABC Product Details")

    abc_filter = st.multiselect(

        "ABC Classes",

        ["A", "B", "C"],

        default=["A", "B", "C"],

    )

    detail = filtered_inventory[

        filtered_inventory["abc_class"].isin(abc_filter)

    ]

    st.dataframe(

        detail[

            [

                "sku",

                "product_name",

                "category",

                "supplier_id",

                "inventory_value",

                "abc_class",

                "cumulative_value_pct",

                "inventory_health",

                "priority",

            ]

        ],

        use_container_width=True,

        hide_index=True,

    )

# ============================================================

# 13. SUPPLIER INTELLIGENCE - PHASE 8C

# ============================================================

elif page == "Supplier Intelligence":

    supplier_file = DATA / "supplier_intelligence.csv"

    if not supplier_file.exists():

        st.error("Supplier intelligence dataset not found.")

        st.stop()

    supplier_data = read_csv(

        str(supplier_file),

        supplier_file.stat().st_mtime,

    )

    show_supplier_intelligence(

        supplier_data,

        selected_suppliers=selected_suppliers,

    )

# ============================================================

# 14. PROCUREMENT ORDERS - PHASE 8D

# ============================================================

elif page == "Procurement Orders":

    show_procurement_orders(

        actions,

        selected_suppliers=selected_suppliers,

        selected_categories=selected_categories,

    )

# ============================================================

# 15. DATA QUALITY & SOURCES

# ============================================================

elif page == "AI Procurement Assistant":
    show_procurement_ai()


# ============================================================
# 16. DATA QUALITY & SOURCES
# ============================================================

elif page == "Planning Analytics":
    show_planning_analytics(
        selected_suppliers=selected_suppliers,
        selected_categories=selected_categories,
    )

elif page == "Data Quality & Sources":

    st.header("Data Quality & Sources")

    st.success(

        "All required files, columns, and SKU keys "

        "passed dashboard startup validation."

    )

    rows = []

    for key, df in data.items():

        rows.append({

            "Dataset": FILES[key],

            "Rows": len(df),

            "Unique SKUs": (

                df["sku"].nunique()

                if "sku" in df.columns

                else None

            ),

        })

    st.dataframe(

        pd.DataFrame(rows),

        use_container_width=True,

        hide_index=True,

    )

    st.divider()

    st.subheader("Planning Reconciliation")

    checks = {

        "Planning SKUs": master["sku"].nunique(),

        "Forecast Records": len(forecasts),

        "Forecast Weeks": forecasts["forecast_week"].nunique(),

        "Consolidated Order Lines": len(actions),

        "Recovery SKUs": recovery["sku"].nunique(),

        "Inventory Projection Rows": len(projection),

        "Recommended Purchasing Value":

            f"${actions['recommended_order_value'].sum():,.2f}",

    }

    reconciliation_df = pd.DataFrame({

        "Metric": list(checks.keys()),

        "Value": [

            str(value)

            for value in checks.values()

        ],

    })

    st.dataframe(

        reconciliation_df,

        use_container_width=True,

        hide_index=True,

    )

    st.info(

        "Inventory forecasts and recovery recommendations "

        "are planning simulations, not confirmed supplier "

        "delivery commitments or actual ERP transactions."

    )

# ============================================================

# FOOTER

# ============================================================

st.divider()

st.caption(

    "Procurement AI | Phase 10C | "

    "Synthetic Planning Data | "

    "Inventory, Supplier & Procurement Intelligence"

)



