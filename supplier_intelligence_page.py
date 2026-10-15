
import pandas as pd
import plotly.express as px
import streamlit as st


def show_supplier_intelligence(
    supplier_data,
    selected_suppliers=None,
):
    st.header("Supplier Performance & Risk Intelligence")

    st.caption(
        "Historical delivery reliability, lead-time performance, "
        "and projected supply risk."
    )

    df = supplier_data.copy()

    if selected_suppliers is not None:
        df = df[
            df["supplier_id"].isin(selected_suppliers)
        ].copy()

    if df.empty:
        st.warning("No suppliers match the selected filters.")
        return

    # ========================================================
    # 1. SUPPLIER PERFORMANCE KPIs
    # ========================================================

    completed = pd.to_numeric(
        df["completed_po_lines"], errors="coerce"
    ).fillna(0)

    on_time = pd.to_numeric(
        df["full_on_time_rate"], errors="coerce"
    )

    valid = on_time.notna() & (completed > 0)

    weighted_on_time = (
        (on_time[valid] * completed[valid]).sum()
        / completed[valid].sum()
        if completed[valid].sum() > 0
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Suppliers", f"{len(df):,}")

    c2.metric(
        "Full On-Time Delivery",
        f"{weighted_on_time:.1%}",
    )

    c3.metric(
        "Projected Stockout SKU-Weeks",
        f"{df['total_stockout_sku_weeks'].sum():,.0f}",
    )

    c4.metric(
        "Historical PO Value",
        f"${df['historical_po_value'].sum():,.0f}",
    )

    st.divider()

    # ========================================================
    # 2. SUPPLIER DELIVERY RELIABILITY
    # ========================================================

    st.subheader("Supplier Delivery Reliability")

    reliability = df.sort_values(
        "full_on_time_rate",
        ascending=False,
    ).copy()

    reliability["on_time_pct"] = (
        reliability["full_on_time_rate"] * 100
    )

    fig = px.bar(
        reliability,
        x="supplier_name",
        y="on_time_pct",
        color="supplier_risk_level",
        text="on_time_pct",
        labels={
            "supplier_name": "Supplier",
            "on_time_pct": "Full On-Time Delivery (%)",
            "supplier_risk_level": "Risk Level",
        },
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig.update_layout(
        xaxis_tickangle=-35,
        yaxis_range=[0, 105],
    )

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # 3. LEAD-TIME ANALYSIS
    # ========================================================

    left, right = st.columns(2)

    with left:
        st.subheader("Promised vs Actual Lead Time")

        lead = df[
            [
                "supplier_name",
                "avg_promised_lead_days",
                "avg_actual_full_lead_days",
            ]
        ].melt(
            id_vars="supplier_name",
            var_name="Lead Time Type",
            value_name="Days",
        )

        fig = px.bar(
            lead,
            x="supplier_name",
            y="Days",
            color="Lead Time Type",
            barmode="group",
        )

        fig.update_layout(xaxis_tickangle=-40)

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Projected Stockout Exposure")

        risk = df.sort_values(
            "total_stockout_sku_weeks",
            ascending=False,
        )

        fig = px.bar(
            risk,
            x="supplier_name",
            y="total_stockout_sku_weeks",
            labels={
                "supplier_name": "Supplier",
                "total_stockout_sku_weeks":
                    "Stockout SKU-Weeks",
            },
        )

        fig.update_layout(xaxis_tickangle=-40)

        st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # 4. SUPPLIER PERFORMANCE SCORECARD
    # ========================================================

    st.divider()

    st.subheader("Supplier Performance Scorecard")

    scorecard_columns = [
        "supplier_id",
        "supplier_name",
        "country",
        "completed_po_lines",
        "full_on_time_rate",
        "avg_full_delay_days",
        "partial_shipment_rate",
        "total_stockout_sku_weeks",
        "critical_skus",
        "supplier_risk_level",
    ]

    scorecard = df[scorecard_columns].copy()

    scorecard["full_on_time_rate"] *= 100
    scorecard["partial_shipment_rate"] *= 100

    st.dataframe(
        scorecard,
        use_container_width=True,
        hide_index=True,
        column_config={
            "full_on_time_rate":
                st.column_config.NumberColumn(
                    "Full On-Time (%)",
                    format="%.1f%%",
                ),
            "partial_shipment_rate":
                st.column_config.NumberColumn(
                    "Partial Shipment (%)",
                    format="%.1f%%",
                ),
        },
    )

    # ========================================================
    # 5. SUPPLIER DRILLDOWN
    # ========================================================

    st.divider()

    st.subheader("Supplier Drilldown")

    supplier_id = st.selectbox(
        "Select Supplier",
        sorted(df["supplier_id"].tolist()),
        key="supplier_intelligence_selection",
    )

    supplier = df[
        df["supplier_id"] == supplier_id
    ].iloc[0]

    st.markdown(f"### {supplier['supplier_name']}")

    st.caption(
        f"Country: {supplier['country']} | "
        f"Behavior: {supplier['behavior_type']} | "
        f"Risk: {supplier['supplier_risk_level']}"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Full On-Time Delivery",
        f"{supplier['full_on_time_rate']:.1%}",
    )

    b.metric(
        "Average Full Delay",
        f"{supplier['avg_full_delay_days']:.1f} days",
    )

    c.metric(
        "Critical SKUs",
        f"{supplier['critical_skus']:,.0f}",
    )

    d.metric(
        "Stockout SKU-Weeks",
        f"{supplier['total_stockout_sku_weeks']:,.0f}",
    )

    # ========================================================
    # 6. SUPPLIER RECOVERY EXPOSURE
    # ========================================================

    st.subheader("Supplier Recovery Exposure")

    recovery_columns = [
        "sku_count",
        "inventory_value",
        "historical_po_lines",
        "historical_po_value",
        "late_order_lines",
        "uncertain_po_lines",
        "uncertain_po_units",
        "new_order_value",
    ]

    details = pd.DataFrame({
        "Metric": recovery_columns,
        "Value": [
            str(supplier[col])
            for col in recovery_columns
        ],
    })

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Delivery performance is calculated from completed "
        "historical purchase orders. Projected shortages "
        "come from the Phase 7 planning simulation. "
        "Supplier risk levels are rule-based classifications."
    )
