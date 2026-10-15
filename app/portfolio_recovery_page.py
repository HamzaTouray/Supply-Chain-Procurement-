
"""
Portfolio Recovery Optimizer - Streamlit View

Read-only visualization of synthetic recovery scenarios.
No supplier confirmations, PO changes, or ERP actions.
"""

import pandas as pd
import plotly.express as px
import streamlit as st


def show_portfolio_recovery(data, skus):
    st.subheader("Portfolio Recovery Optimizer")

    st.caption(
        "Combined existing-PO recovery scenarios | "
        "Synthetic data | Supplier feasibility unverified"
    )

    actions = data["portfolio_actions"].copy()
    summary = data["portfolio_summary"].copy()
    projection = data["portfolio_projection"].copy()
    master = data["master"][
        ["sku", "supplier_id", "category"]
    ].drop_duplicates("sku")

    actions = actions[actions["sku"].isin(skus)].copy()
    summary = summary[summary["sku"].isin(skus)].copy()
    projection = projection[
        projection["sku"].isin(skus)
    ].copy()

    actions = actions.merge(
        master,
        on="sku",
        how="left",
        validate="many_to_one",
    )

    summary = summary.merge(
        master,
        on="sku",
        how="left",
        validate="one_to_one",
    )

    if summary.empty:
        st.info("No SKUs match the current filters.")
        return

    selected_actions = len(actions)

    affected_skus = int(
        (summary["selected_po_actions"] > 0).sum()
    )

    avoided_weeks = int(
        summary["stockout_weeks_avoided"].sum()
    )

    negative_reduction = float(
        summary["negative_balance_reduction"].sum()
    )

    premium = float(
        summary[
            "illustrative_total_freight_premium"
        ].sum()
    )

    columns = st.columns(5)

    columns[0].metric(
        "Selected PO Actions",
        f"{selected_actions:,}",
    )

    columns[1].metric(
        "SKUs With Recovery",
        f"{affected_skus:,}",
    )

    columns[2].metric(
        "Stockout Weeks Avoided",
        f"{avoided_weeks:,}",
    )

    columns[3].metric(
        "Negative Balance Reduction",
        f"{negative_reduction:,.1f}",
        help="Unit-weeks of projected negative inventory",
    )

    columns[4].metric(
        "Illustrative Freight Premium",
        f"${premium:,.0f}",
        help="Assumed cost, not an actual freight quote",
    )

    st.divider()

    st.markdown("#### Recovery actions by supplier")

    supplier_summary = (
        summary.groupby("supplier_id", as_index=False)
        .agg(
            selected_actions=(
                "selected_po_actions",
                "sum",
            ),
            stockout_weeks_avoided=(
                "stockout_weeks_avoided",
                "sum",
            ),
            negative_balance_reduction=(
                "negative_balance_reduction",
                "sum",
            ),
            illustrative_premium=(
                "illustrative_total_freight_premium",
                "sum",
            ),
        )
    )

    supplier_summary = supplier_summary[
        supplier_summary["selected_actions"] > 0
    ]

    if not supplier_summary.empty:
        chart = px.bar(
            supplier_summary.sort_values(
                "stockout_weeks_avoided",
                ascending=False,
            ),
            x="supplier_id",
            y="stockout_weeks_avoided",
            hover_data=[
                "selected_actions",
                "negative_balance_reduction",
                "illustrative_premium",
            ],
            labels={
                "supplier_id": "Supplier",
                "stockout_weeks_avoided": (
                    "Modeled Stockout Weeks Avoided"
                ),
            },
        )

        st.plotly_chart(
            chart,
            use_container_width=True,
        )
    else:
        st.info(
            "No selected recovery actions for these suppliers."
        )

    st.markdown("#### Selected purchase order actions")

    if actions.empty:
        st.info(
            "No recovery actions were selected for "
            "the current SKU filters."
        )
    else:
        display_columns = [
            "po_number",
            "sku",
            "supplier_id",
            "category",
            "open_qty",
            "confirmed_in_baseline",
            "supplier_confirmation_required",
            "original_receipt_week",
            "hypothetical_target_week",
            "incremental_negative_balance_reduction",
            "incremental_stockout_weeks_avoided",
            "illustrative_freight_premium",
        ]

        st.dataframe(
            actions[display_columns].sort_values(
                [
                    "incremental_negative_balance_reduction",
                    "sku",
                ],
                ascending=[False, True],
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.markdown("#### SKU recovery comparison")

    available_skus = sorted(
        summary["sku"].dropna().unique().tolist()
    )

    default_sku = (
        summary.sort_values(
            "negative_balance_reduction",
            ascending=False,
        ).iloc[0]["sku"]
    )

    selected_sku = st.selectbox(
        "Select SKU to compare inventory projections",
        available_skus,
        index=available_skus.index(default_sku),
        key="portfolio_recovery_sku",
    )

    sku_summary = summary[
        summary["sku"] == selected_sku
    ].iloc[0]

    sku_projection = projection[
        projection["sku"] == selected_sku
    ].copy()

    sku_projection["forecast_week"] = pd.to_datetime(
        sku_projection["forecast_week"]
    )

    sku_projection = sku_projection.sort_values(
        "forecast_week"
    )

    comparison = sku_projection.melt(
        id_vars=["forecast_week"],
        value_vars=[
            "baseline_ending_inventory",
            "scenario_ending_inventory",
        ],
        var_name="Scenario",
        value_name="Ending Inventory",
    )

    comparison["Scenario"] = comparison[
        "Scenario"
    ].replace(
        {
            "baseline_ending_inventory": (
                "Baseline"
            ),
            "scenario_ending_inventory": (
                "Combined Recovery Scenario"
            ),
        }
    )

    chart = px.line(
        comparison,
        x="forecast_week",
        y="Ending Inventory",
        color="Scenario",
        markers=True,
        labels={
            "forecast_week": "Forecast Week",
        },
        title=f"{selected_sku}: Baseline vs Recovery",
    )

    chart.add_hline(
        y=0,
        line_dash="dash",
        annotation_text="Stockout Threshold",
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Baseline Stockout Weeks",
        int(sku_summary["baseline_stockout_weeks"]),
    )

    col2.metric(
        "Recovery Stockout Weeks",
        int(sku_summary["optimized_stockout_weeks"]),
    )

    col3.metric(
        "Stockout Weeks Avoided",
        int(sku_summary["stockout_weeks_avoided"]),
    )

    st.markdown("#### SKU recovery details")

    st.dataframe(
        summary[
            [
                "sku",
                "supplier_id",
                "category",
                "eligible_po_actions",
                "selected_po_actions",
                "baseline_stockout_weeks",
                "optimized_stockout_weeks",
                "stockout_weeks_avoided",
                "negative_balance_reduction",
                "illustrative_total_freight_premium",
                "supplier_confirmation_required",
            ]
        ].sort_values(
            "negative_balance_reduction",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "These results are hypothetical and read-only. "
        "Uncertain PO receipts are added only for scenario "
        "analysis; confirmed receipts are shifted in time. "
        "Supplier readiness, actual delivery dates, freight "
        "capacity, and freight quotations have not been "
        "verified. All modeled results use the historical "
        "synthetic planning snapshot."
    )
