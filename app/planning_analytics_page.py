
"""
Procurement AI - Planning Analytics
Enhancement 1H

Read-only Streamlit analytics:
1. Recovery Analysis
2. Safety Stock Scenarios
3. Forecast Risk
4. Historical Performance

Synthetic scenario data only.
No purchase orders, approvals, forecasts, or policies
are modified by this page.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from portfolio_recovery_page import show_portfolio_recovery

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"



REPORTS = {
    "master": "planning_master.csv",
    "bridge": "bridge_quantity_analysis.csv",
    "po_recovery": "existing_po_recovery_summary.csv",
    "simulation": "recovery_simulation_results.csv",
    "freight": "freight_recovery_scenarios.csv",
    "safety": "safety_stock_scenarios.csv",
    "risk": "forecast_risk_analysis.csv",
    "performance": "historical_performance_detail.csv",
    "decision_queue": "procurement_decision_queue.csv",
    "portfolio_actions": "portfolio_recovery_actions.csv",
    "portfolio_summary": "portfolio_recovery_summary.csv",
    "portfolio_projection": "portfolio_recovery_projection.csv",
}




@st.cache_data(show_spinner=False)
def read_report(path, modified_time):
    return pd.read_csv(path)


def load_reports():
    data = {}

    for key, filename in REPORTS.items():
        path = DATA_DIR / filename

        if not path.exists():
            st.error(f"Missing analytics report: {filename}")
            st.info(
                "Run the corresponding validated analytics "
                "module before opening Planning Analytics."
            )
            st.stop()

        data[key] = read_report(
            str(path),
            path.stat().st_mtime,
        )

    return data


def metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def apply_sku_filter(df, skus):
    if "sku" not in df.columns:
        return df.copy()

    return df[df["sku"].isin(skus)].copy()


def show_recovery(data, skus):
    st.subheader("Recovery Analysis")

    st.caption(
        "Conditional recovery scenarios, not confirmed "
        "supplier commitments or executable freight decisions."
    )

    bridge = apply_sku_filter(data["bridge"], skus)
    po = apply_sku_filter(data["po_recovery"], skus)
    simulation = apply_sku_filter(data["simulation"], skus)
    freight = apply_sku_filter(data["freight"], skus)

    bridge_needed = bridge[
        bridge["minimum_stockout_bridge_units"] > 0
    ]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric(
            "Bridge-required SKUs",
            f"{len(bridge_needed):,}",
        )

    with c2:
        metric(
            "Minimum bridge units",
            f"{bridge['minimum_stockout_bridge_units'].sum():,.0f}",
        )

    with c3:
        metric(
            "Existing PO units under review",
            f"{po['open_po_units'].sum():,.0f}",
        )

    with c4:
        metric(
            "Simulated PO lines",
            f"{len(simulation):,}",
        )

    st.markdown("#### Bridge Requirements")

    if not bridge.empty:
        chart = (
            bridge.groupby("bridge_status", as_index=False)
            .agg(skus=("sku", "count"))
        )

        st.plotly_chart(
            px.bar(
                chart,
                x="bridge_status",
                y="skus",
                title="Bridge status by SKU",
            ),
            use_container_width=True,
        )

    st.dataframe(
        bridge[
            [
                "sku",
                "priority",
                "bridge_status",
                "minimum_stockout_bridge_units",
                "minimum_safety_bridge_units",
                "first_pre_receipt_stockout_week",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Existing Purchase Order Recovery")

    st.dataframe(
        po[
            [
                "sku",
                "recovery_status",
                "open_po_lines",
                "open_po_units",
                "confirmed_po_units",
                "uncertain_po_units",
                "recovery_units_under_review",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Conditional Recovery Simulation")

    if not simulation.empty:
        simulated = simulation[
            simulation["scenario_simulated"].astype(str)
            .str.lower().isin(["true", "1"])
        ]

        c1, c2 = st.columns(2)

        with c1:
            metric(
                "Independently simulated improvements",
                f"{(simulated['negative_balance_reduction'] > 0).sum():,}",
            )

        with c2:
            metric(
                "Illustrative scenario premiums",
                f"${simulated['illustrative_freight_premium'].sum():,.0f}",
                "Sum of independent hypothetical scenarios; "
                "not an optimized portfolio budget.",
            )

        st.dataframe(
            simulation[
                [
                    "po_number",
                    "sku",
                    "recovery_action",
                    "scenario_status",
                    "open_qty",
                    "stockout_weeks_avoided",
                    "negative_balance_reduction",
                    "illustrative_freight_premium",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Freight Scenario Assessment")

    st.dataframe(
        freight[
            [
                "sku",
                "minimum_stockout_bridge_units",
                "hypothetical_new_order_air_arrival",
                "new_order_air_before_stockout",
                "existing_po_potential_bridge_units",
                "incremental_air_freight_cost",
                "scenario_status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "PO simulations are independent, conditional scenarios. "
        "Their benefits and freight premiums must not be added "
        "together as an optimized purchasing plan. "
        "Weekly receipt timing does not establish intra-week "
        "stock availability."
    )


def show_safety_stock(data, skus):
    st.subheader("Safety Stock Scenarios")

    st.caption(
        "Illustrative service-level policy comparisons. "
        "Inventory value changes are not proven savings."
    )

    safety = apply_sku_filter(data["safety"], skus)

    if safety.empty:
        st.info("No SKUs match the current filters.")
        return

    summary = (
        safety.groupby("scenario_service_level", as_index=False)
        .agg(
            skus=("sku", "nunique"),
            scenario_units=("scenario_safety_stock", "sum"),
            scenario_value=("scenario_safety_stock_value", "sum"),
            baseline_value=("baseline_safety_stock_value", "sum"),
            value_change=("safety_stock_value_change", "sum"),
        )
        .sort_values("scenario_service_level")
    )

    summary["service_level_label"] = (
        summary["scenario_service_level"]
        .mul(100)
        .round(0)
        .astype(int)
        .astype(str)
        + "%"
    )

    baseline = (
        safety.groupby("sku", as_index=False)
        ["baseline_safety_stock_value"].first()
    )

    baseline_value = baseline[
        "baseline_safety_stock_value"
    ].sum()

    c1, c2 = st.columns(2)

    with c1:
        metric(
            "Baseline safety stock value",
            f"${baseline_value:,.2f}",
        )

    with c2:
        metric(
            "SKUs analyzed",
            f"{safety['sku'].nunique():,}",
        )

    st.plotly_chart(
        px.bar(
            summary,
            x="service_level_label",
            y="scenario_value",
            title="Modeled safety stock value by service level",
            labels={
                "service_level_label": "Service level",
                "scenario_value": "Inventory value",
            },
        ),
        use_container_width=True,
    )

    st.dataframe(
        summary[
            [
                "service_level_label",
                "skus",
                "scenario_units",
                "scenario_value",
                "baseline_value",
                "value_change",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    levels = sorted(
        safety["scenario_service_level"].dropna().unique()
    )

    selected_level = st.selectbox(
        "Inspect service-level scenario",
        levels,
        format_func=lambda x: f"{x:.0%}",
    )

    selected = safety[
        safety["scenario_service_level"] == selected_level
    ]

    st.dataframe(
        selected[
            [
                "sku",
                "final_demand_class",
                "baseline_service_level",
                "scenario_service_level",
                "baseline_safety_stock",
                "scenario_safety_stock",
                "safety_stock_unit_change",
                "safety_stock_value_change",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Reducing safety stock can increase stockout risk. "
        "Any proposed policy change requires independent "
        "service-level and stockout validation."
    )


def show_forecast_risk(data, skus):
    st.subheader("Forecast Risk")

    risk = apply_sku_filter(data["risk"], skus)

    if risk.empty:
        st.info("No SKUs match the current filters.")
        return

    critical = (risk["risk_tier"] == "CRITICAL").sum()
    high = (risk["risk_tier"] == "HIGH").sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric("Critical-risk SKUs", f"{critical:,}")

    with c2:
        metric("High-risk SKUs", f"{high:,}")

    with c3:
        metric(
            "Material underforecast",
            f"{risk['material_underforecast'].sum():,}",
        )

    with c4:
        metric(
            "High forecast error",
            f"{risk['high_forecast_error'].sum():,}",
        )

    counts = (
        risk.groupby("risk_tier", as_index=False)
        .agg(skus=("sku", "count"))
    )

    order = ["CRITICAL", "HIGH", "MODERATE", "LOW"]

    st.plotly_chart(
        px.bar(
            counts,
            x="risk_tier",
            y="skus",
            category_orders={"risk_tier": order},
            title="Forecast risk distribution",
        ),
        use_container_width=True,
    )

    tiers = st.multiselect(
        "Risk tiers",
        order,
        default=order,
    )

    visible = risk[
        risk["risk_tier"].isin(tiers)
    ].sort_values(
        ["risk_score", "selected_wape"],
        ascending=[False, False],
    )

    st.dataframe(
        visible[
            [
                "sku",
                "risk_tier",
                "risk_score",
                "final_demand_class",
                "selected_model",
                "selected_wape",
                "selected_bias",
                "planning_lead_time_weeks",
                "available_coverage_weeks",
                "recommended_review_action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Risk scores are heuristic prioritization indicators, "
        "not validated probabilities of stockout."
    )


def show_historical_performance(data, skus):
    st.subheader("Historical Forecast Performance")

    performance = apply_sku_filter(
        data["performance"], skus
    )

    if performance.empty:
        st.info("No backtested SKUs match the filters.")
        return

    selected_actual = performance[
        "selected_actual_units"
    ].sum()

    naive_actual = performance[
        "naive_actual_units"
    ].sum()

    if selected_actual > 0:
        selected_wape = (
            performance["selected_wape"]
            * performance["selected_actual_units"]
        ).sum() / selected_actual
    else:
        selected_wape = float("nan")

    if naive_actual > 0:
        naive_wape = (
            performance["naive_wape"]
            * performance["naive_actual_units"]
        ).sum() / naive_actual
    else:
        naive_wape = float("nan")

    better = (
        performance["wape_result"] == "BETTER"
    ).sum()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric(
            "Selected model WAPE",
            f"{selected_wape:.2%}",
        )

    with c2:
        metric(
            "Naive baseline WAPE",
            f"{naive_wape:.2%}",
        )

    with c3:
        metric(
            "Improvement",
            f"{(naive_wape - selected_wape) * 100:.2f} pp",
        )

    with c4:
        metric(
            "SKUs outperforming naive",
            f"{better:,} / {len(performance):,}",
        )

    groups = []

    for demand_class, group in performance.groupby(
        "final_demand_class"
    ):
        actual = group["selected_actual_units"].sum()

        if actual <= 0:
            continue

        groups.append(
            {
                "Demand class": demand_class,
                "Selected model": (
                    group["selected_wape"]
                    * group["selected_actual_units"]
                ).sum() / actual * 100,
                "Naive baseline": (
                    group["naive_wape"]
                    * group["naive_actual_units"]
                ).sum() / actual * 100,
            }
        )

    chart_data = pd.DataFrame(groups)

    if not chart_data.empty:
        long_data = chart_data.melt(
            id_vars="Demand class",
            var_name="Forecast approach",
            value_name="WAPE (%)",
        )

        st.plotly_chart(
            px.bar(
                long_data,
                x="Demand class",
                y="WAPE (%)",
                color="Forecast approach",
                barmode="group",
                title="Historical WAPE by demand class",
            ),
            use_container_width=True,
        )

    st.dataframe(
        performance[
            [
                "sku",
                "final_demand_class",
                "selected_model",
                "selected_wape",
                "naive_wape",
                "wape_improvement_pp",
                "selected_bias",
                "wape_result",
                "evaluation_status",
            ]
        ].sort_values(
            "wape_improvement_pp",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "These are model-selection rolling backtests, "
        "not independent holdout results. Improvements "
        "do not establish reductions in stockouts, "
        "inventory investment, or freight costs."
    )


def show_decision_queue(data, skus):
    st.subheader("Procurement Decision Queue")

    st.caption(
        "Ranked procurement actions based on the synthetic "
        "September 30, 2026 planning snapshot. "
        "Recommendations are not live purchasing instructions."
    )

    queue = apply_sku_filter(
        data["decision_queue"], skus
    )

    if queue.empty:
        st.info("No SKUs match the selected filters.")
        return

    priorities = ["P1", "P2", "P3", "P4"]

    selected_priorities = st.multiselect(
        "Decision priority",
        priorities,
        default=priorities,
        key="decision_queue_priority_filter",
    )

    categories = sorted(
        queue["decision_category"].dropna().unique()
    )

    selected_actions = st.multiselect(
        "Decision category",
        categories,
        default=categories,
        key="decision_queue_category_filter",
    )

    queue = queue[
        queue["decision_priority"].isin(selected_priorities)
        & queue["decision_category"].isin(selected_actions)
    ].copy()

    if queue.empty:
        st.info("No decisions match the selected priorities.")
        return

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("SKUs in queue", f"{len(queue):,}")

    with c2:
        st.metric(
            "P1 Immediate",
            f"{(queue['decision_priority'] == 'P1').sum():,}",
        )

    with c3:
        st.metric(
            "Stockout bridge units",
            f"{queue['minimum_stockout_bridge_units'].sum():,.0f}",
        )

    with c4:
        st.metric(
            "Supplier confirmations",
            f"{queue['supplier_confirmation_required'].sum():,}",
        )

    st.markdown("#### Priority distribution")

    priority_counts = (
        queue["decision_priority"]
        .value_counts()
        .reindex(priorities, fill_value=0)
        .reset_index()
    )

    priority_counts.columns = ["Priority", "SKUs"]

    fig = px.bar(
        priority_counts,
        x="Priority",
        y="SKUs",
        text="SKUs",
    )

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("#### Ranked procurement actions")

    display_columns = [
        "decision_rank",
        "sku",
        "supplier_id",
        "decision_priority",
        "decision_category",
        "recommended_action",
        "minimum_stockout_bridge_units",
        "weeks_to_first_stockout",
        "risk_tier",
        "forecast_warning",
        "conditional_air_option",
    ]

    st.dataframe(
        queue[display_columns].sort_values("decision_rank"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### SKU decision details")

    sku = st.selectbox(
        "Select SKU",
        queue.sort_values("decision_rank")["sku"].tolist(),
        key="decision_queue_sku",
    )

    record = queue.loc[
        queue["sku"] == sku
    ].iloc[0]

    st.write(
        f"**Priority:** {record['decision_priority']} "
        f"| **Category:** {record['decision_category']}"
    )

    st.write(
        f"**Recommended action:** "
        f"{record['recommended_action']}"
    )

    st.write(
        f"**Reason:** {record['decision_reason']}"
    )

    st.write(
        f"**Existing PO recovery:** "
        f"{record['recovery_status']}"
    )

    st.write(
        f"**Minimum stockout bridge:** "
        f"{record['minimum_stockout_bridge_units']:,.0f} units"
    )

    if pd.notna(record["first_stockout_date"]):
        st.write(
            f"**First projected stockout week:** "
            f"{str(record['first_stockout_date'])[:10]}"
        )

    st.write(
        f"**Forecast risk:** {record['risk_tier']} "
        f"({record['forecast_warning']})"
    )

    if record["conditional_air_option"]:
        st.info(
            "A hypothetical new air order may arrive before "
            "the first projected stockout. Supplier readiness, "
            "receipt timing, and actual shortage prevention "
            "have not been confirmed."
        )

    st.warning(
        "Decision rankings are heuristic and relative to the "
        "synthetic September 30, 2026 snapshot. "
        "Verify supplier commitments, inventory, demand, "
        "and delivery feasibility before acting."
    )

def show_planning_analytics(
    selected_suppliers=None,
    selected_categories=None,
):
    st.header("Planning Analytics")

    st.caption(
        "Procurement decision-support diagnostics | "
        "Synthetic data | Read-only"
    )

    data = load_reports()
    master = data["master"]

    required = {
        "sku", "supplier_id", "category"
    }

    if not required.issubset(master.columns):
        st.error("Planning master is missing filter fields.")
        st.stop()

    filtered = master.copy()

    if selected_suppliers is not None:
        filtered = filtered[
            filtered["supplier_id"].isin(
                selected_suppliers
            )
        ]

    if selected_categories is not None:
        filtered = filtered[
            filtered["category"].isin(
                selected_categories
            )
        ]

    sku_options = sorted(
        filtered["sku"].dropna().unique().tolist()
    )

    selected_skus = st.multiselect(
        "SKUs (leave empty for all matching SKUs)",
        sku_options,
        default=[],
    )

    skus = (
        set(selected_skus)
        if selected_skus
        else set(sku_options)
    )

    st.caption(
        f"Showing {len(skus):,} of "
        f"{master['sku'].nunique():,} SKUs"
    )

    
    tabs = st.tabs(
        [
            "Recovery Analysis",
            "Safety Stock Scenarios",
            "Forecast Risk",
            "Historical Performance",
            "Procurement Decision Queue",
            "Portfolio Recovery Optimizer",
        ]
    )

    with tabs[0]:
        show_recovery(data, skus)

    with tabs[1]:
        show_safety_stock(data, skus)

    with tabs[2]:
        show_forecast_risk(data, skus)

    with tabs[3]:
        show_historical_performance(data, skus)

    with tabs[4]:
        show_decision_queue(data, skus)

    with tabs[5]:
        show_portfolio_recovery(data, skus)