
from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PHASE 11C.3 - PROCUREMENT ORDERS AND APPROVAL WORKFLOW
# ============================================================

ROOT = Path(__file__).resolve().parent
APPROVAL_DIR = ROOT / "src" / "approval"

if str(APPROVAL_DIR) not in sys.path:
    sys.path.insert(0, str(APPROVAL_DIR))

from purchase_order_review import (
    sync_proposals,
    get_review_register,
    update_planner_review,
    get_proposal_audit,
    load_proposals,
    submit_for_approval,
    approve_proposal,
    reject_proposal,
)


REQUIRED_COLUMNS = [
    "sku",
    "product_name",
    "category",
    "supplier_id",
    "recommended_order_qty",
    "recommended_order_value",
    "required_receipt_week",
    "executable_release_date",
    "earliest_feasible_receipt",
    "receipt_delay_days",
    "execution_action",
    "urgency_priority",
    "expedite_required",
    "receipt_feasible_on_time",
]


def as_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


# ============================================================
# PLANNER REVIEW AND APPROVAL WORKFLOW
# ============================================================

def show_planner_review(
    selected_suppliers=None,
    selected_categories=None,
):
    st.divider()
    st.subheader("Planner Review & Approval Workflow")

    st.caption(
        "Review purchasing proposals, adjust draft quantities, "
        "submit for approval, and record approval decisions. "
        "No ERP purchase orders are released."
    )

    try:
        sync_proposals()
        register = get_review_register()
        source = load_proposals()

    except Exception as exc:
        st.error(
            f"Unable to load purchase order reviews: {exc}"
        )
        return

    if register.empty:
        st.warning("No purchase proposals are registered.")
        return

    source = source.reset_index(drop=True).copy()
    source["source_row_number"] = source.index + 1

    source_columns = [
        "source_row_number",
        "sku",
        "supplier_id",
        "product_name",
        "category",
        "execution_action",
        "urgency_priority",
        "receipt_delay_days",
        "required_receipt_week",
    ]

    source_details = source[source_columns].copy()

    try:
        review_df = register.merge(
            source_details,
            on=[
                "source_row_number",
                "sku",
                "supplier_id",
            ],
            how="left",
            validate="one_to_one",
        )

    except Exception as exc:
        st.error(
            f"Unable to match proposal records: {exc}"
        )
        return

    if review_df["category"].isna().any():
        st.error(
            "Some registered proposals do not match "
            "their source records. Approval actions "
            "have been disabled."
        )
        return

    if selected_suppliers is not None:
        review_df = review_df[
            review_df["supplier_id"].isin(
                selected_suppliers
            )
        ].copy()

    if selected_categories is not None:
        review_df = review_df[
            review_df["category"].isin(
                selected_categories
            )
        ].copy()

    if review_df.empty:
        st.info(
            "No registered proposals match your filters."
        )
        return

    # ========================================================
    # APPROVAL SUMMARY
    # ========================================================

    st.markdown("#### Approval Status Overview")

    status_counts = (
        review_df["status"]
        .value_counts()
        .to_dict()
    )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Draft",
        f"{status_counts.get('Draft', 0):,}",
    )

    s2.metric(
        "Pending Approval",
        f"{status_counts.get('Pending Approval', 0):,}",
    )

    s3.metric(
        "Approved",
        f"{status_counts.get('Approved', 0):,}",
    )

    s4.metric(
        "Rejected",
        f"{status_counts.get('Rejected', 0):,}",
    )

    adjusted_count = (
        review_df["reviewed_qty"]
        != review_df["original_qty"]
    ).sum()

    v1, v2, v3 = st.columns(3)

    v1.metric(
        "Total Proposals",
        f"{len(review_df):,}",
    )

    v2.metric(
        "Adjusted Proposals",
        f"{adjusted_count:,}",
    )

    v3.metric(
        "Reviewed Purchasing Value",
        f"${review_df['reviewed_value'].sum():,.0f}",
    )

    # ========================================================
    # APPROVAL REGISTER
    # ========================================================

    st.markdown("#### Purchase Order Review Register")

    status_options = [
        "Draft",
        "Pending Approval",
        "Approved",
        "Rejected",
    ]

    selected_statuses = st.multiselect(
        "Filter by Approval Status",
        options=status_options,
        default=status_options,
        key="approval_status_filter",
    )

    filtered_register = review_df[
        review_df["status"].isin(
            selected_statuses
        )
    ].copy()

    register_columns = [
        "proposal_id",
        "sku",
        "product_name",
        "supplier_id",
        "original_qty",
        "reviewed_qty",
        "original_value",
        "reviewed_value",
        "status",
        "planner_notes",
    ]

    st.dataframe(
        filtered_register[register_columns],
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # SELECT PROPOSAL
    # ========================================================

    st.divider()
    st.markdown("#### Select Purchase Proposal")

    if filtered_register.empty:
        st.info(
            "No proposals match the selected approval statuses."
        )
        return

    proposal_options = {}

    for _, row in filtered_register.iterrows():
        label = (
            f"{row['sku']} | "
            f"{row['supplier_id']} | "
            f"{row['status']} | "
            f"{row['proposal_id']}"
        )

        proposal_options[label] = row["proposal_id"]

    selected_label = st.selectbox(
        "Purchase Proposal",
        options=list(proposal_options.keys()),
        key="approval_proposal_selection",
    )

    proposal_id = proposal_options[selected_label]

    selected = filtered_register.loc[
        filtered_register["proposal_id"] == proposal_id
    ].iloc[0]

    status = str(selected["status"])

    original_qty = int(selected["original_qty"])
    reviewed_qty = int(selected["reviewed_qty"])

    original_value = float(
        selected["original_value"]
    )

    reviewed_value = float(
        selected["reviewed_value"]
    )

    notes_value = selected["planner_notes"]

    if pd.isna(notes_value):
        notes_value = ""

    notes_value = str(notes_value)

    st.caption(f"Proposal ID: {proposal_id}")

    st.write(f"**SKU:** {selected['sku']}")
    st.write(f"**Supplier:** {selected['supplier_id']}")
    st.write(f"**Current Status:** {status}")

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Original Units",
        f"{original_qty:,}",
    )

    q2.metric(
        "Reviewed Units",
        f"{reviewed_qty:,}",
    )

    q3.metric(
        "Original Value",
        f"${original_value:,.2f}",
    )

    q4.metric(
        "Reviewed Value",
        f"${reviewed_value:,.2f}",
    )

    # ========================================================
    # DRAFT PLANNER REVIEW
    # ========================================================

    st.divider()
    st.markdown("#### Planner Review")

    if status == "Draft":

        st.info(
            "This proposal is in Draft status. "
            "The planner can adjust the quantity "
            "and save review notes."
        )

        with st.form(
            key=f"planner_form_{proposal_id}"
        ):

            planner_id = st.text_input(
                "Planner Identifier",
                placeholder="Enter planner name or ID",
            )

            quantity = st.number_input(
                "Reviewed Order Quantity",
                min_value=1,
                value=reviewed_qty,
                step=1,
            )

            planner_notes = st.text_area(
                "Planner Notes",
                value=notes_value,
                max_chars=2000,
                placeholder=(
                    "Explain any adjustments"
                ),
            )

            save_review = st.form_submit_button(
                "Save Planner Review"
            )

        if save_review:
            try:
                result = update_planner_review(
                    proposal_id=proposal_id,
                    reviewed_qty=quantity,
                    planner_notes=planner_notes,
                    changed_by=planner_id,
                )

                if result["updated"]:
                    st.session_state[
                        "approval_feedback"
                    ] = "Planner review saved successfully."
                else:
                    st.session_state[
                        "approval_feedback"
                    ] = "No changes were detected."

                st.rerun()

            except Exception as exc:
                st.error(
                    f"Unable to save planner review: {exc}"
                )

    else:
        st.info(
            "Planner editing is locked because this "
            "proposal is no longer in Draft status."
        )

    # ========================================================
    # SUBMIT FOR APPROVAL
    # ========================================================

    st.divider()
    st.markdown("#### Submit for Approval")

    if status == "Draft":

        st.caption(
            "Submitting a proposal locks planner editing "
            "and moves it to Pending Approval."
        )

        with st.form(
            key=f"submit_form_{proposal_id}"
        ):

            submitting_planner = st.text_input(
                "Submitting Planner Identifier",
                placeholder="Enter planner name or ID",
            )

            confirm_submit = st.checkbox(
                "I have reviewed this proposal and "
                "confirm it is ready for approval."
            )

            submit_clicked = st.form_submit_button(
                "Submit for Approval"
            )

        if submit_clicked:

            if not confirm_submit:
                st.warning(
                    "Confirm the proposal is ready "
                    "before submitting."
                )

            else:
                try:
                    result = submit_for_approval(
                        proposal_id,
                        submitting_planner,
                    )

                    st.session_state[
                        "approval_feedback"
                    ] = (
                        "Proposal submitted successfully. "
                        f"New status: {result['status']}"
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"Submission failed: {exc}"
                    )

    else:
        st.caption(
            "Submission is only available for Draft proposals."
        )

    # ========================================================
    # APPROVAL DECISION
    # ========================================================

    st.divider()
    st.markdown("#### Approval Decision")

    if status == "Pending Approval":

        st.info(
            "This proposal is awaiting an approval decision."
        )

        with st.form(
            key=f"decision_form_{proposal_id}"
        ):

            approver_id = st.text_input(
                "Approver Identifier",
                placeholder="Enter approver name or ID",
            )

            decision = st.radio(
                "Approval Decision",
                options=[
                    "Approve",
                    "Reject",
                ],
                horizontal=True,
            )

            rejection_reason = st.text_area(
                "Rejection Reason",
                placeholder=(
                    "Required if rejecting the proposal"
                ),
                max_chars=2000,
            )

            confirm_decision = st.checkbox(
                "I confirm this approval decision."
            )

            decision_clicked = st.form_submit_button(
                "Record Approval Decision"
            )

        if decision_clicked:

            if not confirm_decision:
                st.warning(
                    "Confirm your decision before proceeding."
                )

            elif not approver_id.strip():
                st.warning(
                    "Approver identifier is required."
                )

            elif (
                decision == "Reject"
                and not rejection_reason.strip()
            ):
                st.warning(
                    "A rejection reason is required."
                )

            else:
                try:

                    if decision == "Approve":

                        result = approve_proposal(
                            proposal_id,
                            approver_id,
                        )

                    else:

                        result = reject_proposal(
                            proposal_id,
                            approver_id,
                            rejection_reason,
                        )

                    st.session_state[
                        "approval_feedback"
                    ] = (
                        "Approval decision recorded. "
                        f"New status: {result['status']}"
                    )

                    st.rerun()

                except Exception as exc:
                    st.error(
                        f"Approval decision failed: {exc}"
                    )

    elif status == "Approved":

        st.success(
            "This proposal has been approved. "
            "No further planner changes are permitted."
        )

    elif status == "Rejected":

        st.warning(
            "This proposal has been rejected. "
            "The decision is recorded in audit history."
        )

    else:
        st.caption(
            "Approval decisions are available only "
            "for Pending Approval proposals."
        )

    # ========================================================
    # AUDIT HISTORY
    # ========================================================

    st.divider()
    st.markdown("#### Proposal Audit History")

    try:
        audit = get_proposal_audit(proposal_id)

        if audit.empty:
            st.info(
                "No audit history has been recorded."
            )

        else:
            audit_columns = [
                "changed_at",
                "action",
                "old_value",
                "new_value",
                "changed_by",
            ]

            st.dataframe(
                audit[audit_columns],
                use_container_width=True,
                hide_index=True,
            )

    except Exception as exc:
        st.error(
            f"Unable to load audit history: {exc}"
        )

    st.caption(
        "Approval identifiers are recorded for auditing. "
        "User authentication and role permissions are "
        "not yet implemented."
    )


# ============================================================
# MAIN PROCUREMENT ORDERS PAGE
# ============================================================

def show_procurement_orders(
    actions,
    selected_suppliers=None,
    selected_categories=None,
):

    st.header("Procurement Order Management")

    st.caption(
        "Review purchasing proposals, execution priorities, "
        "supplier allocations, and delivery feasibility."
    )

    feedback = st.session_state.pop(
        "approval_feedback",
        None,
    )

    if feedback:
        st.success(feedback)

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in actions.columns
    ]

    if missing:
        st.error(
            f"Missing required columns: {missing}"
        )
        return

    df = actions.copy()

    if selected_suppliers is not None:
        df = df[
            df["supplier_id"].isin(
                selected_suppliers
            )
        ].copy()

    if selected_categories is not None:
        df = df[
            df["category"].isin(
                selected_categories
            )
        ].copy()

    if df.empty:
        st.warning(
            "No purchasing proposals match your filters."
        )
        return

    # ========================================================
    # DATA PREPARATION
    # ========================================================

    date_columns = [
        "required_receipt_week",
        "executable_release_date",
        "earliest_feasible_receipt",
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce",
        )

    numeric_columns = [
        "recommended_order_qty",
        "recommended_order_value",
        "receipt_delay_days",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if df[
        [
            "recommended_order_qty",
            "recommended_order_value",
        ]
    ].isna().any().any():

        st.error(
            "Invalid procurement quantities or values."
        )
        return

    df["receipt_delay_days"] = (
        df["receipt_delay_days"].fillna(0)
    )

    df["expedite_required"] = as_bool(
        df["expedite_required"]
    )

    df["receipt_feasible_on_time"] = as_bool(
        df["receipt_feasible_on_time"]
    )

    # ========================================================
    # 1. PROCUREMENT KPIs
    # ========================================================

    late = df["receipt_delay_days"] > 0

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Recommended Order Lines",
        f"{len(df):,}",
    )

    k2.metric(
        "Recommended Purchasing Value",
        f"${df['recommended_order_value'].sum():,.0f}",
    )

    k3.metric(
        "Recommended Units",
        f"{df['recommended_order_qty'].sum():,.0f}",
    )

    k4.metric(
        "Late-to-Need Lines",
        f"{late.sum():,}",
    )

    k5, k6, k7, k8 = st.columns(4)

    k5.metric(
        "Expedite Required",
        f"{df['expedite_required'].sum():,}",
    )

    k6.metric(
        "Late-to-Need Value",
        f"${df.loc[late, 'recommended_order_value'].sum():,.0f}",
    )

    k7.metric(
        "On-Time Feasible Lines",
        f"{df['receipt_feasible_on_time'].sum():,}",
    )

    k8.metric(
        "Suppliers",
        f"{df['supplier_id'].nunique():,}",
    )

    # ========================================================
    # 2. EXECUTION ACTIONS
    # ========================================================

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader(
            "Execution Action Distribution"
        )

        action_summary = (
            df.groupby(
                "execution_action",
                as_index=False,
            )
            .agg(
                order_lines=("sku", "count"),
                order_value=(
                    "recommended_order_value",
                    "sum",
                ),
            )
            .sort_values(
                "order_lines",
                ascending=False,
            )
        )

        fig = px.bar(
            action_summary,
            x="execution_action",
            y="order_lines",
            text="order_lines",
            labels={
                "execution_action": "Execution Action",
                "order_lines": "Order Lines",
            },
        )

        fig.update_layout(
            xaxis_tickangle=-30
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:
        st.subheader(
            "Purchasing Value by Supplier"
        )

        supplier_summary = (
            df.groupby(
                "supplier_id",
                as_index=False,
            )
            .agg(
                order_lines=("sku", "count"),
                order_units=(
                    "recommended_order_qty",
                    "sum",
                ),
                order_value=(
                    "recommended_order_value",
                    "sum",
                ),
            )
            .sort_values(
                "order_value",
                ascending=False,
            )
        )

        fig = px.bar(
            supplier_summary,
            x="supplier_id",
            y="order_value",
            labels={
                "supplier_id": "Supplier",
                "order_value": (
                    "Recommended Purchasing ($)"
                ),
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # ========================================================
    # 3. REQUIRED RECEIPT SCHEDULE
    # ========================================================

    st.divider()
    st.subheader(
        "Required Receipt Schedule"
    )

    receipt_schedule = (
        df.groupby(
            "required_receipt_week",
            as_index=False,
        )
        .agg(
            order_lines=("sku", "count"),
            order_value=(
                "recommended_order_value",
                "sum",
            ),
        )
        .sort_values(
            "required_receipt_week"
        )
    )

    fig = px.bar(
        receipt_schedule,
        x="required_receipt_week",
        y="order_value",
        labels={
            "required_receipt_week": (
                "Required Receipt Week"
            ),
            "order_value": (
                "Purchasing Value ($)"
            ),
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # ========================================================
    # 4. PROCUREMENT WORKLIST
    # ========================================================

    st.divider()
    st.subheader(
        "Procurement Proposal Worklist"
    )

    a, b, c = st.columns(3)

    with a:
        actions_available = sorted(
            df["execution_action"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_actions = st.multiselect(
            "Execution Actions",
            actions_available,
            default=actions_available,
            key="po_action_filter",
        )

    with b:
        priorities_available = sorted(
            df["urgency_priority"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_priorities = st.multiselect(
            "Urgency Priorities",
            priorities_available,
            default=priorities_available,
            key="po_priority_filter",
        )

    with c:
        late_only = st.checkbox(
            "Late-to-Need Only",
            value=False,
            key="po_late_only",
        )

    worklist = df[
        df["execution_action"].isin(
            selected_actions
        )
        & df["urgency_priority"].isin(
            selected_priorities
        )
    ].copy()

    if late_only:
        worklist = worklist[
            worklist["receipt_delay_days"] > 0
        ].copy()

    worklist = worklist.sort_values(
        [
            "receipt_delay_days",
            "recommended_order_value",
        ],
        ascending=[False, False],
    )

    display_columns = [
        "sku",
        "product_name",
        "supplier_id",
        "category",
        "recommended_order_qty",
        "recommended_order_value",
        "executable_release_date",
        "required_receipt_week",
        "earliest_feasible_receipt",
        "receipt_delay_days",
        "execution_action",
        "urgency_priority",
        "expedite_required",
    ]

    st.caption(
        f"{len(worklist):,} procurement proposals "
        "match your filters."
    )

    st.dataframe(
        worklist[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    export_df = worklist.copy()

    for col in date_columns:
        export_df[col] = (
            export_df[col]
            .dt.strftime("%Y-%m-%d")
        )

    st.download_button(
        "Export Filtered Procurement Proposals",
        data=export_df.to_csv(
            index=False
        ),
        file_name=(
            "procurement_order_proposals.csv"
        ),
        mime="text/csv",
        key="po_export",
    )

    # ========================================================
    # 5. SKU ORDER SCHEDULE
    # ========================================================

    st.divider()
    st.subheader("SKU Order Schedule")

    sku_choice = st.selectbox(
        "Select SKU",
        sorted(
            df["sku"].unique().tolist()
        ),
        key="po_sku_selection",
    )

    sku_orders = df[
        df["sku"] == sku_choice
    ].sort_values(
        "required_receipt_week"
    )

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Order Lines",
        f"{len(sku_orders):,}",
    )

    s2.metric(
        "Recommended Units",
        f"{sku_orders['recommended_order_qty'].sum():,.0f}",
    )

    s3.metric(
        "Purchasing Value",
        f"${sku_orders['recommended_order_value'].sum():,.0f}",
    )

    st.dataframe(
        sku_orders[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "These are model-generated purchasing proposals, "
        "not approved or released ERP purchase orders. "
        "Late-to-need estimates assume no confirmed expedite."
    )

    # ========================================================
    # 6. PLANNER REVIEW AND APPROVAL INTEGRATION
    # ========================================================

    show_planner_review(
        selected_suppliers=selected_suppliers,
        selected_categories=selected_categories,
    )
