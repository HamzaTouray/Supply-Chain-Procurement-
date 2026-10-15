
import re
from datetime import timedelta

import pandas as pd

try:
    from .procurement_data import ProcurementData
except ImportError:
    from procurement_data import ProcurementData


class ProcurementAssistant:
    def __init__(self):
        self.data = ProcurementData()

    @staticmethod
    def money(value):
        return f"${float(value):,.2f}"

    @staticmethod
    def number(value):
        return f"{float(value):,.0f}"

    @staticmethod
    def is_true(series):
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    @staticmethod
    def table(df, columns, limit=10):
        if df.empty:
            return "No matching records found."

        available = [c for c in columns if c in df.columns]

        return df[available].head(limit).to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )

    def ask(self, question):
        question = str(question).strip()

        if not question:
            return "Please enter a procurement question."

        q = question.lower()

        sku_match = re.search(
            r"\bsku[-\s]?(\d{1,6})\b",
            question,
            re.IGNORECASE,
        )

        if sku_match:
            sku = f"SKU-{int(sku_match.group(1)):03d}"
            return self.explain_sku(sku)

        supplier_match = re.search(
            r"\bsup[-\s]?(\d{1,6})\b",
            question,
            re.IGNORECASE,
        )

        if supplier_match:
            supplier_id = (
                f"SUP-{int(supplier_match.group(1)):03d}"
            )
            return self.explain_supplier(supplier_id)

        if any(
            phrase in q
            for phrase in [
                "next 30 days",
                "next 60 days",
                "next 90 days",
                "this month",
            ]
        ):
            if "60" in q:
                days = 60
            elif "90" in q:
                days = 90
            else:
                days = 30

            return self.purchasing_window(days)

        if any(
            phrase in q
            for phrase in [
                "stockout",
                "stock out",
                "out of stock",
                "inventory risk",
            ]
        ):
            return self.stockout_risks()

        if any(
            phrase in q
            for phrase in [
                "urgent",
                "immediately",
                "expedite",
                "priority orders",
            ]
        ):
            return self.urgent_orders()

        if any(
            phrase in q
            for phrase in [
                "supplier risk",
                "risky suppliers",
                "supplier performance",
            ]
        ):
            return self.supplier_risks()

        if any(
            phrase in q
            for phrase in [
                "purchasing summary",
                "total purchasing",
                "total order value",
                "procurement summary",
            ]
        ):
            return self.purchasing_summary()

        return (
            "I can currently answer questions about:\n\n"
            "- Urgent purchasing recommendations\n"
            "- Projected stockout risks\n"
            "- Supplier performance and purchasing exposure\n"
            "- Purchasing totals and release windows\n"
            "- Individual SKU explanations\n\n"
            "Try: 'Why are we ordering SKU-113?'"
        )

    def purchasing_summary(self):
        summary = self.data.get_purchasing_summary()

        return (
            "PURCHASING SUMMARY\n\n"
            f"Recommended order lines: "
            f"{summary['order_lines']:,}\n"
            f"SKUs with proposals: "
            f"{summary['unique_skus']:,}\n"
            f"Suppliers: {summary['suppliers']:,}\n"
            f"Recommended units: "
            f"{summary['total_units']:,}\n"
            f"Total proposed value: "
            f"{self.money(summary['total_value'])}\n\n"
            "These are planning recommendations, "
            "not approved purchase orders."
        )

    def urgent_orders(self):
        df = self.data.get_urgent_orders()

        if df.empty:
            return "No expedite-required proposals were found."

        value = df["recommended_order_value"].sum()
        units = df["recommended_order_qty"].sum()

        top = df.sort_values(
            "recommended_order_value",
            ascending=False,
        )

        return (
            "URGENT PROCUREMENT RECOMMENDATIONS\n\n"
            f"Expedite-required lines: {len(df):,}\n"
            f"Affected SKUs: {df['sku'].nunique():,}\n"
            f"Units: {self.number(units)}\n"
            f"Proposed value: {self.money(value)}\n\n"
            "Highest-value expedite-required proposals:\n\n"
            + self.table(
                top,
                [
                    "sku",
                    "supplier_id",
                    "recommended_order_qty",
                    "recommended_order_value",
                    "receipt_delay_days",
                    "execution_action",
                ],
            )
            + "\n\nExpedite-required means the planning "
              "engine has identified a delivery timing "
              "problem. It does not confirm that a supplier "
              "has agreed to expedite."
        )

    def stockout_risks(self):
        df = self.data.get_stockout_risks()

        if df.empty:
            return "No projected stockout weeks were found."

        first_risk = (
            df.groupby("sku", as_index=False)
            .agg(
                first_stockout_week=(
                    "forecast_week", "min"
                ),
                stockout_weeks=("forecast_week", "size"),
                lowest_ending_inventory=(
                    "ending_inventory", "min"
                ),
            )
            .sort_values(
                ["first_stockout_week", "sku"]
            )
        )

        return (
            "PROJECTED STOCKOUT RISK\n\n"
            f"Affected SKUs: {df['sku'].nunique():,}\n"
            f"SKU-week stockout records: {len(df):,}\n\n"
            "Earliest projected stockout risks:\n\n"
            + self.table(
                first_risk,
                [
                    "sku",
                    "first_stockout_week",
                    "stockout_weeks",
                    "lowest_ending_inventory",
                ],
            )
            + "\n\nThese risks come from the feasible "
              "26-week inventory projection, including "
              "planned receipts and forecast demand."
        )

    def supplier_risks(self):
        actions = self.data.get_dataset(
            "supply_actions"
        )

        summary = (
            actions.groupby(
                "supplier_id", as_index=False
            )
            .agg(
                order_lines=("sku", "size"),
                proposed_value=(
                    "recommended_order_value", "sum"
                ),
                affected_skus=("sku", "nunique"),
            )
            .sort_values(
                "proposed_value",
                ascending=False,
            )
        )

        return (
            "SUPPLIER PURCHASING EXPOSURE\n\n"
            "Suppliers ranked by proposed purchasing value "
            "(not by supplier risk score):\n\n"
            + self.table(
                summary,
                [
                    "supplier_id",
                    "order_lines",
                    "affected_skus",
                    "proposed_value",
                ],
            )
            + "\n\nHigh purchasing exposure does not "
              "necessarily mean high supplier risk. "
              "Supplier reliability should also be reviewed."
        )

    def explain_supplier(self, supplier_id):
        orders = self.data.get_supplier_orders(
            supplier_id
        )

        if orders.empty:
            return (
                f"No purchasing proposals found for "
                f"{supplier_id}."
            )

        expedited = self.is_true(
            orders["expedite_required"]
        )

        return (
            f"SUPPLIER ANALYSIS: {supplier_id}\n\n"
            f"Purchasing lines: {len(orders):,}\n"
            f"Unique SKUs: {orders['sku'].nunique():,}\n"
            f"Proposed units: "
            f"{self.number(orders['recommended_order_qty'].sum())}\n"
            f"Proposed value: "
            f"{self.money(orders['recommended_order_value'].sum())}\n"
            f"Expedite-required lines: "
            f"{expedited.sum():,}\n\n"
            "This is purchasing exposure from the "
            "current planning recommendations."
        )

    def purchasing_window(self, days):
        actions = self.data.get_dataset(
            "supply_actions"
        )

        master = self.data.get_dataset(
            "planning_master"
        )

        snapshot = pd.to_datetime(
            master["snapshot_date"],
            errors="coerce",
        ).dropna()

        if snapshot.empty:
            return "No valid planning snapshot date found."

        start = snapshot.max().normalize()
        end = start + timedelta(days=days)

        actions["executable_release_date"] = (
            pd.to_datetime(
                actions["executable_release_date"],
                errors="coerce",
            )
        )

        selected = actions[
            (
                actions["executable_release_date"]
                >= start
            )
            & (
                actions["executable_release_date"]
                <= end
            )
        ].copy()

        return (
            f"PURCHASING PLAN: NEXT {days} DAYS\n\n"
            f"Planning snapshot: {start.date()}\n"
            f"Through: {end.date()}\n"
            f"Proposed order lines: {len(selected):,}\n"
            f"Proposed units: "
            f"{self.number(selected['recommended_order_qty'].sum())}\n"
            f"Proposed purchasing value: "
            f"{self.money(selected['recommended_order_value'].sum())}\n\n"
            "This window is measured from the planning "
            "snapshot, not today's live date. "
            "Amounts are proposed, not committed."
        )

    def explain_sku(self, sku):
        profile = self.data.get_sku_profile(sku)

        master = profile["planning_master"]
        policy = profile["inventory_policy"]
        timing = profile["order_timing"]
        actions = profile["supply_actions"]

        if master.empty:
            return f"SKU {sku} was not found."

        m = master.iloc[0]
        p = policy.iloc[0]
        t = timing.iloc[0]

        proposed_units = (
            actions["recommended_order_qty"].sum()
            if not actions.empty else 0
        )

        proposed_value = (
            actions["recommended_order_value"].sum()
            if not actions.empty else 0
        )

        explanation = (
            f"SKU ANALYSIS: {sku}\n\n"
            f"Product: {m['product_name']}\n"
            f"Supplier: {m['supplier_id']}\n"
            f"Demand class: {p['final_demand_class']}\n\n"
            "INVENTORY\n"
            f"Net available: {self.number(m['net_available'])}\n"
            f"Open PO units: "
            f"{self.number(m['total_open_po_qty'])}\n"
            f"Inventory position: "
            f"{self.number(m['inventory_position'])}\n"
            f"Safety stock: {self.number(p['safety_stock'])}\n"
            f"Reorder point: "
            f"{self.number(p['reorder_point'])}\n\n"
            "DEMAND AND LEAD TIME\n"
            f"Planning weekly demand: "
            f"{float(p['planning_weekly_demand']):,.2f}\n"
            f"Planning lead time: "
            f"{float(p['planning_lead_time_weeks']):,.2f} "
            "weeks\n\n"
            "PURCHASING RECOMMENDATIONS\n"
            f"Immediate timing recommendation: "
            f"{self.number(t['recommended_order_qty'])} "
            "units\n"
            f"Future feasible proposal lines: "
            f"{len(actions):,}\n"
            f"Future proposed units: "
            f"{self.number(proposed_units)}\n"
            f"Future proposed value: "
            f"{self.money(proposed_value)}\n"
            f"Timing action: {t['order_timing_action']}\n"
        )

        if pd.isna(p["selected_wape"]):
            explanation += (
                "\nFORECAST CAUTION\n"
                "Selected forecast WAPE is unavailable. "
                "Review forecast readiness before approving "
                "purchase quantities.\n"
            )

        explanation += (
            "\nThe immediate timing recommendation and "
            "the future feasible proposals are different "
            "planning views. Do not add them together."
        )

        return explanation


if __name__ == "__main__":
    assistant = ProcurementAssistant()

    questions = [
        "Give me the purchasing summary",
        "Which orders need to be expedited?",
        "Which products are at risk of stockouts?",
        "Why are we ordering SKU-113?",
        "Analyze supplier SUP-001",
        "How much purchasing is planned for the next 30 days?",
    ]

    print("=" * 70)
    print("PHASE 10B - PROCUREMENT ASSISTANT TEST")
    print("=" * 70)

    for question in questions:
        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("-" * 70)
        print(assistant.ask(question))
