
from pathlib import Path
import sys
import re

import pandas as pd


# ============================================================
# PHASE 10D - AI PROCUREMENT ASSISTANT VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
AI_DIR = ROOT / "src" / "ai"

sys.path.insert(0, str(AI_DIR))

from procurement_assistant import ProcurementAssistant


class AssistantValidator:

    def __init__(self):
        self.assistant = ProcurementAssistant()
        self.data = self.assistant.data

        self.passed = 0
        self.failed = 0
        self.results = []

    def check(self, name, condition, details=""):
        status = "PASS" if condition else "FAIL"

        if condition:
            self.passed += 1
        else:
            self.failed += 1

        self.results.append({
            "test": name,
            "status": status,
            "details": details,
        })

        print(f"[{status}] {name}")

        if details and not condition:
            print(f"       {details}")

    @staticmethod
    def has_amount(answer, amount):
        expected = f"${amount:,.2f}"
        return expected in answer

    @staticmethod
    def has_integer(answer, value):
        pattern = rf"(?<![\d,]){re.escape(f'{int(value):,}')}(?![\d,])"
        return bool(re.search(pattern, answer))

    def test_purchasing_summary(self):
        actions = self.data.get_dataset("supply_actions")
        answer = self.assistant.ask(
            "Give me the purchasing summary"
        )

        self.check(
            "Purchasing total matches source data",
            self.has_amount(
                answer,
                actions["recommended_order_value"].sum(),
            ),
        )

        self.check(
            "Purchasing quantity matches source data",
            self.has_integer(
                answer,
                actions["recommended_order_qty"].sum(),
            ),
        )

        self.check(
            "Purchasing line count matches source data",
            f"Recommended order lines: {len(actions):,}"
            in answer,
        )

        self.check(
            "Purchasing SKU count matches source data",
            f"SKUs with proposals: "
            f"{actions['sku'].nunique():,}" in answer,
        )

        self.check(
            "Purchasing recommendations not labeled approved",
            "not approved purchase orders" in answer.lower(),
        )

    def test_urgent_orders(self):
        actions = self.data.get_dataset("supply_actions")

        urgent = actions[
            actions["expedite_required"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        ]

        answer = self.assistant.ask(
            "Which orders need to be expedited?"
        )

        self.check(
            "Expedite line count matches",
            f"Expedite-required lines: {len(urgent):,}"
            in answer,
        )

        self.check(
            "Expedite SKU count matches",
            f"Affected SKUs: {urgent['sku'].nunique():,}"
            in answer,
        )

        self.check(
            "Expedite purchasing value matches",
            self.has_amount(
                answer,
                urgent["recommended_order_value"].sum(),
            ),
        )

        self.check(
            "Supplier expedite commitment not assumed",
            "does not confirm" in answer.lower(),
        )

    def test_stockout_risks(self):
        projection = self.data.get_dataset(
            "inventory_projection"
        )

        stockouts = projection[
            projection["stockout"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        ]

        answer = self.assistant.ask(
            "Which products are at risk of stockouts?"
        )

        self.check(
            "Stockout SKU count matches",
            f"Affected SKUs: {stockouts['sku'].nunique():,}"
            in answer,
        )

        self.check(
            "Stockout SKU-week count matches",
            f"SKU-week stockout records: {len(stockouts):,}"
            in answer,
        )

    def test_sku_explanation(self):
        sku = "SKU-113"

        policy = self.data.get_dataset(
            "inventory_policy"
        )

        master = self.data.get_dataset(
            "planning_master"
        )

        p = policy.loc[policy["sku"] == sku].iloc[0]
        m = master.loc[master["sku"] == sku].iloc[0]

        answer = self.assistant.ask(
            "Why are we ordering SKU-113?"
        )

        self.check(
            "SKU identified correctly",
            sku in answer,
        )

        self.check(
            "SKU inventory position matches",
            f"Inventory position: "
            f"{m['inventory_position']:,.0f}" in answer,
        )

        self.check(
            "SKU safety stock matches",
            f"Safety stock: "
            f"{p['safety_stock']:,.0f}" in answer,
        )

        self.check(
            "SKU reorder point matches",
            f"Reorder point: "
            f"{p['reorder_point']:,.0f}" in answer,
        )

        self.check(
            "SKU demand rate matches",
            f"{p['planning_weekly_demand']:,.2f}"
            in answer,
        )

        self.check(
            "Missing forecast accuracy flagged",
            "forecast caution" in answer.lower(),
        )

        self.check(
            "Immediate and future proposals distinguished",
            "do not add them together" in answer.lower(),
        )

    def test_supplier(self):
        supplier = "SUP-001"

        actions = self.data.get_dataset(
            "supply_actions"
        )

        supplier_orders = actions[
            actions["supplier_id"] == supplier
        ]

        answer = self.assistant.ask(
            "Analyze supplier SUP-001"
        )

        self.check(
            "Supplier identified correctly",
            supplier in answer,
        )

        self.check(
            "Supplier purchasing value matches",
            self.has_amount(
                answer,
                supplier_orders[
                    "recommended_order_value"
                ].sum(),
            ),
        )

        self.check(
            "Supplier purchasing lines match",
            f"Purchasing lines: {len(supplier_orders):,}"
            in answer,
        )

    def test_edge_cases(self):
        unknown = self.assistant.ask(
            "Why are we ordering SKU-999?"
        )

        self.check(
            "Unknown SKU handled safely",
            "not found" in unknown.lower(),
        )

        blank = self.assistant.ask("")

        self.check(
            "Empty question handled safely",
            "please enter" in blank.lower(),
        )

        unsupported = self.assistant.ask(
            "What is the weather tomorrow?"
        )

        self.check(
            "Unsupported question handled safely",
            "currently answer" in unsupported.lower(),
        )

    def test_purchasing_window(self):
        actions = self.data.get_dataset(
            "supply_actions"
        ).copy()

        master = self.data.get_dataset(
            "planning_master"
        )

        snapshot = pd.to_datetime(
            master["snapshot_date"],
            errors="coerce",
        ).dropna().max().normalize()

        actions["executable_release_date"] = (
            pd.to_datetime(
                actions["executable_release_date"],
                errors="coerce",
            )
        )

        selected = actions[
            (
                actions["executable_release_date"]
                >= snapshot
            )
            & (
                actions["executable_release_date"]
                <= snapshot + pd.Timedelta(days=30)
            )
        ]

        answer = self.assistant.ask(
            "How much purchasing is planned "
            "for the next 30 days?"
        )

        self.check(
            "30-day purchasing value matches",
            self.has_amount(
                answer,
                selected["recommended_order_value"].sum(),
            ),
        )

        self.check(
            "30-day purchasing line count matches",
            f"Proposed order lines: {len(selected):,}"
            in answer,
        )

        self.check(
            "30-day window references planning snapshot",
            str(snapshot.date()) in answer,
        )

    def run(self):
        print("=" * 70)
        print("PHASE 10D - AI PROCUREMENT ACCURACY VALIDATION")
        print("=" * 70)

        self.test_purchasing_summary()
        self.test_urgent_orders()
        self.test_stockout_risks()
        self.test_sku_explanation()
        self.test_supplier()
        self.test_edge_cases()
        self.test_purchasing_window()

        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        print(f"PASSED: {self.passed}")
        print(f"FAILED: {self.failed}")
        print(f"TOTAL:  {self.passed + self.failed}")

        output = (
            ROOT
            / "data"
            / "processed"
            / "ai_assistant_validation_results.csv"
        )

        pd.DataFrame(self.results).to_csv(
            output,
            index=False,
        )

        print(f"\nValidation report saved: {output}")

        if self.failed:
            print("\nVALIDATION FAILED - REVIEW REQUIRED")
            sys.exit(1)

        print("\nAI ASSISTANT ACCURACY VALIDATION PASSED")


if __name__ == "__main__":
    AssistantValidator().run()
