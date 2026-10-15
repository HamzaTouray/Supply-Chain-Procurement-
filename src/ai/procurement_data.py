
from pathlib import Path
import pandas as pd


# ============================================================
# PHASE 10A - AI PROCUREMENT DATA FOUNDATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"


DATASETS = {
    "planning_master": "planning_master.csv",
    "inventory_policy": "inventory_policy.csv",
    "order_timing": "order_timing_recommendations.csv",
    "supply_actions": "feasible_supply_actions.csv",
    "inventory_projection": "feasible_inventory_projection.csv",
    "supplier_intelligence": "supplier_intelligence.csv",
    "forecasts": "production_forecasts.csv",
}


class ProcurementData:
    def __init__(self):
        self.datasets = {}
        self.load_data()

    def load_data(self):
        for name, filename in DATASETS.items():
            path = DATA_DIR / filename

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing required dataset: {path}"
                )

            self.datasets[name] = pd.read_csv(path)

        self.validate_data()

    def validate_data(self):
        master = self.datasets["planning_master"]

        if master["sku"].duplicated().any():
            raise ValueError(
                "Planning Master contains duplicate SKUs."
            )

        master_skus = set(master["sku"])

        for name in [
            "inventory_policy",
            "order_timing",
            "inventory_projection",
        ]:
            df = self.datasets[name]

            if set(df["sku"]) != master_skus:
                raise ValueError(
                    f"SKU coverage mismatch: {name}"
                )

        actions = self.datasets["supply_actions"]

        if not set(actions["sku"]).issubset(master_skus):
            raise ValueError(
                "Supply actions contain unknown SKUs."
            )

    def get_dataset(self, name):
        if name not in self.datasets:
            raise KeyError(
                f"Unknown dataset: {name}"
            )

        return self.datasets[name].copy()

    def get_sku_profile(self, sku):
        sku = str(sku).strip().upper()

        profile = {}

        for name in [
            "planning_master",
            "inventory_policy",
            "order_timing",
            "supply_actions",
            "inventory_projection",
        ]:
            df = self.datasets[name]

            matches = df[
                df["sku"].astype(str).str.upper() == sku
            ]

            profile[name] = matches.copy()

        return profile

    def get_urgent_orders(self):
        actions = self.get_dataset("supply_actions")

        urgent = actions[
            actions["expedite_required"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        ]

        return urgent.sort_values(
            "recommended_order_value",
            ascending=False,
        )

    def get_stockout_risks(self):
        projection = self.get_dataset(
            "inventory_projection"
        )

        risks = projection[
            projection["stockout"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        ]

        return risks.sort_values(
            ["forecast_week", "sku"]
        )

    def get_supplier_orders(self, supplier_id):
        actions = self.get_dataset("supply_actions")

        return actions[
            actions["supplier_id"].astype(str).str.upper()
            == str(supplier_id).strip().upper()
        ].copy()

    def get_purchasing_summary(self):
        actions = self.get_dataset("supply_actions")

        return {
            "order_lines": len(actions),
            "unique_skus": actions["sku"].nunique(),
            "suppliers": actions["supplier_id"].nunique(),
            "total_units": int(
                actions["recommended_order_qty"].sum()
            ),
            "total_value": float(
                actions["recommended_order_value"].sum()
            ),
        }

    def get_dataset_summary(self):
        return {
            name: {
                "rows": len(df),
                "columns": len(df.columns),
            }
            for name, df in self.datasets.items()
        }


if __name__ == "__main__":
    print("=" * 65)
    print("PHASE 10A - AI PROCUREMENT DATA FOUNDATION")
    print("=" * 65)

    data = ProcurementData()

    print("\nDATASETS LOADED")
    print("-" * 65)

    for name, info in data.get_dataset_summary().items():
        print(
            f"{name:<25} "
            f"Rows: {info['rows']:<7} "
            f"Columns: {info['columns']}"
        )

    print("\nPURCHASING SUMMARY")
    print("-" * 65)

    summary = data.get_purchasing_summary()

    for key, value in summary.items():
        if key == "total_value":
            print(f"{key}: ${value:,.2f}")
        else:
            print(f"{key}: {value:,}")

    print("\nURGENT PURCHASE ORDERS")
    print("-" * 65)

    urgent = data.get_urgent_orders()

    print("Expedite-required lines:", len(urgent))

    print("\nPROJECTED STOCKOUT RISK")
    print("-" * 65)

    risks = data.get_stockout_risks()

    print("Stockout weeks:", len(risks))
    print("Affected SKUs:", risks["sku"].nunique())

    print("\n" + "=" * 65)
    print("AI DATA FOUNDATION VALIDATION PASSED")
    print("=" * 65)
