from pathlib import Path
import pandas as pd
import numpy as np


def calculate_safety_stock(rmse, z_score=1.65, lead_time=7):
    return z_score * rmse * np.sqrt(lead_time)


def main():
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "results" / "forecasting" / "evaluation_metrics_by_sku.csv"

    if not path.exists():
        print(f"Error: File not found at {path}. Please check the directory.")
        return

    df = pd.read_csv(path)

    unit_cost = 15.0

    df["ss_spec"] = calculate_safety_stock(df["rmse_spec"])
    df["ss_base"] = calculate_safety_stock(df["rmse_base"])
    df["saving"] = (df["ss_base"] - df["ss_spec"]) * unit_cost

    total_money_saved = df["saving"].sum()

    print("\n" + "=" * 60)
    print("      PROJECT IMPACT REPORT: INVENTORY OPTIMIZATION")
    print("=" * 60)
    print(f"Total SKUs Evaluated          : {len(df)}")
    print(f"Total Inventory Cost Savings  : ${total_money_saved:,.2f}")
    print("=" * 60)
    print("Optimization Insight: Accuracy improvements directly reduce")
    print("safety stock requirements and associated holding costs.")
    print("=" * 60)


if __name__ == "__main__":
    main()