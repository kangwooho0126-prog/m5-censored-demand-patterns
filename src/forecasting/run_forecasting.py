from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CLUSTERING_RESULTS_DIR = RESULTS_DIR / "clustering"
FORECASTING_RESULTS_DIR = RESULTS_DIR / "forecasting"

FORECASTING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def calc_wape(y_true, y_pred):
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0:
        return 0.0 if np.sum(np.abs(y_pred)) == 0 else 1.0
    return np.sum(np.abs(y_true - y_pred)) / sum_actual


def get_baseline_forecast(y_train, h=28):
    sma_val = y_train[-28:].mean() if len(y_train) >= 28 else y_train.mean()
    return np.full(h, sma_val)


def get_pattern_aware_forecast(y_pred_base, pattern, h=28):
    if pattern == "intermittent":
        return np.zeros(h), "zero_rule"
    if pattern == "burst":
        return y_pred_base * 0.85, "downward_adjustment"
    if pattern == "smooth":
        return y_pred_base * 1.05, "upward_adjustment"
    if pattern == "volatile":
        return y_pred_base * 1.10, "strong_upward_adjustment"
    return y_pred_base, "no_adjustment"


def main():
    sales_path = DATA_DIR / "m5_sales_subset.csv"
    assign_path = CLUSTERING_RESULTS_DIR / "cluster_assignments_k7.csv"
    pattern_path = CLUSTERING_RESULTS_DIR / "cluster_pattern_summary_k7.csv"

    sales_df = pd.read_csv(sales_path)
    assign_df = pd.read_csv(assign_path)
    pattern_df = pd.read_csv(pattern_path)

    merged_df = pd.merge(sales_df, assign_df, on="item_id", how="inner")
    merged_df = pd.merge(
        merged_df,
        pattern_df[["cluster", "pattern_type"]],
        on="cluster",
        how="left"
    )

    h = 28
    day_cols = [col for col in merged_df.columns if col.startswith("Day_")]
    train_end_idx = len(day_cols) - h

    all_sku_evaluations = []

    print(f"INFO: Processing {len(merged_df)} SKUs using pattern-aware forecasting strategy...")

    for _, sku_row in merged_df.iterrows():
        ts = sku_row[day_cols].values.astype(float)
        y_train = ts[:train_end_idx]
        y_true = ts[train_end_idx:]
        cluster = sku_row["cluster"]
        pattern = sku_row["pattern_type"]

        y_pred_base = get_baseline_forecast(y_train, h=h)

        y_pred_spec, forecast_strategy = get_pattern_aware_forecast(
            y_pred_base=y_pred_base,
            pattern=pattern,
            h=h
        )

        all_sku_evaluations.append(
            {
                "item_id": sku_row["item_id"],
                "cluster": cluster,
                "pattern_type": pattern,
                "forecast_strategy": forecast_strategy,
                "wape_spec": calc_wape(y_true, y_pred_spec),
                "wape_base": calc_wape(y_true, y_pred_base),
                "rmse_spec": np.sqrt(((y_true - y_pred_spec) ** 2).mean()),
                "rmse_base": np.sqrt(((y_true - y_pred_base) ** 2).mean()),
            }
        )

    df_detail = pd.DataFrame(all_sku_evaluations)
    detail_path = FORECASTING_RESULTS_DIR / "evaluation_metrics_by_sku.csv"
    df_detail.to_csv(detail_path, index=False)

    df_summary = (
        df_detail.groupby(["cluster", "pattern_type", "forecast_strategy"])
        .agg(
            {
                "wape_spec": "mean",
                "wape_base": "mean",
                "rmse_spec": "mean",
                "rmse_base": "mean",
            }
        )
        .reset_index()
    )

    summary_path = FORECASTING_RESULTS_DIR / "evaluation_summary_by_cluster.csv"
    df_summary.to_csv(summary_path, index=False)

    print("-" * 40)
    print("Execution Status: SUCCESS")
    print(f"Detailed results saved to: {detail_path}")
    print(f"Summary results saved to: {summary_path}")
    print("-" * 40)


if __name__ == "__main__":
    main()