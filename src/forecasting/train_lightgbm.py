from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb

warnings.filterwarnings("ignore")


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


def fix_columns(df):
    mapping = {}
    for col in df.columns:
        c_low = col.lower()
        if "cluster" in c_low or "label" in c_low:
            mapping[col] = "cluster"
        elif "method" in c_low:
            mapping[col] = "method"
        elif "forecast_strategy" in c_low:
            mapping[col] = "forecast_strategy"
        elif "pattern_type" in c_low:
            mapping[col] = "pattern_type"
        elif "wape" in c_low and "base" in c_low:
            mapping[col] = "wape_base"
        elif "wape" in c_low and "spec" in c_low:
            mapping[col] = "wape_spec"
        elif "wape" in c_low and "lgbm" in c_low:
            mapping[col] = "wape_lgbm"
    return df.rename(columns=mapping)


def build_lgbm_features(sales_df, assign_df, h=28):
    day_cols = [c for c in sales_df.columns if c.startswith("Day_")]
    data_list = []

    for _, row in sales_df.iterrows():
        item_id = row["item_id"]
        ts = row[day_cols].values.astype(float)

        if len(ts) < h + 28:
            continue

        c_match = assign_df[assign_df["item_id"] == item_id]
        cluster_val = c_match["cluster"].values[0] if not c_match.empty else 0

        train_series = ts[:-h]
        future_series = ts[-h:]

        data_list.append(
            {
                "item_id": item_id,
                "cluster": cluster_val,
                "lag_7": train_series[-7],
                "lag_14": train_series[-14],
                "lag_28": train_series[-28],
                "rolling_mean_7": train_series[-7:].mean(),
                "rolling_mean_14": train_series[-14:].mean(),
                "rolling_std_14": train_series[-14:].std(),
                "target": future_series.mean(),
            }
        )

    return pd.DataFrame(data_list)


def main():
    old_summary_path = FORECASTING_RESULTS_DIR / "evaluation_summary_by_cluster.csv"
    sales_path = DATA_DIR / "m5_sales_subset.csv"
    assign_path = CLUSTERING_RESULTS_DIR / "cluster_assignments_k7.csv"

    sales_df = pd.read_csv(sales_path)
    assign_df = fix_columns(pd.read_csv(assign_path))
    old_summary = fix_columns(pd.read_csv(old_summary_path))

    h = 28

    print(">>> Engineering features for global LightGBM forecasting benchmark...")
    df_feat = build_lgbm_features(sales_df, assign_df, h=h)

    if df_feat.empty:
        raise ValueError("Feature dataframe is empty. Please check the input data and horizon setting.")

    feature_cols = [
        "cluster",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_14",
    ]

    x = df_feat[feature_cols]
    y = df_feat["target"]

    print(">>> Training LightGBM regression model...")
    params = {
        "objective": "regression",
        "verbosity": -1,
        "metric": "rmse",
    }

    train_data = lgb.Dataset(x, label=y)
    model = lgb.train(params, train_data)

    df_feat["y_pred_lgbm"] = model.predict(x)

    day_cols = [c for c in sales_df.columns if c.startswith("Day_")]

    print(">>> Evaluating cluster-level LightGBM performance...")
    lgbm_res = []

    for _, row in df_feat.iterrows():
        actual = sales_df.loc[sales_df["item_id"] == row["item_id"], day_cols].values[0][-h:]
        pred = np.full(h, row["y_pred_lgbm"])

        lgbm_res.append(
            {
                "item_id": row["item_id"],
                "cluster": row["cluster"],
                "wape_lgbm": calc_wape(actual, pred),
            }
        )

    lgbm_detail = pd.DataFrame(lgbm_res)
    detail_save_path = FORECASTING_RESULTS_DIR / "lgbm_evaluation_by_sku.csv"
    lgbm_detail.to_csv(detail_save_path, index=False)

    new_summary = lgbm_detail.groupby("cluster")["wape_lgbm"].mean().reset_index()
    final_comparison = pd.merge(old_summary, new_summary, on="cluster", how="inner")

    print("\n" + "=" * 70)
    print("Final Model Comparison Report (Metric: WAPE)")
    print("=" * 70)

    cols_to_show = [
        c
        for c in ["cluster", "pattern_type", "forecast_strategy", "wape_spec", "wape_base", "wape_lgbm"]
        if c in final_comparison.columns
    ]
    print(final_comparison[cols_to_show].to_string(index=False))
    print("=" * 70)

    save_path = FORECASTING_RESULTS_DIR / "final_comparison_with_lgbm.csv"
    final_comparison.to_csv(save_path, index=False)

    print(f"Detailed LightGBM results saved to: {detail_save_path}")
    print(f"Final comparison saved to: {save_path}")


if __name__ == "__main__":
    main()