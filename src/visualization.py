from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_sales_data(sales_csv_path) -> pd.DataFrame:
    sales_csv_path = Path(sales_csv_path)
    if not sales_csv_path.exists():
        raise FileNotFoundError(f"Sales file not found: {sales_csv_path}")

    sales_df = pd.read_csv(sales_csv_path)
    sales_df["item_id"] = sales_df["item_id"].astype(str).str.strip()
    return sales_df


def load_cluster_assignments(assign_csv_path) -> pd.DataFrame:
    assign_csv_path = Path(assign_csv_path)
    if not assign_csv_path.exists():
        raise FileNotFoundError(f"Cluster assignment file not found: {assign_csv_path}")

    assign_df = pd.read_csv(assign_csv_path)
    assign_df["item_id"] = assign_df["item_id"].astype(str).str.strip()
    return assign_df


def get_day_columns(sales_df: pd.DataFrame):
    day_cols = [c for c in sales_df.columns if c.startswith("Day_")]
    return sorted(day_cols, key=lambda x: int(x.split("_")[1]))


def plot_cluster_mean_patterns(
    sales_df: pd.DataFrame,
    assign_df: pd.DataFrame,
    save_path
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    merged_df = assign_df.merge(sales_df, on="item_id", how="inner")
    day_cols = get_day_columns(sales_df)
    clusters = sorted(merged_df["cluster"].unique())

    plt.figure(figsize=(12, 7))

    for cluster_id in clusters:
        cluster_data = merged_df[merged_df["cluster"] == cluster_id]
        mean_pattern = cluster_data[day_cols].mean(axis=0).values
        plt.plot(mean_pattern, label=f"Cluster {cluster_id}")

    plt.title("Mean Sales Patterns by Cluster")
    plt.xlabel("Day")
    plt.ylabel("Average Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Cluster pattern plot saved to: {save_path}")