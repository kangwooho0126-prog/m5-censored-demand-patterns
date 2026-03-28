from pathlib import Path
import pandas as pd
import numpy as np


def get_day_columns(df: pd.DataFrame):
    return sorted(
        [c for c in df.columns if c.startswith("Day_")],
        key=lambda x: int(x.split("_")[1])
    )


def compute_cluster_statistics(sales_df: pd.DataFrame, assign_df: pd.DataFrame) -> pd.DataFrame:
    merged = assign_df.merge(sales_df, on="item_id", how="inner")
    day_cols = get_day_columns(sales_df)

    results = []

    for cluster_id in sorted(merged["cluster"].unique()):
        cluster_data = merged[merged["cluster"] == cluster_id]
        x = cluster_data[day_cols].values.astype(float)

        mean_sales = np.mean(x)
        std_sales = np.std(x)
        zero_ratio = np.mean(x == 0)
        cv = std_sales / (mean_sales + 1e-6)
        burst_ratio = np.mean(x > 2 * mean_sales)

        results.append(
            {
                "cluster": cluster_id,
                "mean_sales": round(mean_sales, 4),
                "std_sales": round(std_sales, 4),
                "cv": round(cv, 4),
                "zero_ratio": round(zero_ratio, 4),
                "burst_ratio": round(burst_ratio, 4),
            }
        )

    return pd.DataFrame(results)


def classify_pattern(row: pd.Series) -> str:
    if row["zero_ratio"] > 0.6:
        return "intermittent"
    if row["burst_ratio"] > 0.1:
        return "burst"
    if row["cv"] < 0.5:
        return "smooth"
    return "volatile"


def assign_pattern_labels(stats_df: pd.DataFrame) -> pd.DataFrame:
    stats_df = stats_df.copy()
    stats_df["pattern_type"] = stats_df.apply(classify_pattern, axis=1)
    return stats_df


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    clustering_dir = project_root / "results" / "clustering"

    sales_path = data_dir / "m5_sales_subset.csv"
    assign_path = clustering_dir / "cluster_assignments_k7.csv"
    output_path = clustering_dir / "cluster_pattern_summary_k7.csv"

    print("Building cluster pattern summary...")

    sales_df = pd.read_csv(sales_path)
    assign_df = pd.read_csv(assign_path)

    stats_df = compute_cluster_statistics(sales_df, assign_df)
    pattern_df = assign_pattern_labels(stats_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pattern_df.to_csv(output_path, index=False)

    print("Pattern summary generated successfully.")
    print(f"Saved to: {output_path}")
    print(pattern_df)


if __name__ == "__main__":
    main()