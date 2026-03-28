from pathlib import Path
import sys
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_fusion import (
    load_static_features,
    load_dynamic_features,
    fuse_static_dynamic_features,
)

from src.clustering import run_kmeans_multiple_k

from src.visualization import (
    load_sales_data,
    load_cluster_assignments,
    plot_cluster_mean_patterns,
)

from src.pattern_analysis import (
    compute_cluster_statistics,
    assign_pattern_labels,
    get_day_columns,
)

from src.inventory_decision import inventory_decision
from src.optimization.pattern_aware_inventory_optimization import optimize_inventory_policy


DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CLUSTERING_RESULTS_DIR = RESULTS_DIR / "clustering"
FORECASTING_RESULTS_DIR = RESULTS_DIR / "forecasting"
DECISION_RESULTS_DIR = RESULTS_DIR / "decision"


def ensure_directories() -> None:
    CLUSTERING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FORECASTING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DECISION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_directories()

    static_csv_path = DATA_DIR / "processed" / "static_features_12d.csv"
    dynamic_csv_path = DATA_DIR / "dynamic_features_16d.csv"
    sales_csv_path = DATA_DIR / "m5_sales_subset.csv"

    print("Step 1: Loading feature files...")
    static_df = load_static_features(str(static_csv_path))
    dynamic_df = load_dynamic_features(str(dynamic_csv_path))

    print("Step 2: Fusing static and dynamic features...")
    fused_df = fuse_static_dynamic_features(static_df, dynamic_df)

    print("Step 3: Running multi-K clustering...")
    metrics_df, best_results = run_kmeans_multiple_k(
        fused_df,
        k_range=range(3, 9),
        n_runs=20,
    )

    metrics_output_path = CLUSTERING_RESULTS_DIR / "fused_clustering_metrics.csv"
    metrics_df.to_csv(metrics_output_path, index=False)
    print(f"Saved: {metrics_output_path}")

    for k, result in best_results.items():
        save_path = CLUSTERING_RESULTS_DIR / f"cluster_assignments_k{k}.csv"
        result["assignment_df"].to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    selected_k = 7
    assign_csv_path = CLUSTERING_RESULTS_DIR / f"cluster_assignments_k{selected_k}.csv"

    print(f"Step 4: Visualizing cluster mean patterns for K={selected_k}...")
    sales_df = load_sales_data(str(sales_csv_path))
    assign_df = load_cluster_assignments(str(assign_csv_path))

    pattern_plot_path = CLUSTERING_RESULTS_DIR / f"cluster_patterns_k{selected_k}.png"
    plot_cluster_mean_patterns(
        sales_df,
        assign_df,
        save_path=str(pattern_plot_path),
    )

    print(f"Step 5: Computing cluster-level pattern summary for K={selected_k}...")
    stats_df = compute_cluster_statistics(sales_df, assign_df)
    stats_df = assign_pattern_labels(stats_df)

    pattern_summary_path = CLUSTERING_RESULTS_DIR / f"cluster_pattern_summary_k{selected_k}.csv"
    stats_df.to_csv(pattern_summary_path, index=False)
    print(f"Saved: {pattern_summary_path}")

    pattern_col = "pattern_type"

    merged_df = assign_df.merge(
        stats_df[["cluster", pattern_col]],
        on="cluster",
        how="left",
    )

    print("Step 6: Generating SKU-level inventory decisions...")
    day_cols = get_day_columns(sales_df)

    sku_stats = sales_df[["item_id"] + day_cols].copy()
    sku_stats["mean_demand"] = sku_stats[day_cols].mean(axis=1)
    sku_stats["std"] = sku_stats[day_cols].std(axis=1)
    sku_stats["lead_time"] = 7

    merged_df = merged_df.merge(
        sku_stats[["item_id", "mean_demand", "std", "lead_time"]],
        on="item_id",
        how="left",
    )

    inventory_results = []

    for _, row in merged_df.iterrows():
        pattern = row[pattern_col]
        mean_demand = row["mean_demand"]
        std = row["std"]
        lead_time = row["lead_time"]

        decision = inventory_decision(
            mean_demand=mean_demand,
            std=std,
            lead_time=lead_time,
            pattern=pattern,
        )

        opt_result = optimize_inventory_policy(
            mean_demand=mean_demand,
            std=std,
            lead_time=lead_time,
            pattern=pattern,
        )

        inventory_results.append(
            {
                "item_id": row["item_id"],
                "cluster": row["cluster"],
                "pattern": pattern,
                "mean_demand": float(mean_demand),
                "std": float(std),
                "lead_time": float(lead_time),
                "safety_stock": decision["safety_stock"],
                "reorder_point": decision["reorder_point"],
                "policy_description": decision["policy_description"],
                "pattern_policy_type": opt_result["pattern_policy_type"],
                "recommended_service_level": opt_result["recommended_service_level"],
                "holding_cost_weight": opt_result["holding_cost_weight"],
                "shortage_cost_weight": opt_result["shortage_cost_weight"],
                "safety_stock_opt": opt_result["safety_stock_opt"],
                "reorder_point_opt": opt_result["reorder_point_opt"],
                "order_up_to_level_opt": opt_result["order_up_to_level_opt"],
                "optimal_order_qty": opt_result["optimal_order_qty"],
                "estimated_total_cost": opt_result["estimated_total_cost"],
            }
        )

    inventory_df = pd.DataFrame(inventory_results)
    inventory_output_path = DECISION_RESULTS_DIR / f"inventory_decision_k{selected_k}.csv"
    inventory_df.to_csv(inventory_output_path, index=False)
    print(f"Saved: {inventory_output_path}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
