import math


def get_pattern_policy(pattern: str):
    """
    Return pattern-specific inventory policy parameters.
    """
    pattern = str(pattern).strip().lower()

    policy_map = {
        "intermittent": {
            "service_level": 0.90,
            "holding_cost": 0.8,
            "shortage_cost": 4.0,
            "policy_type": "conservative_replenishment",
        },
        "burst": {
            "service_level": 0.97,
            "holding_cost": 1.0,
            "shortage_cost": 6.0,
            "policy_type": "buffered_replenishment",
        },
        "smooth": {
            "service_level": 0.95,
            "holding_cost": 1.0,
            "shortage_cost": 5.0,
            "policy_type": "stable_replenishment",
        },
        "volatile": {
            "service_level": 0.98,
            "holding_cost": 1.2,
            "shortage_cost": 7.0,
            "policy_type": "high_resilience_replenishment",
        },
    }

    return policy_map.get(
        pattern,
        {
            "service_level": 0.95,
            "holding_cost": 1.0,
            "shortage_cost": 5.0,
            "policy_type": "default_replenishment",
        },
    )


def get_z_score(service_level: float) -> float:
    """
    Approximate z-score lookup for common service levels.
    """
    z_map = {
        0.90: 1.28,
        0.95: 1.65,
        0.97: 1.88,
        0.98: 2.05,
        0.99: 2.33,
    }

    rounded_level = round(service_level, 2)
    return z_map.get(rounded_level, 1.65)


def optimize_inventory_policy(
    mean_demand: float,
    std: float,
    lead_time: float,
    pattern: str,
    review_period: float = 7.0,
):
    """
    Pattern-aware inventory policy optimization.

    Parameters
    ----------
    mean_demand : float
        Average daily demand.
    std : float
        Standard deviation of daily demand.
    lead_time : float
        Lead time in days.
    pattern : str
        Demand pattern label.
    review_period : float
        Periodic review cycle in days.

    Returns
    -------
    dict
        Optimized inventory policy parameters.
    """
    mean_demand = max(float(mean_demand), 0.0)
    std = max(float(std), 0.0)
    lead_time = max(float(lead_time), 1.0)
    review_period = max(float(review_period), 1.0)

    policy = get_pattern_policy(pattern)
    service_level = policy["service_level"]
    holding_cost = policy["holding_cost"]
    shortage_cost = policy["shortage_cost"]
    policy_type = policy["policy_type"]

    z = get_z_score(service_level)

    # Demand during lead time
    lead_time_demand = mean_demand * lead_time

    # Demand variability during lead time
    lead_time_std = std * math.sqrt(lead_time)

    # Safety stock
    safety_stock_opt = z * lead_time_std

    # Reorder point
    reorder_point_opt = lead_time_demand + safety_stock_opt

    # Order-up-to level for periodic review
    order_up_to_level = mean_demand * (lead_time + review_period) + z * std * math.sqrt(lead_time + review_period)

    # Simplified EOQ-like order quantity
    annual_demand = mean_demand * 365.0
    ordering_cost = 20.0

    if holding_cost <= 0:
        holding_cost = 1.0

    optimal_order_qty = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost) if annual_demand > 0 else 0.0

    # Approximate total cost
    avg_cycle_stock = optimal_order_qty / 2.0
    avg_inventory = avg_cycle_stock + safety_stock_opt
    holding_cost_total = avg_inventory * holding_cost
    shortage_cost_total = shortage_cost * std
    total_cost = holding_cost_total + shortage_cost_total

    return {
        "pattern_policy_type": policy_type,
        "recommended_service_level": round(service_level, 2),
        "holding_cost_weight": round(holding_cost, 4),
        "shortage_cost_weight": round(shortage_cost, 4),
        "safety_stock_opt": round(safety_stock_opt, 4),
        "reorder_point_opt": round(reorder_point_opt, 4),
        "order_up_to_level_opt": round(order_up_to_level, 4),
        "optimal_order_qty": round(optimal_order_qty, 4),
        "estimated_total_cost": round(total_cost, 4),
    }