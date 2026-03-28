import numpy as np


pattern_policy = {
    "smooth": {
        "safety_factor": 0.8,
        "description": "Smooth demand → lower safety stock"
    },
    "intermittent": {
        "safety_factor": 1.2,
        "description": "Intermittent demand → moderate safety stock"
    },
    "burst": {
        "safety_factor": 1.4,
        "description": "Burst demand → higher safety stock"
    },
    "volatile": {
        "safety_factor": 1.6,
        "description": "Volatile demand → highest safety stock"
    }
}


def compute_safety_stock(std, lead_time, z=1.65):
    """
    Calculate basic safety stock.
    """
    return z * std * np.sqrt(lead_time)


def adjusted_safety_stock(std, lead_time, pattern):
    base_ss = compute_safety_stock(std, lead_time)
    factor = pattern_policy.get(pattern, {}).get("safety_factor", 1.0)
    return base_ss * factor


def reorder_point(mean_demand, lead_time, safety_stock):
    return mean_demand * lead_time + safety_stock


def inventory_decision(mean_demand, std, lead_time, pattern):
    ss = adjusted_safety_stock(std, lead_time, pattern)
    rop = reorder_point(mean_demand, lead_time, ss)

    return {
        "pattern": pattern,
        "safety_stock": float(ss),
        "reorder_point": float(rop),
        "policy_description": pattern_policy.get(pattern, {}).get("description", "Default inventory policy")
    }