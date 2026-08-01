import numpy as np

def generate_route_rates(seed, baseline_rate=500.0, noise_pct=0.15):
    """
    Returns a dict of flow_id -> perturbed perHour value,
    deterministic given the seed.
    """
    rng = np.random.default_rng(seed)

    flow_ids = [
        "flow_EN", "flow_ES", "flow_EW",
        "flow_NE", "flow_NS", "flow_NW",
        "flow_SE", "flow_SN", "flow_SW",
        "flow_WE", "flow_WN", "flow_WS",
    ]

    rates = {}
    for fid in flow_ids:
        # uniform perturbation: rate in [baseline*(1-noise), baseline*(1+noise)]
        factor = 1.0 + rng.uniform(-noise_pct, noise_pct)
        rates[fid] = round(baseline_rate * factor, 2)

    return rates
