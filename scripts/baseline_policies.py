import sys, os
import numpy as np
import traci
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_campaign import create_env_for_seed, evaluate_model


class FixedTime:
    def __init__(self, green_s=40, tls="J4"):
        self.green_s, self.tls = green_s, tls
    def predict(self, obs, deterministic=True):
        spent = traci.trafficlight.getSpentDuration(self.tls)
        return np.array(int(spent >= self.green_s)), None


class FixedTime90:
    def predict(self, obs, deterministic=True):
        return np.array(0), None


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    for name, policy in [("FixedTime40", FixedTime(40)), ("FixedTime30", FixedTime(30)), ("FixedTime90", FixedTime90())]:
        env = create_env_for_seed(seed + 1000, f"baseline_{name}")
        m = evaluate_model(policy, env)
        print(name, {k: round(v, 2) for k, v in m.items()})