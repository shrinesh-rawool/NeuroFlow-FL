import sys, os, csv, argparse
import numpy as np
import traci
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.run_campaign import create_env_for_seed, evaluate_model

HEADER = ["Policy", "Rate", "Seed", "Reward", "Avg_Wait_s", "Avg_Queue_veh",
          "Throughput_veh", "Backlog_veh", "Switch_Frac"]


class FixedTime:
    """Switch once the current through phase has been green for green_s seconds.
    Use multiples of 5: decisions happen on a 5s grid."""
    def __init__(self, green_s=40, tls="J4"):
        self.green_s, self.tls = green_s, tls

    def predict(self, obs, deterministic=True):
        spent = traci.trafficlight.getSpentDuration(self.tls)
        return np.array(int(spent >= self.green_s)), None


class FixedTime90:
    """Always 'keep'; the env's max_green cap (90s) forces the switch."""
    def predict(self, obs, deterministic=True):
        return np.array(0), None


class MaxQueue:
    """Heuristic: switch when the waiting axis has margin more halted vehicles
    than the green axis. min_green is enforced by the env."""
    def __init__(self, margin=3, tls="J4"):
        self.margin, self.tls = margin, tls

    def predict(self, obs, deterministic=True):
        q = np.asarray(obs[:4]) * 50.0            # [N, S, E, W]
        ns, ew = q[0] + q[1], q[2] + q[3]
        phase = traci.trafficlight.getPhase(self.tls)
        if phase == 0:
            switch = ew > ns + self.margin
        elif phase == 4:
            switch = ns > ew + self.margin
        else:
            switch = False
        return np.array(int(switch)), None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 42, 99, 128, 256, 512, 1024, 2048])
    parser.add_argument("--rates", type=float, nargs="+", default=[225])
    parser.add_argument("--greens", type=int, nargs="+", default=[15, 20, 25, 30, 35, 40])
    parser.add_argument("--csv", type=str, default="results/baselines.csv")
    args = parser.parse_args()

    policies = {f"FixedTime{g}": FixedTime(g) for g in args.greens}
    policies["FixedTime90"] = FixedTime90()
    policies["MaxQueue"] = MaxQueue()

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for rate in args.rates:
            for seed in args.seeds:
                for name, policy in policies.items():
                    env = create_env_for_seed(seed + 1000, f"baseline_{name}_r{int(rate)}", baseline_rate=rate)
                    m = evaluate_model(policy, env)
                    w.writerow([name, int(rate), seed, round(m["reward"], 2), round(m["avg_wait"], 2),
                                round(m["avg_queue"], 2), m["throughput"], m["backlog"], round(m["switch_frac"], 3)])
                    f.flush()
                    print(f"r={int(rate)} seed={seed} {name:12s} reward={m['reward']:.0f} "
                          f"wait={m['avg_wait']:.1f} queue={m['avg_queue']:.1f} "
                          f"thru={m['throughput']} backlog={m['backlog']}")