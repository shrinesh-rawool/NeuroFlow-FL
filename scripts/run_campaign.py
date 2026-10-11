"""
scripts/run_campaign.py
Task 2.4: Multi-seed training campaign (DQN + PPO) for NeuroFlow-FL single-intersection track.
Supports distributed execution via CLI arguments.
"""
import sys
import os
import csv
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import traci

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from single_intersection_simulation.route_generator import generate_route_rates
from single_intersection_simulation.write_route import write_route_file
from env.traffic_env import TrafficEnv
from agents.dqn_agent import train_dqn
from agents.ppo_agent import train_ppo

CSV_HEADER = [
    "Algorithm", "Seed", "Reward", "Avg_Wait_s", "Avg_Queue_veh",
    "Throughput_veh", "Backlog_veh", "Switch_Frac",
]


def create_env_for_seed(seed, mode):
    base_dir = "single_intersection_simulation"
    base_route = os.path.join(base_dir, "traffic.rou.xml")
    base_cfg = os.path.join(base_dir, "simulation.sumocfg")

    # Generate rates and write route file (passing seed for randomized emergency vehicles)
    rates = generate_route_rates(seed, baseline_rate=500.0, noise_pct=0.15)
    new_route_name = f"traffic_{mode}_seed{seed}.rou.xml"
    write_route_file(rates, base_route, os.path.join(base_dir, new_route_name), seed=seed)

    # Update .sumocfg to point to the new route file
    new_cfg_name = f"simulation_{mode}_seed{seed}.sumocfg"
    new_cfg_path = os.path.join(base_dir, new_cfg_name)

    tree = ET.parse(base_cfg)
    root = tree.getroot()
    root.find("input").find("route-files").set("value", new_route_name)
    tree.write(new_cfg_path, xml_declaration=True, encoding="UTF-8")

    return TrafficEnv(sumo_cfg=new_cfg_path)


def evaluate_model(model, eval_env, num_steps=720):
    """
    Returns a dict of metrics for one deterministic evaluation episode:
      reward        total episode reward
      avg_wait      mean per-vehicle waiting time, sampled each decision step
      avg_queue     mean total halting-vehicle count across approaches
      throughput    vehicles that completed their trip (env.arrived_total)
      backlog       vehicles still waiting to enter the network at the end
      switch_frac   fraction of decision steps where the policy chose "switch"
    """
    obs, info = eval_env.reset()
    total_reward = 0.0
    actions = []
    queues = []
    wait_times = []

    for _ in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward

        # obs = [q_n, q_s, q_e, q_w, phase] normalized. Multiply by 50 to get raw counts.
        queues.append(np.sum(obs[0:4]) * 50.0)

        # Extract raw wait times for all vehicles currently in the network.
        # Switch explicitly so we always query THIS env's simulation.
        traci.switch(eval_env.label)
        for veh_id in traci.vehicle.getIDList():
            wait_times.append(traci.vehicle.getWaitingTime(veh_id))

        if terminated or truncated:
            break

    # Read end-of-episode stats BEFORE closing the simulation.
    traci.switch(eval_env.label)
    backlog = len(traci.simulation.getPendingVehicles())
    throughput = getattr(eval_env, "arrived_total", 0)

    eval_env.close()

    return {
        "reward": total_reward,
        "avg_wait": float(np.mean(wait_times)) if wait_times else 0.0,
        "avg_queue": float(np.mean(queues)) if queues else 0.0,
        "throughput": throughput,
        "backlog": backlog,
        "switch_frac": float(np.mean(actions)) if actions else 0.0,
    }


def init_csv(csv_file):
    """Create the CSV with its header exactly once. Opening with mode 'x'
    fails atomically if another process created it first, so concurrent
    campaigns can't write duplicate headers."""
    try:
        with open(csv_file, mode="x", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
    except FileExistsError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run NeuroFlow-FL Campaign")
    parser.add_argument("--algo", type=str, choices=["DQN", "PPO", "ALL"], default="ALL", help="Algorithm to train")
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 42, 99, 128, 256, 512, 1024, 2048], help="List of random seeds")
    parser.add_argument("--timesteps", type=int, default=200000, help="Training timesteps per run")
    parser.add_argument("--eval-freq", type=int, default=10000, help="EvalCallback frequency (timesteps)")
    parser.add_argument("--csv", type=str, default="results/campaign_metrics.csv", help="Output metrics CSV")
    args = parser.parse_args()

    algorithms = {}
    if args.algo in ["DQN", "ALL"]:
        algorithms["DQN"] = train_dqn
    if args.algo in ["PPO", "ALL"]:
        algorithms["PPO"] = train_ppo

    seeds = args.seeds
    timesteps = args.timesteps
    csv_file = args.csv

    # Ensure output directories exist even if old results were cleared
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    csv_dir = os.path.dirname(csv_file)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    init_csv(csv_file)

    print("=" * 60)
    print("STARTING TASK 2.4 TRAINING CAMPAIGN")
    print(f"Algorithms: {list(algorithms.keys())} | Seeds: {seeds} | Timesteps: {timesteps} | Eval freq: {args.eval_freq}")
    print(f"Metrics CSV: {csv_file}")
    print("=" * 60)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        for algo_name, train_func in algorithms.items():
            print(f"\n>>> Running Campaign for {algo_name}")

            for seed in seeds:
                run_id = f"{algo_name}_seed{seed}"
                print(f"  -> Training {run_id}")

                train_env = create_env_for_seed(seed, f"{algo_name}_train")
                eval_env = create_env_for_seed(seed + 1000, f"{algo_name}_eval")
                metric_eval_env = None

                try:
                    model = train_func(
                        env=train_env,
                        eval_env=eval_env,
                        timesteps=timesteps,
                        run_name=run_id,
                        seed=seed,
                        eval_freq=args.eval_freq,
                    )

                    # Evaluate model on the held-out demand (seed + 1000)
                    print(f"  -> Evaluating {run_id}")
                    metric_eval_env = create_env_for_seed(seed + 1000, f"{algo_name}_metric_eval")
                    m = evaluate_model(model, metric_eval_env)

                    print(
                        f"     [Result] Reward: {m['reward']:.2f} | Avg Wait: {m['avg_wait']:.2f}s | "
                        f"Avg Total Q: {m['avg_queue']:.2f} | Throughput: {m['throughput']} | "
                        f"Backlog: {m['backlog']}"
                    )
                    print(f"     [Policy] fraction of switch actions: {m['switch_frac']:.2f}")

                    # Save Model
                    model_path = os.path.join("models", run_id)
                    model.save(model_path)
                    print(f"     [Saved Model] {model_path}.zip")

                    # Save Data
                    writer.writerow([
                        algo_name, seed,
                        round(m["reward"], 2), round(m["avg_wait"], 2), round(m["avg_queue"], 2),
                        m["throughput"], m["backlog"], round(m["switch_frac"], 3),
                    ])
                    f.flush()  # Force write to disk immediately in case of crash

                except Exception as e:
                    print(f"     [ERROR] Run failed for {run_id}: {type(e).__name__}: {e}")
                finally:
                    train_env.close()
                    eval_env.close()
                    if metric_eval_env is not None:
                        metric_eval_env.close()

    print("\n" + "=" * 60)
    print(f"CAMPAIGN COMPLETE for given args. Data saved to {csv_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()