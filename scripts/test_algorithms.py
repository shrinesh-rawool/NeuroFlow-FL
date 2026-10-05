"""
scripts/test_algorithms.py
Quick configuration check for DQN and PPO agent wrappers.
"""
import sys
import os
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from single_intersection_simulation.route_generator import generate_route_rates
from single_intersection_simulation.write_route import write_route_file
from env.traffic_env import TrafficEnv
from agents.dqn_agent import train_dqn
from agents.ppo_agent import train_ppo

def create_quick_env(seed, mode):
    base_dir = "single_intersection_simulation"
    base_route = os.path.join(base_dir, "traffic.rou.xml")
    base_cfg = os.path.join(base_dir, "simulation.sumocfg")
    
    rates = generate_route_rates(seed, baseline_rate=500.0, noise_pct=0.15)
    new_route_name = f"traffic_test_{mode}_seed{seed}.rou.xml"
    write_route_file(rates, base_route, os.path.join(base_dir, new_route_name), seed=seed)
    
    new_cfg_name = f"simulation_test_{mode}_seed{seed}.sumocfg"
    new_cfg_path = os.path.join(base_dir, new_cfg_name)
    
    tree = ET.parse(base_cfg)
    root = tree.getroot()
    root.find("input").find("route-files").set("value", new_route_name)
    tree.write(new_cfg_path, xml_declaration=True, encoding="UTF-8")
    
    return TrafficEnv(sumo_cfg=new_cfg_path)

def main():
    print("Testing Algorithm Configurations (1000 timesteps each)...")
    seed = 999
    timesteps = 1000  # Tiny number just to verify the code paths work

    algorithms = {"DQN": train_dqn, "PPO": train_ppo}

    for algo_name, train_func in algorithms.items():
        print(f"\n[Testing {algo_name}]")
        train_env = create_quick_env(seed, f"{algo_name}_train")
        eval_env = create_quick_env(seed + 1, f"{algo_name}_eval")
        
        try:
            # Train model
            run_id = f"{algo_name}_seed{seed}"
            model = train_func(env=train_env, eval_env=eval_env, timesteps=timesteps, run_name=run_id)
            if model is not None:
                print(f"PASS: {algo_name} initialized, trained, and returned a model successfully.")
            else:
                print(f"FAIL: {algo_name} returned None instead of a model.")
        except Exception as e:
            print(f"FAIL: {algo_name} crashed. Error: {e}")
        finally:
            train_env.close()
            eval_env.close()

if __name__ == "__main__":
    main()