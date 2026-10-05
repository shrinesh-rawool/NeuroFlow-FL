"""
scripts/task23_multiseed.py
Task 2.3: Wire multi-seed route generation into training
"""
import sys
import os
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from single_intersection_simulation.route_generator import generate_route_rates
from single_intersection_simulation.write_route import write_route_file
from env.traffic_env import TrafficEnv

def create_seed_environment(seed):
    base_dir = "single_intersection_simulation"
    base_route_path = os.path.join(base_dir, "traffic.rou.xml")
    base_cfg_path = os.path.join(base_dir, "simulation.sumocfg")
    
    new_route_name = f"traffic_seed{seed}.rou.xml"
    new_route_path = os.path.join(base_dir, new_route_name)
    new_cfg_name = f"simulation_seed{seed}.sumocfg"
    new_cfg_path = os.path.join(base_dir, new_cfg_name)
    
    # 1. Generate rates and write route file
    rates = generate_route_rates(seed)
    write_route_file(rates, base_route_path, new_route_path)
    
    # 2. Write seed-specific .sumocfg pointing to the new route file
    tree = ET.parse(base_cfg_path)
    root = tree.getroot()
    route_files_node = root.find("input").find("route-files")
    route_files_node.set("value", new_route_name)
    tree.write(new_cfg_path, xml_declaration=True, encoding="UTF-8")
    
    # 3. Construct and return isolated TrafficEnv
    return TrafficEnv(sumo_cfg=new_cfg_path)

def main():
    test_seeds = [42, 99]
    for seed in test_seeds:
        print(f"\nSetting up environment for seed {seed}...")
        env = create_seed_environment(seed)
        try:
            env.reset()
            # Step the simulation briefly to prove the generated XML is valid SUMO input
            for _ in range(5):
                env.step(0)
            print(f"PASS: Environment for seed {seed} generated, initialized, and stepped successfully.")
        except Exception as e:
            print(f"FAIL: Environment for seed {seed} crashed. Error: {e}")
        finally:
            env.close()

if __name__ == "__main__":
    main()