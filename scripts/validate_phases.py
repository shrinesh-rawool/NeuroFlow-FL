"""
scripts/validate_phases.py
Task 2.1: Phase-index validation script for NeuroFlow-FL single-intersection track.
"""
import sys
import os
import traci

# Ensure the parent directory is in the path to import env modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.traffic_env import TrafficEnv

def main():
    print("Initializing TrafficEnv for Phase Validation...")
    try:
        # Construct the environment pointing to the single-intersection config
        env = TrafficEnv(sumo_cfg="single_intersection_simulation/simulation.sumocfg")
    except Exception as e:
        print(f"FAIL: Could not initialize environment. Error: {e}")
        sys.exit(1)

    env.reset()
    print("Environment reset successfully. Stepping through simulation...")

    checked_phases = set()
    # Fetch initial phase via the controller's tls_id
    last_phase = traci.trafficlight.getPhase(env.controller.tls_id)

    # Run for 200 steps (1000 simulation seconds at 5s per step) to cycle through all phases
    steps = 200 
    for i in range(steps):
        # Heuristic to force phase switches: apply action 1 ("switch") every 10 steps (50s)
        action = 1 if i % 10 == 0 else 0
        obs, reward, terminated, truncated, info = env.step(action)

        current_phase = traci.trafficlight.getPhase(env.controller.tls_id)

        # Detect phase change and validate the NEW phase's duration
        if current_phase != last_phase:
            is_valid, msg = env.controller.validate_phase_durations()
            print(msg)

            if not is_valid:
                print(f"\nFAIL: Phase validation failed at step {i}!")
                env.close()
                sys.exit(1)

            checked_phases.add(current_phase)
            last_phase = current_phase

        if terminated or truncated:
            env.reset()

    env.close()

    # Verify that the core green phases (0, 2, 4, 6) were actually reached and checked
    expected_green_phases = {0, 2, 4, 6}
    if not expected_green_phases.issubset(checked_phases):
        missing = expected_green_phases - checked_phases
        print(f"\nFAIL: Simulation completed, but failed to validate all green phases. Missing: {missing}")
        sys.exit(1)

    print("\nSUCCESS: All green and yellow phases validated successfully per the 8-phase design!")
    sys.exit(0)

if __name__ == "__main__":
    main()