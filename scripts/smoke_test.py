"""
scripts/smoke_test.py
Task 2.2: Smoke-test episodes for NeuroFlow-FL single-intersection track.

Runs episodes:
1. Background traffic with the seed's own emergency vehicles and a simple
     switching heuristic. Checks reward sanity, general flow, and hold release.
2. Emergency direction tests (W_to_E and E_to_W) with no agent switching.
     Checks that the phase moves to EW green (4) and releases on the exit road.
3. Red-arrival wait test where an EW emergency arrives during NS green (phase 0)
     and must wait for yellow/protected clearance.
"""
import sys
import os
import csv
import traci
import numpy as np
import xml.etree.ElementTree as ET

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.traffic_env import TrafficEnv
from env.emergency_preemption import APPROACH_EDGES

MAX_STEPS = 720  # 3600 s / 5 s per decision step
SUMO_CFG = "single_intersection_simulation/simulation.sumocfg"
EW_TEST_ROUTE = "single_intersection_simulation/smoke_ew_emergency.rou.xml"
EW_TEST_CFG = "single_intersection_simulation/smoke_ew_emergency.sumocfg"


def make_ew_test_files(route="W_to_E", tag="ew"):
    """Writes a route file with one emergency on the given route, plus the base
    background flows, and a matching sumocfg."""
    base = ET.parse("single_intersection_simulation/traffic.rou.xml").getroot()
    for v in base.findall("vehicle"):
        base.remove(v)
    ET.SubElement(base, "vehicle", {
        "id": f"emg_{tag}_test", "type": "emergency",
        "route": route, "depart": "100.00",
    })
    rou = f"single_intersection_simulation/smoke_{tag}.rou.xml"
    ET.ElementTree(base).write(rou, xml_declaration=True, encoding="UTF-8")

    cfg = ET.parse("single_intersection_simulation/simulation.sumocfg").getroot()
    cfg.find("input").find("route-files").set("value", os.path.basename(rou))
    cfg_path = f"single_intersection_simulation/smoke_{tag}.sumocfg"
    ET.ElementTree(cfg).write(cfg_path, xml_declaration=True, encoding="UTF-8")
    return cfg_path


def make_wait_test_files(depart="10.00"):
    """EW emergency departing while NS (phase 0) is green, so it waits on red."""
    base = ET.parse("single_intersection_simulation/traffic.rou.xml").getroot()
    for v in base.findall("vehicle"):
        base.remove(v)
    ET.SubElement(base, "vehicle", {
        "id": "emg_wait_test", "type": "emergency",
        "route": "W_to_E", "depart": depart,
    })
    path = "single_intersection_simulation/smoke_wait.rou.xml"
    ET.ElementTree(base).write(path, xml_declaration=True, encoding="UTF-8")

    cfg = ET.parse("single_intersection_simulation/simulation.sumocfg").getroot()
    cfg.find("input").find("route-files").set("value", os.path.basename(path))
    cfg_path = "single_intersection_simulation/smoke_wait.sumocfg"
    ET.ElementTree(cfg).write(cfg_path, xml_declaration=True, encoding="UTF-8")
    return cfg_path


def log_emergency(env, step):
    """Prints emergency state for vehicles on the approach or in the junction.
    Returns a list of (veh_id, phase, hold, road) tuples for later checks."""
    rows = []
    phase = traci.trafficlight.getPhase(env.tls_id)
    for veh in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(veh) != "emergency":
            continue
        road = traci.vehicle.getRoadID(veh)
        if road not in APPROACH_EDGES and not road.startswith(":J4_"):
            continue
        hold = env.emergency.should_hold_green()
        print(f"step={step} phase={phase} veh={veh} road={road} "
              f"hold={hold} dir={env.emergency.current_direction}")
        rows.append((veh, phase, hold, road))
    return rows


def run_background_episode():
    print("\n[Episode 1] Background traffic, seed emergencies, naive switching")
    env = TrafficEnv(sumo_cfg=SUMO_CFG)
    obs, _ = env.reset()

    rewards, flow_vehicles = [], set()
    arrived_total = 0
    pegged = 0
    emg_max_wait = {}
    emg_released = {}   # veh_id -> True once it reached an outgoing edge with hold False

    with open("smoke_test_metrics.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Phase", "Q_North", "Q_South", "Q_East", "Q_West", "Reward"])

        for step in range(MAX_STEPS):
            action = 1 if step % 8 == 0 else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if reward <= -5000:
                pegged += 1

            q_n, q_s, q_e, q_w, phase, _ = obs
            writer.writerow([step, phase * 7.0, q_n * 50, q_s * 50, q_e * 50, q_w * 50, reward])

            log_emergency(env, step)

            for veh in traci.vehicle.getIDList():
                if traci.vehicle.getTypeID(veh) != "emergency":
                    continue
                road = traci.vehicle.getRoadID(veh)
                if road.endswith("_out") and not env.emergency.should_hold_green():
                    emg_released[veh] = True

            for veh in traci.vehicle.getIDList():
                flow_vehicles.add(veh)
                if traci.vehicle.getTypeID(veh) == "emergency":
                    w = traci.vehicle.getWaitingTime(veh)
                    emg_max_wait[veh] = max(emg_max_wait.get(veh, 0.0), w)
            arrived_total += traci.simulation.getArrivedNumber()

            if terminated or truncated:
                break

    env.close()

    avg_reward = float(np.mean(rewards))
    print(f"Episode length: {len(rewards)} steps ({len(rewards) * 5} s)")
    print(f"Average reward: {avg_reward:.2f} | Minimum reward: {np.min(rewards):.2f}")
    if pegged > MAX_STEPS * 0.1:
        print("FAIL: rewards heavily pegged at the -5000 clip floor.")
    else:
        print("PASS: rewards are not pegged at the -5000 clip floor.")

    print(f"Emergency vehicles seen: {len(emg_max_wait)}")
    for veh, w in emg_max_wait.items():
        print(f"  {veh}: max wait {w:.1f} s")
    for veh in emg_max_wait:
        status = "released" if emg_released.get(veh) else "NOT released"
        print(f"  {veh}: hold {status} on exit")
    if any(not emg_released.get(v) for v in emg_max_wait):
        print("FAIL: at least one emergency was not released on exit.")
    else:
        print("PASS: every seed emergency released its hold on exit.")
    print("NOTE: hold/phase checks for these vehicles are in the log lines above.")

    print(f"Vehicles seen in network: {len(flow_vehicles)} | Arrived at destination: {arrived_total}")
    if arrived_total > 100:
        print("PASS: significant throughput, no sign of a permanent deadlock.")
    else:
        print("WARNING: low throughput. Check for deadlock or starvation.")


def run_ew_emergency_test():
    print("\n[Episode 2] Single EW emergency (W_to_E), no agent switching")
    cfg = make_ew_test_files()
    env = TrafficEnv(sumo_cfg=cfg)
    env.reset()

    phases_seen = []
    seen_emergency = False
    released = False
    max_wait = 0.0
    exit_roads = {"east_out", "west_out"}  # W_to_E exits on east_out

    for step in range(MAX_STEPS):
        env.step(0)  # agent never requests a switch; emergency logic alone acts

        phase = traci.trafficlight.getPhase(env.tls_id)
        if not phases_seen or phases_seen[-1] != phase:
            phases_seen.append(phase)

        rows = log_emergency(env, step)
        if rows:
            seen_emergency = True

        veh_ids = traci.vehicle.getIDList()
        if veh_ids and "emg_ew_test" in veh_ids:
            max_wait = max(max_wait, traci.vehicle.getWaitingTime("emg_ew_test"))
            road_now = traci.vehicle.getRoadID("emg_ew_test")
            if road_now in exit_roads:
                hold_now = env.emergency.should_hold_green()
                print(f"step={step} on exit {road_now} hold={hold_now} "
                      f"dir={env.emergency.current_direction}")
                if not hold_now:
                    released = True

        if seen_emergency and released:
            break

    env.close()

    print(f"Phase sequence: {phases_seen}")
    expected_green = 4
    phase_order_ok = (
        phases_seen == [0, expected_green]
        or (phases_seen[:2] == [0, 2] and phases_seen[-1] == expected_green)
    )
    if phase_order_ok and phases_seen[-1] == expected_green:
        print("PASS: phase moved 0 -> (yellow) -> protected -> 4 (EW green).")
    else:
        print(f"FAIL: unexpected phase sequence {phases_seen}.")

    if released:
        print("PASS: emergency hold released explicitly on the exit road.")
    else:
        print("FAIL: emergency hold never released on exit road.")
    print(f"Emergency max wait: {max_wait:.1f} s")


def run_emergency_direction_test(route, veh_id, exit_road, expected_green):
    print(f"\n[{route}] emergency, expecting green phase {expected_green}")
    tag = route.lower()
    cfg = make_ew_test_files(route=route, tag=tag)
    env = TrafficEnv(sumo_cfg=cfg)
    env.reset()

    phases_seen, released = [], False
    for step in range(MAX_STEPS):
        env.step(0)
        phase = traci.trafficlight.getPhase(env.tls_id)
        if not phases_seen or phases_seen[-1] != phase:
            phases_seen.append(phase)
        if veh_id in traci.vehicle.getIDList():
            if traci.vehicle.getRoadID(veh_id) == exit_road and not env.emergency.should_hold_green():
                released = True
                break
    env.close()

    print(f"Phase sequence: {phases_seen}")
    if phases_seen and phases_seen[-1] == expected_green:
        print(f"PASS: phase ends on green {expected_green}.")
    else:
        print("FAIL: phase does not end on the expected green.")
    print("PASS: hold released on exit." if released else "FAIL: hold not released on exit.")


def run_wait_test():
    print("\n[Episode 3] EW emergency arriving on red, agent holds NS green")
    cfg = make_wait_test_files(depart="10.00")
    env = TrafficEnv(sumo_cfg=cfg)
    env.reset()

    max_wait = 0.0
    seen_wait_veh = False
    for step in range(MAX_STEPS):
        env.step(0)
        veh_ids = traci.vehicle.getIDList()
        if "emg_wait_test" in veh_ids:
            seen_wait_veh = True
            max_wait = max(max_wait, traci.vehicle.getWaitingTime("emg_wait_test"))
        elif seen_wait_veh:
            break
    env.close()

    print(f"Emergency max wait: {max_wait:.1f} s")
    if 0.0 < max_wait < 30:
        print("PASS: emergency waited on red and was cleared within 30 s.")
    elif max_wait == 0.0:
        print("FAIL: emergency never waited. Depart time may not put it on red.")
    else:
        print("FAIL: emergency waited 30 s or more.")


def main():
    run_background_episode()
    run_emergency_direction_test("W_to_E", "emg_w_to_e_test", "east_out", expected_green=4)
    run_emergency_direction_test("E_to_W", "emg_e_to_w_test", "west_out", expected_green=4)
    run_wait_test()


if __name__ == "__main__":
    main()
