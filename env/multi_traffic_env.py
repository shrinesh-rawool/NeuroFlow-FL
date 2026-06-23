# env/multi_traffic_env.py
# Independent multi-agent environment — one RL agent per intersection.
# Each agent observes only its own intersection state.

import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces
import time


# Update these after running Step 4 above
INTERSECTION_CONFIGS = {
    "J1": {
        "lanes_in": ["start_J1_0", "start_J1_1",
                     "J1N_J1_0", "J1S_J1_0",
                     "J2_J1_0",  "J2_J1_1"],
    },
    "J2": {
        "lanes_in": ["J1_J2_0", "J1_J2_1",
                     "J2N_J2_0", "J2S_J2_0",
                     "J3_J2_0",  "J3_J2_1"],
    },
    "J3": {
        "lanes_in": ["J2_J3_0", "J2_J3_1",
                     "J3N_J3_0", "J3S_J3_0",
                     "J4_J3_0",  "J4_J3_1"],
    },
    "J4": {
        "lanes_in": ["J3_J4_0", "J3_J4_1",
                     "J4N_J4_0", "J4S_J4_0"],
    },
}

TLS_IDS = list(INTERSECTION_CONFIGS.keys())   # ["J1","J2","J3","J4"]


class MultiAgentTrafficEnv:
    """
    Wraps a single SUMO simulation exposing 4 independent agents.
    Each agent controls one traffic light and sees only its own state.

    Usage:
        env = MultiAgentTrafficEnv("network_multi/arterial.sumocfg")
        obs = env.reset()                    # dict: {tls_id: np.array}
        obs, rews, dones, info = env.step({  # dict inputs/outputs
            "J1": action_j1,
            "J2": action_j2,
            "J3": action_j3,
            "J4": action_j4,
        })
    """

    def __init__(self, sumo_cfg, max_steps=3600, delta_time=5, min_green=10, use_gui=False, delay=0):
        self.sumo_cfg   = sumo_cfg
        self.max_steps  = max_steps
        self.delta_time = delta_time
        self.min_green  = min_green

        self.use_gui = use_gui
        self.delay = delay

        self.step_count = 0
        self.phase_time = {tls: 0 for tls in TLS_IDS}

        # Each agent: 5-dim observation, 2 discrete actions — same as Phase 1
        self.observation_spaces = {
            tls: spaces.Box(low=np.zeros(5, dtype=np.float32),
                            high=np.ones(5,  dtype=np.float32),
                            dtype=np.float32)
            for tls in TLS_IDS
        }
        self.action_spaces = {
            tls: spaces.Discrete(2) for tls in TLS_IDS
        }

        binary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" if use_gui \
         else r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
        self.sumo_cmd = [
            binary,
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end"
        ]
        if use_gui and delay > 0:
            self.sumo_cmd += ["--delay", str(delay)]

    def reset(self):
        if traci.isLoaded():
            traci.close()

        traci.start(self.sumo_cmd)
        self.step_count = 0
        self.phase_time = {tls: 0 for tls in TLS_IDS}

        return self._get_all_states()

    def step(self, actions):
        """
        actions: dict {tls_id: int (0 or 1)}
        Returns: obs, rewards, dones, info  — all dicts keyed by tls_id
        """

        # Apply actions for each agent
        for tls_id, action in actions.items():
            self.phase_time[tls_id] += self.delta_time

            # Enforce minimum green time per intersection
            if self.phase_time[tls_id] < self.min_green:
                action = 0

            self._apply_action(tls_id, action)

        # Advance simulation
        for _ in range(self.delta_time):
            traci.simulationStep()

        self.step_count += self.delta_time

        if self.use_gui and self.delay > 0:
            time.sleep(self.delay)

        obs     = self._get_all_states()
        rewards = self._compute_all_rewards()

        done = self.step_count >= self.max_steps
        dones = {tls: done for tls in TLS_IDS}
        dones["__all__"] = done

        return obs, rewards, dones, {}

    def close(self):
        if traci.isLoaded():
            traci.close()

    # ── Private helpers ────────────────────────────────────────────────

    def _get_all_states(self):
        return {tls: self._get_state(tls) for tls in TLS_IDS}

    def _get_state(self, tls_id):
        lanes = INTERSECTION_CONFIGS[tls_id]["lanes_in"]

        # Group into approach directions for queue measurement
        # Lanes 0-1: main arterial, 2-3: side roads, 4-5: reverse arterial
        groups = [lanes[i:i+2] for i in range(0, len(lanes), 2)]

        queues = []
        for group in groups[:4]:   # up to 4 directions
            q = sum(traci.lane.getLastStepHaltingNumber(l)
                    for l in group if l in self._get_valid_lanes())
            queues.append(q)

        # Pad to always have 4 queue values
        while len(queues) < 4:
            queues.append(0)

        phase = traci.trafficlight.getPhase(tls_id)
        state = np.array(queues[:4] + [phase], dtype=np.float32)
        state[0:4] /= 50.0
        state[4]   /= 3.0

        return np.clip(state, 0.0, 1.0)

    def _get_valid_lanes(self):
        """Cache of lanes that actually exist in the simulation."""
        if not hasattr(self, "_valid_lanes_cache"):
            self._valid_lanes_cache = set(traci.lane.getIDList())
        return self._valid_lanes_cache

    def _apply_action(self, tls_id, action):
        if action == 1:
            current_phase = traci.trafficlight.getPhase(tls_id)

            if current_phase in (1, 3):
                return

            if current_phase == 0:
                traci.trafficlight.setPhase(tls_id, 1)
                traci.simulationStep()
                traci.trafficlight.setPhase(tls_id, 2)
                self.phase_time[tls_id] = 0

            elif current_phase == 2:
                traci.trafficlight.setPhase(tls_id, 3)
                traci.simulationStep()
                traci.trafficlight.setPhase(tls_id, 0)
                self.phase_time[tls_id] = 0

    def _compute_all_rewards(self):
        return {tls: self._compute_reward(tls) for tls in TLS_IDS}

    def _compute_reward(self, tls_id):
        lanes     = INTERSECTION_CONFIGS[tls_id]["lanes_in"]
        valid     = self._get_valid_lanes()

        total_wait  = sum(traci.lane.getWaitingTime(l)
                         for l in lanes if l in valid)
        total_queue = sum(traci.lane.getLastStepHaltingNumber(l)
                         for l in lanes if l in valid)

        reward = -0.6 * total_wait - 0.3 * total_queue
        return float(np.clip(reward, -100, 0))