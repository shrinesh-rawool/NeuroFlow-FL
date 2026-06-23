import os
import sys
import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces
from env.emergency_preemption import EmergencyPreemption


class TrafficEnv(gym.Env):

    def __init__(self, sumo_cfg, max_steps=3600, use_gui=False, delay=0):

        super(TrafficEnv, self).__init__()      # Initialize parent class (gym.Env) before initializing TrafficEnv

        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.step_count = 0

        self.tls_id = "J4"  # traffic light id

        # Observation space
        # [queue_n, queue_s, queue_e, queue_w, phase]
        # All values normalised to [0, 1] — bounds match actual output of _get_state()
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32
        )

        # Actions
        # 0 = keep phase
        # 1 = switch phase
        self.action_space = spaces.Discrete(2)

        # Minimum green time enforcement (seconds)
        # Prevents the agent from flickering the signal every 5 s
        self.min_green = 10
        self.phase_time = 0   # tracks how long current phase has been active

        # Emergency vehicle preemption module
        self.emergency = EmergencyPreemption(tls_id=self.tls_id)

        # Change __init__ signature:

        binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_cmd = [
            binary,
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end"
        ]
        if use_gui and delay > 0:
            self.sumo_cmd += ["--delay", str(delay)]


    def reset(self, seed=None, options=None):

        if traci.isLoaded():
            traci.close()

        traci.start(self.sumo_cmd)

        self.step_count = 0
        self.phase_time = 0   # reset minimum green timer on each new episode

        observation = self._get_state()

        return observation, {}


    def step(self, action):

        # Enforce minimum green time — block switch if phase too young
        self.phase_time += 5
        if self.phase_time < self.min_green:
            action = 0  # force keep, ignore agent's switch request

        # Emergency preemption overrides agent action when an emergency
        # vehicle is approaching; holds green until vehicle has cleared
        if self.emergency.check_for_emergency():
            self.emergency.apply_preemption()
            if self.emergency.should_hold_green():
                action = 0  # prevent agent from switching away

        self._apply_action(action)

        for _ in range(5):  # 5 sec decision interval
            traci.simulationStep()

        self.step_count += 5

        state = self._get_state()   # already normalised inside _get_state()
        reward = self._compute_reward()

        terminated = self.step_count >= self.max_steps
        truncated = False

        return state, reward, terminated, truncated, {}


    def close(self):
        if traci.isLoaded():
            traci.close()


    def _get_state(self):

        lanes = {
            "north": ["north_in_0", "north_in_1"],
            "south": ["south_in_0", "south_in_1"],
            "east":  ["east_in_0",  "east_in_1"],
            "west":  ["west_in_0",  "west_in_1"],
        }

        queues = []

        for direction in lanes:
            q = 0
            for lane in lanes[direction]:
                q += traci.lane.getLastStepHaltingNumber(lane)
            queues.append(q)

        phase = traci.trafficlight.getPhase(self.tls_id)

        state = np.array(queues + [phase], dtype=np.float32)

        # Normalise to [0, 1] — must match observation_space bounds
        state[0:4] /= 50.0   # queues: assume max ~50 vehicles per direction
        state[4]   /= 3.0    # phase: 0–3 → 0.0–1.0

        return state


    def _apply_action(self, action):

        if action == 1:
            current_phase = traci.trafficlight.getPhase(self.tls_id)

            # Skip if already in a yellow/transitional phase
            if current_phase in (1, 3):
                return

            if current_phase == 0:
                traci.trafficlight.setPhase(self.tls_id, 1)  # yellow
                traci.simulationStep()
                traci.trafficlight.setPhase(self.tls_id, 2)
                self.phase_time = 0  # reset green timer after switch

            elif current_phase == 2:
                traci.trafficlight.setPhase(self.tls_id, 3)  # yellow
                traci.simulationStep()
                traci.trafficlight.setPhase(self.tls_id, 0)
                self.phase_time = 0  # reset green timer after switch


    def _compute_reward(self):

        lanes = [
            "north_in_0", "north_in_1",
            "south_in_0", "south_in_1",
            "east_in_0", "east_in_1",
            "west_in_0", "west_in_1"
        ]

        total_wait = 0
        total_queue = 0
        emergency_penalty = 0

        for lane in lanes:

            total_wait += traci.lane.getWaitingTime(lane)
            total_queue += traci.lane.getLastStepHaltingNumber(lane)

            vehicles = traci.lane.getLastStepVehicleIDs(lane)

            for v in vehicles:
                if traci.vehicle.getTypeID(v) == "emergency":
                    emergency_penalty += traci.vehicle.getWaitingTime(v)

        reward = (
            -0.6 * total_wait
            -0.3 * total_queue
            -2.0 * emergency_penalty
        )

        reward = np.clip(reward, -100, 0)

        return reward