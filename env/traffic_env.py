import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces

from env.emergency_preemption import EmergencyPreemption
from env.intersection_controller import IntersectionController


class TrafficEnv(gym.Env):
    """
    Single-intersection environment for the 8-phase signal design
    (single_intersection_simulation/map.net.xml).

    Action space is binary (keep/switch); "switch" advances the fixed
    green cycle 0 -> 2 -> 4 -> 6 -> 0, inserting the mandatory yellow
    automatically. See intersection_controller.py for the full phase map
    and the rationale for this "strict cycle advance" behavior.
    """

    def __init__(self, sumo_cfg, max_steps=3600, use_gui=False, delay=0):
        super().__init__()

        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.step_count = 0

        self.tls_id = "J4"  # traffic light id — matches map.net.xml

        # Observation space
        # [queue_n, queue_s, queue_e, queue_w, phase]
        # All values normalised to [0, 1] — bounds match IntersectionController.get_normalized_state()
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        )

        # Actions
        # 0 = keep phase
        # 1 = switch phase (advance to next phase in the 4-green cycle)
        self.action_space = spaces.Discrete(2)

        # Lane config for J4 in single_intersection_simulation/map.net.xml
        lanes_by_direction = {
            "north": ["north_in_0", "north_in_1"],
            "south": ["south_in_0", "south_in_1"],
            "east": ["east_in_0", "east_in_1"],
            "west": ["west_in_0", "west_in_1"],
        }

        self.controller = IntersectionController(
            tls_id=self.tls_id,
            lanes_by_direction=lanes_by_direction,
            min_green=10,
        )

        # Emergency vehicle preemption module
        self.emergency = EmergencyPreemption(tls_id=self.tls_id)

        binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_cmd = [binary, "-c", self.sumo_cfg, "--start", "--quit-on-end"]
        if use_gui and delay > 0:
            self.sumo_cmd += ["--delay", str(delay)]

    def reset(self, seed=None, options=None):
        if traci.isLoaded():
            traci.close()

        traci.start(self.sumo_cmd)

        self.step_count = 0
        self.controller.reset()

        observation = self.controller.get_normalized_state()
        return observation, {}

    def step(self, action):
        # Emergency preemption overrides the agent when a vehicle is
        # approaching; holds/forces green toward the vehicle's direction
        # until it has cleared.
        if self.emergency.check_for_emergency():
            self.emergency.apply_preemption(self.controller)
            if self.emergency.should_hold_green():
                action = 0  # prevent agent from switching away

        # min-green enforcement lives inside the controller now
        self.controller.apply_action(action)

        for _ in range(5):  # 5 sec decision interval
            traci.simulationStep()

        self.step_count += 5

        state = self.controller.get_normalized_state()
        reward = self.controller.compute_reward()

        terminated = self.step_count >= self.max_steps
        truncated = False

        return state, reward, terminated, truncated, {}

    def close(self):
        if traci.isLoaded():
            traci.close()