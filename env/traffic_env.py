import os
import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces

from env.emergency_preemption import EmergencyPreemption
from env.intersection_controller import IntersectionController, YELLOW_AFTER, SWITCH_PENALTY


class TrafficEnv(gym.Env):
    """
    Single-intersection environment for the 8-phase signal design
    (single_intersection_simulation/map.net.xml).

    Action space is binary (keep/switch). A switch is only applied when the
    controller can act on it (min-green elapsed, in a through phase). Requests
    that cannot take effect are masked to keep and counted separately.

    Counters, reset every episode:
      requested_switches  policy asked to switch
      masked_switches     request turned into keep because a switch was not allowed
      executed_switches   controller actually moved from a through phase to its yellow
    """

    def __init__(self, sumo_cfg, max_steps=3600, use_gui=False, delay=0, max_green=90):
        super().__init__()

        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.step_count = 0
        self.arrived_total = 0
        self.requested_switches = 0
        self.masked_switches = 0
        self.executed_switches = 0
        self.label = f"env_{os.getpid()}_{id(self)}"
        self._started = False

        self.tls_id = "J4"  # traffic light id, matches map.net.xml

        # Observation: [queue_n, queue_s, queue_e, queue_w, phase, time_in_phase],
        # all normalised to [0, 1] by IntersectionController.get_normalized_state()
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

        # 0 = keep current through phase, 1 = switch (subject to min-green)
        self.action_space = spaces.Discrete(2)

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
            max_green=max_green,
        )

        self.emergency = EmergencyPreemption(tls_id=self.tls_id)

        binary = "sumo-gui" if use_gui else "sumo"
        self.sumo_cmd = [
            binary,
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--no-step-log", "true",
        ]
        if use_gui and delay > 0:
            self.sumo_cmd += ["--delay", str(delay)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        traci.start(self.sumo_cmd, label=self.label)
        self._started = True

        self.step_count = 0
        self.arrived_total = 0
        self.requested_switches = 0
        self.masked_switches = 0
        self.executed_switches = 0
        self.controller.reset()

        observation = self.controller.get_normalized_state()
        return observation, {}

    def step(self, action):
        traci.switch(self.label)

        # 1. Emergency preemption. While a vehicle is tracked, it can force the
        #    signal toward its through phase and hold it until it leaves the junction.
        hold = False
        if self.emergency.check_for_emergency():
            self.emergency.apply_preemption(self.controller)
            if self.emergency.should_hold_green():
                action = 0   # the agent cannot switch away during the hold
                hold = True  # the hold overrides the max-green cap

        # 2. Record what the policy asked for, before any masking.
        if action == 1:
            self.requested_switches += 1

        # 3. Mask switches that the controller cannot act on. This uses the same
        #    min-green rule as apply_action, via can_switch().
        if action == 1 and not self.controller.can_switch():
            action = 0
            self.masked_switches += 1

        # 4. Apply the action, and count an executed switch when the controller
        #    moves from a through phase to its yellow.
        phase_before = traci.trafficlight.getPhase(self.tls_id)
        self.controller.apply_action(action, hold=hold)
        phase_after = traci.trafficlight.getPhase(self.tls_id)

        executed_now = False
        if phase_before in YELLOW_AFTER and phase_after == YELLOW_AFTER[phase_before]:
            self.executed_switches += 1
            executed_now = True

        # 5. Advance the simulation by one 5 s decision interval.
        for _ in range(5):
            traci.simulationStep()
            self.arrived_total += traci.simulation.getArrivedNumber()

        self.step_count += 5

        state = self.controller.get_normalized_state()
        reward = self.controller.compute_reward()
        if executed_now:
            reward -= SWITCH_PENALTY

        terminated = self.step_count >= self.max_steps
        truncated = False

        return state, reward, terminated, truncated, {}

    def close(self):
        if self._started:
            try:
                traci.switch(self.label)
                traci.close()
            except Exception:
                pass
            finally:
                self._started = False