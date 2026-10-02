"""
Shared per-junction control logic for the 8-phase signal design.

This encodes the phase map validated in single_intersection_simulation/map.net.xml:

    Phase 0 -> NS through            (green, ~37s nominal)
    Phase 1 -> NS->EW yellow         (3s, automatic)
    Phase 2 -> protected right turns: N->E (idx3) + S->W (idx11)  (5s green)
    Phase 3 -> protected-turn yellow (2s, automatic)
    Phase 4 -> EW through            (green, ~37s nominal)
    Phase 5 -> EW->NS yellow         (3s, automatic)
    Phase 6 -> protected right turns: E->S (idx7) + W->N (idx15) (5s green)
    Phase 7 -> protected-turn yellow (2s, automatic)

Both single-agent (TrafficEnv) and multi-agent (MultiAgentTrafficEnv) wrap
one IntersectionController per traffic light instead of hand-duplicating
this logic, so a fix here fixes every topology at once.

ACTION SEMANTICS (binary action space):
Action 1 ("switch") always advances to the NEXT phase in the fixed cycle
0 -> 2 -> 4 -> 6 -> 0, inserting the mandatory yellow automatically. The
agent cannot skip a protected-turn phase or jump directly between the two
through-phases — this is "Option A" (strict cycle advance), chosen to keep
the binary action space's meaning unambiguous while supporting >2 greens.
If/when a Discrete(4) direct-phase-selection action space is added for
comparison, it can call advance_to(target_phase) instead of advance().

REWARD WEIGHTING — two independent mechanisms, kept deliberately separate:

1. Ordinary-traffic length weighting: non-emergency vehicles are weighted
   by physical length relative to a car (length already present in the
   .rou.xml vType definitions — no external citation needed). This ties
   the penalty to road-space / clearance-time cost, read live via
   traci.vehicle.getLength().

2. Emergency priority: emergency vehicles are EXCLUDED from the
   length-weighted sum and instead penalized through a separate,
   independently-tunable term (their waiting time x a flat priority
   coefficient). This is deliberate — an emergency vehicle's cost is
   categorically different (life-safety / response-time), not a function
   of its physical size, so conflating the two into one multiplicative
   weight would be incoherent (a bulkier ambulance would otherwise be
   weighted arbitrarily higher than a smaller one for no principled
   reason). Keeping them as separate additive terms lets each be
   motivated and tuned independently, mirroring the original reward
   structure's two-term design.
"""

import traci


# The four GREEN phases in cycle order. Each maps to the yellow phase that
# must play immediately after it, before the next green takes effect.
GREEN_PHASES = [0, 2, 4, 6]
YELLOW_AFTER = {0: 1, 2: 3, 4: 5, 6: 7}

# Nominal/expected durations (seconds) for sanity-checking against the
# actual net.xml tlLogic — used by the validation check, not by control logic.
EXPECTED_GREEN_DURATION = {0: 37, 2: 5, 4: 37, 6: 5}
EXPECTED_YELLOW_DURATION = {1: 3, 3: 2, 5: 3, 7: 2}

# --- Reward weighting constants ---

# Ordinary-traffic length weighting: vehicle length relative to a car,
# used as a proxy for road-space / clearance-time cost. Car = 1.0 reference.
CAR_REFERENCE_LENGTH = 5.0

# Reward term coefficients
WAIT_WEIGHT = 0.6          # coefficient on length-weighted ordinary wait time
QUEUE_WEIGHT = 0.3         # coefficient on raw (unweighted) queue count
EMERGENCY_WEIGHT = 2.0     # coefficient on raw emergency waiting time (separate term)


class IntersectionController:
    """
    Wraps one traffic-light junction: phase switching, queue/state
    reading, and reward computation, parameterized by tls_id and the
    junction's own incoming-lane list.
    """

    def __init__(self, tls_id, lanes_by_direction, min_green=10):
        """
        tls_id: SUMO traffic light id, e.g. "J4"
        lanes_by_direction: dict with keys "north","south","east","west",
            each a list of incoming lane IDs for that approach.
        min_green: minimum seconds a green phase must hold before the
            agent's switch action is allowed to take effect.
        """
        self.tls_id = tls_id
        self.lanes_by_direction = lanes_by_direction
        self.min_green = min_green
        self.phase_time = 0

    def reset(self):
        self.phase_time = 0

    # ---- phase control -------------------------------------------------

    def current_green(self):
        """Returns the current phase if it's a green phase, else None
        (i.e. we're mid-yellow — callers should treat this as 'busy')."""
        phase = traci.trafficlight.getPhase(self.tls_id)
        return phase if phase in GREEN_PHASES else None

    def apply_action(self, action):
        """
        action: 0 = keep current phase, 1 = advance to next phase in cycle.
        Enforces minimum green time. No-ops silently if called mid-yellow
        (yellow phases run to completion automatically via SUMO's own
        tlLogic durations — the agent doesn't control yellow timing).
        """
        self.phase_time += 5  # caller advances sim in 5s steps; kept here
                               # so this class owns its own timer state.

        current_phase = traci.trafficlight.getPhase(self.tls_id)

        if current_phase not in GREEN_PHASES:
            # Mid-transition (yellow) — do nothing, let SUMO finish it.
            return

        if action == 1 and self.phase_time >= self.min_green:
            self.advance()

    def advance(self):
        """Force-advance to the next phase in the fixed green cycle,
        playing the mandatory yellow in between. Resets the green timer."""
        current_phase = traci.trafficlight.getPhase(self.tls_id)
        if current_phase not in GREEN_PHASES:
            return  # already mid-transition; ignore

        yellow = YELLOW_AFTER[current_phase]
        next_index = (GREEN_PHASES.index(current_phase) + 1) % len(GREEN_PHASES)
        next_green = GREEN_PHASES[next_index]

        traci.trafficlight.setPhase(self.tls_id, yellow)
        traci.simulationStep()
        traci.trafficlight.setPhase(self.tls_id, next_green)
        self.phase_time = 0

    def force_green(self, target_green):
        """
        Used by emergency preemption: advance directly toward a specific
        through-phase (0 or 4), stepping through the cycle one hop at a
        time if needed, playing yellows correctly. No-ops if already there
        or if currently mid-yellow (caller should retry next step).
        """
        if target_green not in (0, 4):
            raise ValueError("force_green only supports through-phases 0 or 4")

        current_phase = traci.trafficlight.getPhase(self.tls_id)
        if current_phase == target_green:
            return
        if current_phase not in GREEN_PHASES:
            return  # mid-yellow — caller retries next step

        self.advance()  # one hop toward target; caller calls again if not there yet

    # ---- state / reward --------------------------------------------------

    def get_queues(self):
        """Returns [q_north, q_south, q_east, q_west] halting-vehicle counts."""
        valid = set(traci.lane.getIDList())
        queues = []
        for direction in ("north", "south", "east", "west"):
            lanes = self.lanes_by_direction.get(direction, [])
            q = sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes if l in valid)
            queues.append(q)
        return queues

    def get_normalized_state(self):
        """[q_n, q_s, q_e, q_w, phase] normalised to [0,1]."""
        import numpy as np

        queues = self.get_queues()
        phase = traci.trafficlight.getPhase(self.tls_id)

        state = np.array(queues + [phase], dtype=np.float32)
        state[0:4] /= 50.0
        state[4] /= 7.0  # 8 phases now (0-7), was /3.0 under the old 4-phase scheme
        return np.clip(state, 0.0, 1.0)

    def _length_weight(self, vehicle_id):
        """Length-based weight relative to a car, read live via TraCI so it
        tracks .rou.xml vType lengths without needing a hardcoded table.
        Only used for ordinary (non-emergency) vehicles — see module docstring."""
        return traci.vehicle.getLength(vehicle_id) / CAR_REFERENCE_LENGTH

    def compute_reward(self):
        """
        Three-part reward, mirroring the original structure but with
        ordinary-traffic wait time now length-weighted:

          -WAIT_WEIGHT * (length-weighted ordinary wait time)
          -QUEUE_WEIGHT * (raw queue count, all vehicles)
          -EMERGENCY_WEIGHT * (raw emergency wait time, separate term)

        Emergency vehicles are excluded from the length-weighted sum —
        their cost is handled entirely by the separate emergency term.
        """
        import numpy as np

        valid = set(traci.lane.getIDList())
        all_lanes = [l for lanes in self.lanes_by_direction.values() for l in lanes if l in valid]

        total_weighted_wait = 0.0
        total_queue = 0
        emergency_wait = 0.0

        for lane in all_lanes:
            total_queue += traci.lane.getLastStepHaltingNumber(lane)

            for v in traci.lane.getLastStepVehicleIDs(lane):
                wait = traci.vehicle.getWaitingTime(v)
                if wait <= 0:
                    continue

                if traci.vehicle.getTypeID(v) == "emergency":
                    emergency_wait += wait
                else:
                    total_weighted_wait += wait * self._length_weight(v)

        reward = (
            -WAIT_WEIGHT * total_weighted_wait
            - QUEUE_WEIGHT * total_queue
            - EMERGENCY_WEIGHT * emergency_wait
        )
        return float(np.clip(reward, -100, 0))

    # ---- validation ------------------------------------------------------

    def validate_phase_durations(self, tolerance=1.0):
        """
        Debug helper — call after reset() or a phase switch. Confirms the
        currently-active phase's programmed duration matches what we expect
        for the 8-phase design. Returns (ok: bool, message: str).
        Catches the 'agent silently stuck in a 5s phase thinking it's a
        37s phase' failure mode.
        """
        phase = traci.trafficlight.getPhase(self.tls_id)
        actual = traci.trafficlight.getPhaseDuration(self.tls_id)

        expected = EXPECTED_GREEN_DURATION.get(phase, EXPECTED_YELLOW_DURATION.get(phase))
        if expected is None:
            return False, f"[{self.tls_id}] Unknown phase index {phase} — not in 8-phase map."

        if abs(actual - expected) > tolerance:
            return False, (
                f"[{self.tls_id}] Phase {phase}: expected ~{expected}s, "
                f"got {actual}s. Check tlLogic in net.xml matches the 8-phase design."
            )
        return True, f"[{self.tls_id}] Phase {phase} duration OK (~{actual}s)."