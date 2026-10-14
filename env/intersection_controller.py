"""
Shared per-junction control logic for the 8-phase signal design.

This encodes the phase map in single_intersection_simulation/map.net.xml:

    Phase 0 -> NS through            (agent-controlled: held until "switch")
    Phase 1 -> NS->EW yellow         (3s, automatic)
    Phase 2 -> protected right turns: N->E (idx3) + S->W (idx11)  (5s, automatic)
    Phase 3 -> protected-turn yellow (2s, automatic)
    Phase 4 -> EW through            (agent-controlled: held until "switch")
    Phase 5 -> EW->NS yellow         (3s, automatic)
    Phase 6 -> protected right turns: E->S (idx7) + W->N (idx15) (5s, automatic)
    Phase 7 -> protected-turn yellow (2s, automatic)

Both single-agent (TrafficEnv) and multi-agent (MultiAgentTrafficEnv) wrap
one IntersectionController per traffic light instead of hand-duplicating
this logic, so a fix here fixes every topology at once.

PHASE TIMING OWNERSHIP:
The through phases (0 and 4) have a very long programmed duration in the
tlLogic (3600s), so SUMO never ends them on its own: "keep" really holds.
Every other phase (yellows and protected-turn greens) runs on its fixed
programmed timer and SUMO advances through them in index order. The
controller therefore only ever calls setPhase() on a through phase, to
jump to its yellow; SUMO then plays the full yellow, the protected turn
and the next through phase automatically. Minimum green is measured with
SUMO's own getSpentDuration(), so there is no separate timer to drift.

ACTION SEMANTICS (binary action space):
Action 1 ("switch") ends the current through phase. The fixed sequence
that follows is: yellow -> protected right turns -> yellow -> opposite
through phase (0 -> 1 -> 2 -> 3 -> 4 and 4 -> 5 -> 6 -> 7 -> 0). The
agent cannot skip a protected-turn phase or shorten a yellow. Action 1 is
ignored during yellow / protected-turn phases and before min_green has
elapsed on the current through phase. If/when a Discrete(4) direct-phase
selection action space is added for comparison, it can reuse advance().

REWARD WEIGHTING:
Ordinary vehicles' waiting time is weighted by physical length relative
to a car (length is already in the .rou.xml vType definitions), read live
via traci.vehicle.getLength(). This ties the penalty to road-space /
clearance-time cost. Emergency vehicles are excluded from that sum and
penalised through a separate, length-independent term with its own
coefficient, since their priority comes from being an emergency vehicle,
not from their size.
"""

import numpy as np
import traci


# Through phases the agent controls, and the yellow that follows each one.
# After the yellow, SUMO auto-advances (protected turn -> yellow -> next
# through phase) using the durations programmed in the tlLogic.
CONTROLLED_PHASES = [0, 4]
YELLOW_AFTER = {0: 1, 4: 5}

# Expected programmed durations (seconds) per phase, for validating against
# the actual net.xml tlLogic. Through phases are "held" (very long).
EXPECTED_DURATION = {0: 3600, 1: 3, 2: 5, 3: 2, 4: 3600, 5: 3, 6: 5, 7: 2}

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

    def __init__(self, tls_id, lanes_by_direction, min_green=10, max_green=90):
        """
        tls_id: SUMO traffic light id, e.g. "J4"
        lanes_by_direction: dict with keys "north","south","east","west",
            each a list of incoming lane IDs for that approach.
        min_green: minimum seconds a through phase must have been active
            before the agent's switch action is allowed to take effect.
        """
        self.tls_id = tls_id
        self.lanes_by_direction = lanes_by_direction
        self.min_green = min_green
        self.max_green = max_green

    def reset(self):
        # No controller-side timers: phase timing is read from SUMO.
        pass

    # ---- phase control -------------------------------------------------

    def current_controlled_phase(self):
        """Returns the current phase if it's an agent-controlled through
        phase, else None (we're in a yellow / protected-turn phase that
        SUMO is running on its own timer)."""
        phase = traci.trafficlight.getPhase(self.tls_id)
        return phase if phase in CONTROLLED_PHASES else None

    def apply_action(self, action, hold=False):
        """
        action: 0 = keep current phase, 1 = end the current through phase.
        No-ops silently during yellow / protected-turn phases (SUMO runs
        those to completion) and until min_green has elapsed.
        """
        phase = traci.trafficlight.getPhase(self.tls_id)
        if phase not in CONTROLLED_PHASES:
            return

        spent = traci.trafficlight.getSpentDuration(self.tls_id)

        if not hold and spent >= self.max_green:
            action = 1

        if action == 1 and spent >= self.min_green:
            self.advance()

    def advance(self):
        """End the current through phase by jumping to its yellow. SUMO then
        plays the full yellow, the protected-turn phase and the next
        through phase on its own. Ignored if not in a through phase."""
        phase = traci.trafficlight.getPhase(self.tls_id)
        if phase not in YELLOW_AFTER:
            return
        traci.trafficlight.setPhase(self.tls_id, YELLOW_AFTER[phase])

    def force_green(self, target_green):
        """
        Used by emergency preemption: head toward a specific through phase
        (0 or 4). No-ops if already there, or if mid-transition (SUMO is
        already moving to the next through phase; caller retries next step).
        Bypasses min_green by design.
        """
        if target_green not in (0, 4):
            raise ValueError("force_green only supports through-phases 0 or 4")

        phase = traci.trafficlight.getPhase(self.tls_id)
        if phase == target_green or phase not in CONTROLLED_PHASES:
            return

        self.advance()

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
        queues = self.get_queues()
        phase = traci.trafficlight.getPhase(self.tls_id)

        state = np.array(queues + [phase], dtype=np.float32)
        state[0:4] /= 50.0
        state[4] /= 7.0  # phases 0-7
        return np.clip(state, 0.0, 1.0)

    def _length_weight(self, vehicle_id):
        """Length-based weight relative to a car, read live via TraCI so it
        tracks .rou.xml vType lengths without needing a hardcoded table.
        Only used for ordinary (non-emergency) vehicles."""
        return traci.vehicle.getLength(vehicle_id) / CAR_REFERENCE_LENGTH

    def compute_reward(self):
        """
        Three-part reward:

          -WAIT_WEIGHT * (length-weighted ordinary wait time)
          -QUEUE_WEIGHT * (raw queue count, all vehicles)
          -EMERGENCY_WEIGHT * (raw emergency wait time, separate term)

        Emergency vehicles are excluded from the length-weighted sum —
        their cost is handled entirely by the separate emergency term.
        """
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
        return float(np.clip(reward, -5000, 0))

    # ---- validation ------------------------------------------------------

    def validate_phase_durations(self, tolerance=1.0):
        """
        Debug helper — call after reset() or a phase change. Confirms the
        currently-active phase's programmed duration matches the 8-phase
        design (through phases 3600s, protected turns 5s, yellows 3s/2s).
        Returns (ok: bool, message: str). Catches a net.xml tlLogic that
        doesn't match what the controller assumes, e.g. through phases
        that still time out on their own.
        """
        phase = traci.trafficlight.getPhase(self.tls_id)
        actual = traci.trafficlight.getPhaseDuration(self.tls_id)

        expected = EXPECTED_DURATION.get(phase)
        if expected is None:
            return False, f"[{self.tls_id}] Unknown phase index {phase} — not in 8-phase map."

        if abs(actual - expected) > tolerance:
            return False, (
                f"[{self.tls_id}] Phase {phase}: expected ~{expected}s, "
                f"got {actual}s. Check tlLogic in net.xml matches the 8-phase design."
            )
        return True, f"[{self.tls_id}] Phase {phase} duration OK (~{actual}s)."