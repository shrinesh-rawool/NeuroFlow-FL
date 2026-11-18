import traci


# Approach edges into J4, from single_intersection_simulation/map.net.xml.
APPROACH_EDGES = {"north_in", "south_in", "east_in", "west_in"}
INCOMING_NS = {"north_in", "south_in"}
INCOMING_EW = {"east_in", "west_in"}


class EmergencyPreemption:
    """
    Detects an approaching emergency vehicle and forces the signal toward
    the through-phase (0 = NS, 4 = EW) that matches its direction, using
    the shared IntersectionController to do the actual phase advancing
    (so yellow transitions are always respected, no hard phase jumps).

    The hold is active while the vehicle is on an approach edge or inside
    the junction, and ends when it moves onto an outgoing edge.

    Edge IDs match single_intersection_simulation/map.net.xml.
    """

    def __init__(self, tls_id="J4", detection_distance=100):
        self.tls_id = tls_id
        self.detection_distance = detection_distance
        self.active = False
        self.current_vehicle = None
        self.current_direction = None

    def get_vehicle_direction(self):
        """Direction stored at detection time, so it stays valid after the
        vehicle enters the junction."""
        return self.current_direction

    def apply_preemption(self, controller):
        """
        Forces the signal toward phase 0 (NS through) or phase 4 (EW through)
        depending on the emergency vehicle's direction. force_green() respects
        yellow transitions, so this is a no-op while mid-yellow and resumes on
        the next call.
        """
        direction = self.get_vehicle_direction()
        if direction is None:
            return

        target_green = 0 if direction == "NS" else 4
        controller.force_green(target_green)

    def _release_if_gone(self):
        """Clears the tracked vehicle once it has left the network."""
        if self.current_vehicle and self.current_vehicle not in traci.vehicle.getIDList():
            self.current_vehicle = None
            self.current_direction = None
            self.active = False

    def check_for_emergency(self):
        """
        Keeps tracking the current emergency vehicle until it leaves the
        network. If nothing is tracked yet, looks for a new emergency vehicle
        on an approach edge within detection_distance of the stop line.
        Returns True while a vehicle is being tracked.
        """
        self._release_if_gone()
        if self.current_vehicle:
            return True

        for v in traci.vehicle.getIDList():
            if traci.vehicle.getTypeID(v) != "emergency":
                continue

            road = traci.vehicle.getRoadID(v)
            if road not in APPROACH_EDGES:
                continue

            lane_id = traci.vehicle.getLaneID(v)
            dist_to_stop_line = (
                traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(v)
            )
            if dist_to_stop_line < self.detection_distance:
                self.current_vehicle = v
                self.current_direction = "NS" if road in INCOMING_NS else "EW"
                self.active = True
                return True

        return False

    def should_hold_green(self):
        """
        True while the tracked emergency vehicle is on an approach edge or
        inside the junction. False once it is on an outgoing edge or has left
        the network.
        """
        self._release_if_gone()
        if not self.current_vehicle:
            return False

        road = traci.vehicle.getRoadID(self.current_vehicle)
        return road in APPROACH_EDGES or road.startswith(":J4_")
