import traci


class EmergencyPreemption:
    """
    Detects an approaching emergency vehicle and forces the signal toward
    the through-phase (0 = NS, 4 = EW) that matches its direction, using
    the shared IntersectionController to do the actual phase advancing
    (so yellow transitions are always respected — no hard phase jumps).

    Edge IDs below match single_intersection_simulation/map.net.xml.
    """

    def __init__(self, tls_id="J4", detection_distance=100):
        self.tls_id = tls_id
        self.detection_distance = detection_distance
        self.active = False
        self.current_vehicle = None

    def check_for_emergency(self):
        vehicle_ids = traci.vehicle.getIDList()

        for v in vehicle_ids:
            if traci.vehicle.getTypeID(v) == "emergency":
                lane_id = traci.vehicle.getLaneID(v)
                distance = traci.vehicle.getLanePosition(v)
                lane_length = traci.lane.getLength(lane_id)
                dist_to_junction = lane_length - distance

                if dist_to_junction < self.detection_distance:
                    self.current_vehicle = v
                    self.active = True
                    return True

        self.active = False
        self.current_vehicle = None
        return False

    # Edge IDs of lanes approaching J4 from each axis, from
    # single_intersection_simulation/map.net.xml.
    INCOMING_NS = {"north_in", "south_in"}
    INCOMING_EW = {"east_in", "west_in"}

    def get_vehicle_direction(self):
        if not self.current_vehicle:
            return None

        edge = traci.vehicle.getRoadID(self.current_vehicle)

        if edge in self.INCOMING_NS:
            return "NS"
        elif edge in self.INCOMING_EW:
            return "EW"

        return None

    def apply_preemption(self, controller):
        """
        controller: the IntersectionController for this tls_id.

        Forces the signal toward phase 0 (NS through) or phase 4 (EW
        through) depending on the emergency vehicle's approach direction.
        If the signal is currently in the *other* through-phase or in a
        protected-turn phase, this advances one hop per call — since this
        is called every environment step while the vehicle is detected,
        it will reach the target phase within a few steps rather than in
        one jump, always playing the correct yellow in between.

        If currently mid-yellow, this is a no-op for this step; it will
        resume forcing on the next call once SUMO completes the yellow.
        """
        direction = self.get_vehicle_direction()
        if direction is None:
            return

        target_green = 0 if direction == "NS" else 4
        controller.force_green(target_green)

    def should_hold_green(self):
        if not self.current_vehicle:
            return False
        return self.current_vehicle in traci.vehicle.getIDList()