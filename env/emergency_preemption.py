import traci


class EmergencyPreemption:

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

                # Check if vehicle is close to intersection
                lane_length = traci.lane.getLength(lane_id)
                dist_to_junction = lane_length - distance

                if dist_to_junction < self.detection_distance:
                    self.current_vehicle = v
                    self.active = True
                    return True

        self.active = False
        self.current_vehicle = None
        return False

    # Edge IDs of lanes approaching this intersection from each axis.
    # Update these to match the actual edge IDs in your .net.xml file.
    INCOMING_NS = {"north_in", "south_in"}
    INCOMING_EW = {"east_in", "west_in"}

    def get_vehicle_direction(self):

        if not self.current_vehicle:
            return None

        # Use edge ID (road), not lane ID — avoids fragile string matching
        # on lane suffixes like _0, _1 that vary by network
        edge = traci.vehicle.getRoadID(self.current_vehicle)

        if edge in self.INCOMING_NS:
            return "NS"
        elif edge in self.INCOMING_EW:
            return "EW"

        return None

    def apply_preemption(self):

        direction = self.get_vehicle_direction()

        if direction is None:
            return

        current_phase = traci.trafficlight.getPhase(self.tls_id)

        # Phase 0 = NS green
        # Phase 2 = EW green
        # Yellow is set here but NOT stepped — the environment's own
        # step loop advances the clock, keeping step_count in sync.

        if direction == "NS":
            if current_phase not in (0, 1):   # not already NS green or going there
                traci.trafficlight.setPhase(self.tls_id, 1)  # yellow first

        elif direction == "EW":
            if current_phase not in (2, 3):   # not already EW green or going there
                traci.trafficlight.setPhase(self.tls_id, 3)  # yellow first

    def should_hold_green(self):

        if not self.current_vehicle:
            return False

        # If vehicle still exists → hold green
        return self.current_vehicle in traci.vehicle.getIDList()