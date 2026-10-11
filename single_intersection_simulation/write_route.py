import xml.etree.ElementTree as ET
import numpy as np


def write_route_file(rates, input_path, output_path, seed=None):
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Apply the perturbed rates to the standard traffic flows
    for flow in root.findall("flow"):
        fid = flow.get("id")
        if fid in rates:
            flow.set("perHour", str(rates[fid]))

    # Dynamically inject random emergency vehicles if a seed is provided
    if seed is not None:
        rng = np.random.default_rng(seed + 999)  # offset so RNG state differs from flow generation

        # Spawn between 1 and 3 emergency vehicles per hour-long episode
        num_emergencies = int(rng.integers(1, 4))
        possible_routes = ["N_to_S", "S_to_N", "E_to_W", "W_to_E"]

        # Pull out the vehicles already in the base file (e.g. emg_test) so
        # that ALL vehicles can be re-inserted in departure order. SUMO ignores
        # any vehicle that is out of order in the file.
        vehicles = root.findall("vehicle")
        for v in vehicles:
            root.remove(v)

        for i in range(num_emergencies):
            depart_time = round(float(rng.uniform(50.0, 3500.0)), 2)
            route = str(rng.choice(possible_routes))
            vehicles.append(ET.Element("vehicle", {
                "id": f"emg_{seed}_{i}",
                "type": "emergency",
                "route": route,
                "depart": str(depart_time),
            }))

        vehicles.sort(key=lambda e: float(e.get("depart")))

        ids = [v.get("id") for v in vehicles]
        assert len(ids) == len(set(ids)), f"Duplicate vehicle ids: {ids}"

        for v in vehicles:
            root.append(v)

    tree.write(output_path, xml_declaration=True, encoding="UTF-8")