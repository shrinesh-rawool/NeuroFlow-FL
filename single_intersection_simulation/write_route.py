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
        rng = np.random.default_rng(seed + 999) # Offset to ensure different RNG state from flow generation
        
        # Spawn between 1 and 3 emergency vehicles per hour-long episode
        num_emergencies = rng.integers(1, 4)
        
        # Standard through-routes for emergency vehicles
        possible_routes = ["N_to_S", "S_to_N", "E_to_W", "W_to_E"]
        
        for i in range(num_emergencies):
            # Random depart time between 50s and 3500s
            depart_time = round(rng.uniform(50.0, 3500.0), 2)
            route = rng.choice(possible_routes)
            
            emg_veh = ET.Element("vehicle", {
                "id": f"emg_{seed}_{i}",
                "type": "emergency",
                "route": route,
                "depart": str(depart_time)
            })
            root.append(emg_veh)

    tree.write(output_path, xml_declaration=True, encoding="UTF-8")