import xml.etree.ElementTree as ET

def write_route_file(rates, input_path, output_path):
    tree = ET.parse(input_path)
    root = tree.getroot()

    for flow in root.findall("flow"):
        fid = flow.get("id")
        if fid in rates:
            flow.set("perHour", str(rates[fid]))

    tree.write(output_path, xml_declaration=True, encoding="UTF-8")


# Wire this into training:
# for seed in [0, 1, 2, 3, 4]:
#     rates = generate_route_rates(seed, baseline_rate=500.0, noise_pct=0.15)
#     write_route_file(rates, "traffic.rou.xml", f"traffic_seed{seed}.rou.xml")

     # point your .sumocfg (or the sumo_cmd list in TrafficEnv) at traffic_seed{seed}.rou.xml
#     env = TrafficEnv(sumo_cfg=f"configs/simulation_seed{seed}.sumocfg", ...)
    # train...
