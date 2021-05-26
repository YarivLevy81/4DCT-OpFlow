import json


MAIN_JSON      = 'configs/server_deb.json'
SECONDARY_JSON = 'configs/server_admm.json'
TARGET_JSON    = 'configs/server_admm_v2.json'

def merge_configs(main=MAIN_JSON, secondary=SECONDARY_JSON, target=TARGET_JSON):
    with open(main, 'r') as f:
        main_data = json.load(f)

    with open(secondary, 'r') as f:
        secondary_data = json.load(f)
    
    merged_data = {}

    for entry in main_data:
        merged_data[entry] = main_data[entry]
    
    for entry in secondary_data:
        if entry not in merged_data:
            merged_data[entry] = secondary_data[entry]
    
    with open(target, 'w') as f:
        json.dump(merged_data, f, indent=4)

merge_configs()

