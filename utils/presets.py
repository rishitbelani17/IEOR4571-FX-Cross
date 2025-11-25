import json
import os

PRESET_FILE = "presets.json"

def load_presets():
    if not os.path.exists(PRESET_FILE):
        return {}
    with open(PRESET_FILE, "r") as f:
        return json.load(f)

def save_presets(data):
    with open(PRESET_FILE, "w") as f:
        json.dump(data, f, indent=4)