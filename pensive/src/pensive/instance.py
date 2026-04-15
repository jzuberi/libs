import os
import json
from typing import Any

class InstanceManager:
    """Handles creating and loading Pensive instances."""

    def __init__(self, instance_path: str):
        self.instance_path = instance_path
        self.config_file = os.path.join(instance_path, "config.json")
        self.strategies_file = os.path.join(instance_path, "strategies.json")
        self.criteria_file = os.path.join(instance_path, "criteria.json")
        self.corpus_model_file = os.path.join(instance_path, "corpus_model.json")
        self.save_config_file = os.path.join(instance_path, "save_config.json")

    # -----------------------------
    # Create a new instance folder
    # -----------------------------
    def init_instance(self, config: dict):
        os.makedirs(self.instance_path, exist_ok=True)

        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

        os.makedirs(os.path.join(self.instance_path, "decisions"), exist_ok=True)
        os.makedirs(os.path.join(self.instance_path, "turns"), exist_ok=True)


    # -----------------------------
    # Load an existing instance
    # -----------------------------
    def load_instance(self) -> dict:
        with open(self.config_file, "r") as f:
            return json.load(f)

