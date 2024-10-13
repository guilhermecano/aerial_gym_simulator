import os
import random
import json
import time
from omegaconf import OmegaConf
import urdfpy
from glob import glob


class DroneConfigManager:
    def __init__(self, urdf_dir, base_config_path, batch_size):
        self.urdf_dir = urdf_dir
        self.base_config = OmegaConf.load(base_config_path)
        self.batch_size = batch_size
        self.urdf_files = self._index_urdf_files()
        self.current_seed = None
        self.current_batch = None

    def _index_urdf_files(self):
        return list(glob(os.path.join(self.urdf_dir, '*.urdf')))
        # return [f for f in os.listdir(self.urdf_dir) if f.endswith('.urdf')]

    def _generate_drone_config(self, urdf_file):
        drone_config = OmegaConf.create(self.base_config)
        drone_config.robot_asset.file = os.path.join(self.urdf_dir, urdf_file)
        drone_config.robot_asset.name = os.path.splitext(urdf_file)[0]
        drone_config.action_space_size = self._determine_action_space_size(urdf_file)
        return drone_config

    def _determine_action_space_size(self, urdf_file, drone_config=None):
        if drone_config is None:
            # Load the URDF file
            urdf_path = os.path.join(self.urdf_dir, urdf_file)
            drone = urdfpy.URDF.load(urdf_path)
        else:
            drone = drone_config
        return len([j for j in drone.joint_map.keys() if "propeller_joint" in j])

    def get_random_batch(self, seed=None):
        if seed is None:
            seed = int(time.time() * 1000)
        self.current_seed = seed
        random.seed(seed)
        selected_urdfs = random.choices(self.urdf_files, k=self.batch_size)
        self.current_batch = [self._generate_drone_config(urdf) for urdf in selected_urdfs]
        return self.current_batch

    def get_current_seed(self):
        return self.current_seed

    def save_experiment_state(self, experiment_id):
        state = {
            "experiment_id": experiment_id,
            "seed": self.current_seed,
            "batch": [cfg.robot_asset.file for cfg in self.current_batch] if self.current_batch else None
        }
        with open(f"experiment_state_{experiment_id}.json", "w") as f:
            json.dump(state, f)

    def load_experiment_state(self, experiment_id):
        try:
            with open(f"experiment_state_{experiment_id}.json", "r") as f:
                state = json.load(f)
            self.current_seed = state["seed"]
            if state["batch"]:
                self.current_batch = [self._generate_drone_config(os.path.basename(file)) for file in state["batch"]]
            return True
        except FileNotFoundError:
            return False

# Example usage
if __name__ == "__main__":
    base_config_path = "/home/guilherme/phd/aerial_gym_simulator/aerial_gym/envs/base/modular/base_config.yml"
    urdf_dir = '/home/guilherme/phd/aerial_gym_simulator/resources/robots/modular/training'
    config_manager = DroneConfigManager(urdf_dir, base_config_path, batch_size=300)

    # Get a random batch of drone configs
    drone_configs = config_manager.get_random_batch()
    print(f"Generated {len(drone_configs)} drone configurations")
    print(f"Current seed: {config_manager.get_current_seed()}")

    # Save experiment state
    config_manager.save_experiment_state("test_experiment")

    # Load experiment state
    if config_manager.load_experiment_state("test_experiment"):
        print("Successfully loaded experiment state")
    else:
        print("Failed to load experiment state")
