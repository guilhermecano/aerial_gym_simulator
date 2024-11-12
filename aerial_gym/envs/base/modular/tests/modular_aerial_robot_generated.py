import numpy as np
from isaacgym import gymapi
from aerial_gym.envs.base.base_task import BaseTask
from omegaconf import OmegaConf
import os
import torch


class ModularAerialRobot(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, drone_configs):
        self.cfg = cfg
        self.drone_configs = drone_configs
        self.num_drones = len(drone_configs)
        
        # Ensure num_envs is divisible by num_drones
        assert cfg.env.num_envs % self.num_drones == 0, "num_envs must be divisible by the number of drone configs"
        self.envs_per_drone = cfg.env.num_envs // self.num_drones

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.gym.prepare_sim(self.sim)
        self._allocate_buffers()

    def create_sim(self):
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs()

    def _create_envs(self):
        self.actor_handles = []
        self.envs = []

        for drone_idx, drone_cfg in enumerate(self.drone_configs):
            asset_path = drone_cfg.robot_asset.file
            asset_root = os.path.dirname(asset_path)
            asset_file = os.path.basename(asset_path)

            asset_options = gymapi.AssetOptions()
            # Set asset options based on drone_cfg

            robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

            for env_idx in range(self.envs_per_drone):
                global_env_idx = drone_idx * self.envs_per_drone + env_idx

                # Create environment
                env_ptr = self.gym.create_env(self.sim, self.env_lower, self.env_upper, int(np.sqrt(self.num_envs)))

                # Create actor
                pose = gymapi.Transform()
                pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # Start pose, adjust as needed
                actor_handle = self.gym.create_actor(env_ptr, robot_asset, pose, f"drone_{global_env_idx}", global_env_idx, 1)

                self.actor_handles.append(actor_handle)
                self.envs.append(env_ptr)

        self.robot_body_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])

    def _allocate_buffers(self):
        # Allocate buffers for observations, rewards, etc.
        # Make sure to account for potentially different observation spaces across drones
        max_obs_size = max(cfg.action_space_size for cfg in self.drone_configs)
        
        self.obs_buf = torch.zeros((self.num_envs, max_obs_size), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def step(self, actions):
        # Implement the step function
        # You may need to handle different action spaces for different drones
        # ...

    def reset_idx(self, env_ids):
        # Implement reset logic
        # Consider that different env_ids might correspond to different drone types
        # ...

    def compute_observations(self):
        # Compute observations
        # Handle potentially different observation spaces for different drones
        # ...

    def compute_reward(self):
        # Compute rewards
        # You might want to use different reward functions for different drone types
        # ...

# Usage example
if __name__ == "__main__":
    from drone_config_manager import DroneConfigManager

    base_config_path = 'path/to/base_config.yaml'
    urdf_dir = 'path/to/urdf/directory'
    config_manager = DroneConfigManager(urdf_dir, base_config_path, batch_size=4)  # Using 4 for this example

    # Load base configuration
    base_cfg = OmegaConf.load(base_config_path)

    # Get a random batch of drone configs
    drone_configs = config_manager.get_random_batch()

    # Create the ModularAerialRobot instance
    robot = ModularAerialRobot(
        cfg=base_cfg,
        sim_params=sim_params,  # You need to define this
        physics_engine=physics_engine,  # You need to define this
        sim_device=sim_device,  # You need to define this
        headless=headless,  # You need to define this
        drone_configs=drone_configs
    )

    # Now you can use this robot instance in your RL training loop
