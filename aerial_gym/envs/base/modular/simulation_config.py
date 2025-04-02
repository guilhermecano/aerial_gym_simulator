# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


class SimulationCfg():
    seed = 1
    class env:
        num_envs = 65536
        num_observations = 13
        get_privileged_obs = False # if True the states of all entitites in the environment will be returned as privileged observations, otherwise None will be returned
        env_spacing = 1
        episode_length_s = 8 # episode length in seconds
        num_control_steps_per_env_step = 1 # number of physics steps per env step

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [-5, -5, 4]  # [m]
        lookat = [0, 0, 0]  # [m]

    class sim:
        dt =  0.01
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 0 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
