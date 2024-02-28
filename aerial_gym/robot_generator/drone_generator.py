import random
from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import cdist

import isaacgym
import logging
from aerial_gym.exceptions.common import InvalidArchitectureException
from aerial_gym.robot_generator.models.drone_modules import (
    DroneConfig,
    DroneConnection,
    DroneModule,
    LevelEnum,
    ModuleTypeEnum,
)
from aerial_gym.utils.constants import MID_LEVEL_PROBABILITY
import torch


def setup_edge_indexes(drone: DroneConfig) -> torch.Tensor:
    """
    Setup edge indexes and defines edge index for a representative graph of the drone
    architecture. The edges are undirected.

    Parameters
    ----------
        drone: A DroneConfig object
    Returns
    -------
        edge_indexes: torch tensor of edge indexes.
    """
    # setup edge indexes
    edge_indexes = []
    for module in drone.modules:
        if module.connections.connector_1 is not None:
            edge_indexes.append([module.id, module.connections.connector_1])
        if module.connections.connector_2 is not None:
            edge_indexes.append([module.id, module.connections.connector_2])
        if module.connections.connector_3 is not None:
            edge_indexes.append([module.id, module.connections.connector_3])
        if module.connections.connector_4 is not None:
            edge_indexes.append([module.id, module.connections.connector_4])
        if module.connections.connector_5 is not None:
            edge_indexes.append([module.id, module.connections.connector_5])
        if module.connections.connector_6 is not None:
            edge_indexes.append([module.id, module.connections.connector_6])
    drone.edge_index = torch.tensor(edge_indexes).t().contiguous()
    return drone


def get_target_position(source_module, connector):
    """
    Get the position of the target module based on the source module and connector.
    Args:
        source_module (DroneModule): The source module.
        connector (int): The connector index.
    Returns:
        Tuple[float, float]: The position of the target module.
    """
    diameter = 2 * source_module.radius
    if connector == 1:
        target_position = (
            source_module.position[0],
            source_module.position[1] + diameter,
        )
    elif connector == 2:
        target_position = (
            source_module.position[0] + (np.sqrt(3) / 2) * diameter,
            source_module.position[1] + 0.5 * diameter,
        )
    elif connector == 3:
        target_position = (
            source_module.position[0] + (np.sqrt(3) / 2) * diameter,
            source_module.position[1] - 0.5 * diameter,
        )
    elif connector == 4:
        target_position = (
            source_module.position[0],
            source_module.position[1] - diameter,
        )
    elif connector == 5:
        target_position = (
            source_module.position[0] - (np.sqrt(3) / 2) * diameter,
            source_module.position[1] - 0.5 * diameter,
        )
    elif connector == 6:
        target_position = (
            source_module.position[0] - (np.sqrt(3) / 2) * diameter,
            source_module.position[1] + 0.5 * diameter,
        )
    else:
        raise ValueError("Invalid connector index")
    return target_position


def check_free_space(source_module, connector, other_modules):
    """
    Check if the module is not overlapping with other modules.
    Args:
        source_module (DroneModule): The source module to check.
        connector (int): The connector to check.
        other_modules (List[DroneModule]): The list of other modules.
    Returns:
        bool: True if the module is not overlapping, False otherwise.
    """
    # Get the position of the target module
    target_position = get_target_position(source_module, connector)
    # Check if the target position is overlapping with any other module
    for other_module in other_modules:
        # Compute the distance between the target position and the other module's position
        distance = cdist([target_position], [other_module.position], metric="euclidean")
        if distance[0][0] < 2 * other_module.radius:
            return False
    return True


def randomize_modules(source_module, module_list) -> List[int]:
    """
    Randomize number of modules and return the number of modules and the connectors
    that they will be attached on.
    Args:
        source_module (DroneModule): The source module to be randomized.
        module_list (List[DroneModule]): The list of current modules.
    Returns:
        List[int]: The connectors that will be used in this module.
    """
    # Instance the module IDs that are being randomized
    free_connectors = [
        i
        for i in range(1, 7)
        if source_module.connections.get_connector_value(i) is None
    ]
    if not len(free_connectors):
        return []
    n_modules = (
        np.random.randint(1, len(free_connectors)) if len(free_connectors) > 1 else 1
    )
    # sample allowed_modules from free_connectors
    connectors = np.random.choice(free_connectors, n_modules, replace=False)
    # check if there is no overlap
    connectors = [
        c for c in connectors if check_free_space(source_module, c, module_list)
    ]
    return connectors


def create_module(
    id: int, num_bases: int, num_propellers: int, allow_unevenness: bool, **kwargs
):
    """
    Creates a module with a randomized type and vertical alignment.
    Args:
        num_bases (int): The number of bases to be created.
        num_propellers (int): The number of propellers to be created.
    Returns:
        DroneModule: The created module.
    """
    # Randomize module type
    prob_base = num_bases / (num_bases + num_propellers)
    if num_propellers > 1:
        prob_cc_propeller = prob_cw_propeller = (
            0.5 * num_propellers / (num_bases + num_propellers)
        )
    else:
        prob_cc_propeller = prob_cw_propeller = num_propellers / (
            num_bases + num_propellers
        )
    module_type = random.choices(
        population=[
            ModuleTypeEnum.BASE,
            ModuleTypeEnum.CC_PROPELLER,
            ModuleTypeEnum.CW_PROPELLER,
        ],
        weights=[prob_base, prob_cc_propeller, prob_cw_propeller],
        k=1,
    )[0]
    # Randomize module vertical alignment
    if allow_unevenness:
        module_level = random.choices(
            population=[LevelEnum.BOTTOM, LevelEnum.MID, LevelEnum.TOP],
            weights=[
                (1 - MID_LEVEL_PROBABILITY) / 2,
                MID_LEVEL_PROBABILITY,
                (1 - MID_LEVEL_PROBABILITY) / 2,
            ],
            k=1,
        )[0]
    else:
        module_level = LevelEnum.MID
    return DroneModule(id=id, type=module_type, level=module_level, **kwargs)


def attach_module(
    source_module: DroneModule, target_module: DroneModule, source_connector: int
) -> Tuple[DroneModule, DroneModule]:
    """Connects two modules and return their updated versions (with the correspondent connections).
    Note that the positions are indexed in a clockwise pattern (where 1 is located at the 12 position in the clock), and they are connected as follows:
        connector 1 <-> connector_4
        connector 2 <-> connector_5
        connector 3 <-> connector_6
    Args:
        source_module (DroneModule): The source module to be connected.
        target_module (DroneModule): The target module to be connected.
        connector (int): The connector to connect the modules.
    Returns:
        Tuple[DroneModule, DroneModule]: The updated versions of the modules.
    """
    target_connector = (
        source_connector + 3 if source_connector <= 3 else source_connector - 3
    )
    # Instance the module IDs that are being connected
    source_module.connections.set_connector_value(source_connector, target_module.id)
    target_module.connections.set_connector_value(target_connector, source_module.id)
    target_module.position = get_target_position(source_module, source_connector)
    logging.debug(
        f"Attaching module {target_module.id} into {source_module.id} at connector {source_connector}"
    )
    logging.debug(
        f"Source module position = {source_module.position}, and target module position = {target_module.position}"
    )
    return source_module, target_module


def generate_random_drone(
    num_propellers: int,
    num_bases: int = 1,
    allow_unevenness: bool = True,
):
    if num_propellers < 3 or num_bases < 1:
        raise InvalidArchitectureException().create(num_propellers, num_bases)
    # num_propellers = num_bases + num_propellers if not symmetric else (num_bases + num_propellers)//2
    # sym_remainer = 0 if not symmetric else (num_bases + num_propellers)%2
    modules = []
    modules.append(
        DroneModule(
            id=0,
            type=ModuleTypeEnum.BASE,
            level=LevelEnum.MID,
            position=(0, 0),
            connections=DroneConnection(),
        )
    )
    remaining_bases = np.clip(num_bases - 1, 0, None)
    remaining_propellers = num_propellers
    remaining_modules = remaining_bases + remaining_propellers
    # # TODO: Solve symmetry problem and num_payloads later
    # remaining_payloads = num_payloads
    # sym_remainer = 0
    # if symmetric:
    #     sym_remainer = remaining_modules % 2
    #     remaining_modules = remaining_modules // 2
    module_id = 0
    source_id = 0
    visited_source_ids = [source_id]
    while remaining_modules > 0:
        chosen_connectors = randomize_modules(modules[source_id], modules)
        print(f"chosen_connectors: {chosen_connectors}")
        for conn in chosen_connectors:
            module_id += 1
            new_module = create_module(
                id=module_id,
                num_bases=remaining_bases,
                num_propellers=remaining_propellers,
                connections=DroneConnection(),
                allow_unevenness=allow_unevenness,
            )
            source_module, target_module = attach_module(
                source_module=modules[source_id],
                target_module=new_module,
                source_connector=conn,
            )
            modules.append(target_module)
            if target_module.type == ModuleTypeEnum.BASE:
                remaining_bases -= 1
            elif (
                target_module.type == ModuleTypeEnum.CW_PROPELLER
                or target_module.type == ModuleTypeEnum.CC_PROPELLER
            ):
                remaining_propellers -= 1
            remaining_modules -= 1
            modules[source_id] = source_module
            visited_source_ids.append(source_id)
            if remaining_modules == 0:
                break
        not_visited = list(set([m.id for m in modules]) - set(visited_source_ids))
        source_id = np.random.choice(not_visited, 1, False)[0]

    drone = DroneConfig(
        modules=modules,
        edge_index=[],
        num_modules=len(modules),
        num_propellers=num_propellers,
        num_bases=num_bases,
        num_nodes=len(modules),
        allow_unevenness=allow_unevenness,
    )
    drone = setup_edge_indexes(drone)
    return drone
