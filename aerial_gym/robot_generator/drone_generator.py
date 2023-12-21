import random
import pygraphviz as pgv
from PIL import Image
from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum
from typing import Tuple
from scipy.spatial.distance import cdist
from aerial_gym import AERIAL_GYM_ROOT_DIR
from urdfpy import URDF
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from aerial_gym.exceptions.common import InvalidArchitectureException

from aerial_gym.utils.constants import MODULE_DEFAULT_RADIUS, MID_LEVEL_PROBABILITY


class LevelEnum(Enum):
    TOP = 1
    MID = 2
    BOTTOM = 3


class ModuleTypeEnum(str, Enum):
    BASE = "BASE"
    CW_PROPELLER = "CW_PROPELLER"
    CC_PROPELLER = "CC_PROPELLER"
    PAYLOAD = "PAYLOAD"


class DroneConnection(BaseModel):
    connector_1: Optional[int] = None
    connector_2: Optional[int] = None
    connector_3: Optional[int] = None
    connector_4: Optional[int] = None
    connector_5: Optional[int] = None
    connector_6: Optional[int] = None

    def get_connector_value(self, i):
        if i == 1:
            return self.connector_1
        elif i == 2:
            return self.connector_2
        elif i == 3:
            return self.connector_3
        elif i == 4:
            return self.connector_4
        elif i == 5:
            return self.connector_5
        elif i == 6:
            return self.connector_6
        else:
            raise ValueError("Invalid connector index")

    def set_connector_value(self, i, value):
        if i == 1:
            self.connector_1 = value
        elif i == 2:
            self.connector_2 = value
        elif i == 3:
            self.connector_3 = value
        elif i == 4:
            self.connector_4 = value
        elif i == 5:
            self.connector_5 = value
        elif i == 6:
            self.connector_6 = value
        else:
            raise ValueError("Invalid connector index")


class DroneModule(BaseModel):
    id: int
    type: ModuleTypeEnum
    connections: DroneConnection
    radius: int = MODULE_DEFAULT_RADIUS
    level: LevelEnum = LevelEnum.MID
    position: Optional[Tuple[float, float]] = None


class DroneConfig(BaseModel):
    modules: List[DroneModule]
    edge_index: List[List[int]]
    num_modules: int
    num_propellers: int
    num_bases: int
    num_nodes: int
    allow_unevenness: bool


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
        if distance[0][0] < 2*other_module.radius:
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
    connectors = [c for c in connectors if check_free_space(source_module, c, module_list)]
    print("-"*50)
    print(source_module.id)
    print([c for c in connectors if check_free_space(source_module, c, module_list)])
    return connectors


def create_module(id: int, num_bases: int, num_propellers: int, allow_unevenness: bool, **kwargs):
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
    print(f'Attaching module {target_module.id} into {source_module.id} at connector {source_connector}')
    print(f"Source module position = {source_module.position}, and target module position = {target_module.position}")
    return source_module, target_module


def generate_random_drone(
    num_propellers: int,
    num_bases: int = 1,
    num_payloads: int = 0,
    allow_unevenness: bool = True,
    symmetric: bool = False,
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
    remaining_payloads = num_payloads
    # TODO: Solve symmetry problem later
    sym_remainer = 0
    if symmetric:
        sym_remainer = remaining_modules % 2
        remaining_modules = remaining_modules // 2
    module_id = 0
    source_id = 0
    visited_source_ids = [source_id]
    while remaining_modules > 0:
        # print(f"Source ID: {source_id}")
        # print(f"Len modules: {len(modules)}")
        # print(f"remaining_propellers: {remaining_propellers}")
        # print(f"remaining_bases: {remaining_bases}")
        # print(f"remaining_modules: {remaining_modules}")
        chosen_connectors = randomize_modules(modules[source_id], modules)
        print(f"chosen_connectors: {chosen_connectors}")
        for conn in chosen_connectors:
            module_id += 1
            new_module = create_module(
                id=module_id,
                num_bases=remaining_bases,
                num_propellers=remaining_propellers,
                connections=DroneConnection(),
                allow_unevenness=allow_unevenness
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
    for md in modules:
        print(md.position)
    return DroneConfig(
        modules=modules,
        edge_index=[],
        num_modules=len(modules),
        num_propellers=num_propellers,
        num_bases=num_bases,
        num_nodes=len(modules),
        allow_unevenness=allow_unevenness,
    )



# drone = generate_random_drone(num_propellers=4, num_bases=1, num_payloads=0, allow_unevenness=False)

# fig, ax = plt.subplots(figsize=(14, 11))

# print("---------------- plotting -----------------")
# for module in drone.modules:
#     circle_id = module.id
#     radius = module.radius
#     x = module.position[0]
#     y = module.position[1]
#     print(f"({x}, {y})")
#     circle_type = module.type

#     # Set circle color based on circle_type
#     color = 'blue' if circle_type == ModuleTypeEnum.BASE else 'green'

#     # Plot the circle
#     circle_plot = plt.Circle((x, y), radius, edgecolor='black', facecolor=color, alpha=0.7)
#     ax.add_patch(circle_plot)

#     # Add text inside the circle with the circle_id
#     ax.text(x, y, str(circle_id) + f"_{module.level.name}", color='white', ha='center', va='center', fontweight='bold')

# # Set aspect ratio to 'equal' for a more accurate representation of circles
# ax.set_aspect('equal', adjustable='box')

# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.ylim(-1, 1)
# plt.xlim(-1, 1)
# plt.title('Indexed Circles')

# plt.show()