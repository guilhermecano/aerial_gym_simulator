import random
import pygraphviz as pgv
from PIL import Image
from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum
from typing import Tuple

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
        return None

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


def randomize_modules(source_module, allowed_modules: int)-> List[int]:
    """
    Given a number of allowed modules to be attached into a source module, randomize it and return the number of modules and the positions they will be attached.
    Args:
        source_module (DroneModule): The source module to be randomized.
        allowed_modules (int): The number of allowed modules to be attached.
    Returns:
        List[int]: The connectors that will be used in this module.
    """
    # Instance the module IDs that are being randomized
    n_modules = np.random.randint(0, allowed_modules)
    free_connectors = [i for i in range(1, 7) if source_module.connections.get_connector_value(i) is None]
    # sample allowed_modules from free_connectors
    connectors = np.random.choice(free_connectors, n_modules, replace=False)
    return connectors


def create_module(id: int, num_bases: int, num_propellers: int, **kwargs):
    """
    Creates a module with a randomized type and vertical alignment.
    Args:
        num_bases (int): The number of bases to be created.
        num_propellers (int): The number of propellers to be created.
    Returns:
        DroneModule: The created module.
    """
    # Randomize module type
    prob_base = num_bases/(num_bases + num_propellers)
    if num_propellers > 1:
        prob_cc_propeller = prob_cw_propeller = 0.5 * num_propellers/(num_bases + num_propellers)
    else:
        prob_cc_propeller = prob_cw_propeller = num_propellers/(num_bases + num_propellers)
    module_type = random.choices(population=[ModuleTypeEnum.BASE, 
    ModuleTypeEnum.CC_PROPELLER, ModuleTypeEnum.CW_PROPELLER], weights=[prob_base, prob_cc_propeller, prob_cw_propeller], k=1)[0]
    # Randomize module vertical alignment
    module_level = random.choices(population=[LevelEnum.BOTTOM, LevelEnum.MID, LevelEnum.TOP], 
        weights=[(1 - MID_LEVEL_PROBABILITY)/2, MID_LEVEL_PROBABILITY, (1 - MID_LEVEL_PROBABILITY)/2], 
        k=1)[0]
    return DroneModule(id=id, type=module_type, level=module_level, **kwargs)
    

def attach_module(source_module: DroneModule, target_module: DroneModule, source_connector:int) -> Tuple[DroneModule, DroneModule]:
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
    target_connector = source_connector + 3 if source_connector < 3 else source_connector - 3
    # Instance the module IDs that are being connected
    source_module.connections.set_connector_value(source_connector, target_module.id)
    target_module.connections.set_connector_value(target_connector, source_module.id)
    # Setup the positions of the target module
    distance = source.radius + target.radius
    if target_connector == 1:
        taget_position = (source_module.position[0], source_module.position[1] + distance)
    elif target_connector == 2:
        taget_position = (source_module.position[0] + 0.5 * distance, source_module.position[1] + (np.sqrt(3)/2) * distance)
    elif target_connector == 3:
        taget_position = (source_module.position[0] + 0.5 * distance, source_module.position[1] - (np.sqrt(3)/2) * distance)
    elif target_connector == 4:
        taget_position = (source_module.position[0], source_module.position[1] - distance)
    elif target_connector == 5:
        taget_position = (source_module.position[0] - 0.5 * distance, source_module.position[1] - (np.sqrt(3)/2) * distance)
    elif target_connector == 6:
        target_position = (source_module.position[0] - 0.5 * distance, source_module.position[1] + (np.sqrt(3)/2) * distance)
    else:
        raise ValueError("Invalid connector index")
    target_module.position = target_position
    return source_module, target_module


def generate_random_drone(num_propellers: int, num_bases: int = 1, num_payloads:int = 0, allow_unevenness: bool = True, symmetric:bool = False):
    if num_propellers < 3 or num_bases < 1:
        # raise InvalidArchitectureException().create(num_propellers, num_bases)
        return None 
    # num_propellers = num_bases + num_propellers if not symmetric else (num_bases + num_propellers)//2
    # sym_remainer = 0 if not symmetric else (num_bases + num_propellers)%2
    modules = []
    modules.append(DroneModule(id=0, type=ModuleTypeEnum.BASE, level=LevelEnum.MID, position=(0,0), connections=DroneConnection()))
    remaining_modules = num_bases + num_propellers + num_payloads
    sym_remainer = 0
    if symmetric:
        sym_remainer = remaining_modules%2
        remaining_modules = remaining_modules//2
    i = 0
    while remaining_modules > 0:
        chosen_connectors = randomize_modules(modules[i])
        for conn in chosen_connectors:
            source_module, target_module = attach_module(modules[i], )

    
    
    
    # for i in range(len(base_id), num_propellers):
    #     if i == 0:
    #         module_type = ModuleTypeEnum.BASE
    #     else:
    #         module_type = ModuleTypeEnum.CW_PROPELLER if i % 2 == 0 else ModuleTypeEnum.CC_PROPELLER
        
        
    # # If valid number of propellers and bases, generate a viable random tree
    # propellers = [TreeNode("Propeller_"+str(i)) for i in range(num_propellers)]
    # bases = [TreeNode("Base_"+str(i)) for i in range(num_bases)]
    # num_nodes = num_bases + num_propellers
    # reference_base = bases.pop(0)
    # nodes = bases + propellers
    # parent = reference_base
    # for i in range(0, num_nodes):
    #     if i >= 1:
    #         parent = random.choice(nodes[:i])
    #     parent.children.append(nodes[i])
    # return nodes[0]