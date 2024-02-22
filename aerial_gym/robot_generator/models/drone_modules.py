from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel

from aerial_gym.utils.constants import MODULE_DEFAULT_RADIUS


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