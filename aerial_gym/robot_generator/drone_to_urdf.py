from typing import List, Tuple

import numpy as np
import torch
from urdfpy import (
    URDF,
    Collision,
    Cylinder,
    Geometry,
    Inertial,
    Joint,
    Link,
    Material,
    Sphere,
    Visual,
    xyz_rpy_to_matrix,
)

from aerial_gym.robot_generator.models.drone_modules import (
    DroneConfig,
    DroneModule,
    LevelEnum,
    ModuleTypeEnum,
)
from aerial_gym.utils.constants import (
    BASE_DEFAULT_LENGTH,
    BASE_DEFAULT_INERTIA,
    BASE_DEFAULT_MASS,
    PROPELLER_DEFAULT_LENGTH,
    PROPELLER_DEFAULT_RADIUS,
    PROPELLER_DEFAULT_INERTIA,
    PROPELLER_DEFAULT_MASS,
)


def get_z_position(module: DroneModule, length: float) -> float:
    """
    Gets the z position of a drone module.
    Args:
        module (DroneModule): The drone module.
        lenght (float): The length of the component.
    Returns:
        float: The z position.
    """
    if module.level == LevelEnum.MID:
        return 0.0
    elif module.level == LevelEnum.TOP:
        return length
    elif module.level == LevelEnum.BOTTOM:
        return -length
    else:
        raise ValueError(f"Invalid level: {module.level}")


def create_module_links(
    module: DroneModule,
    base_length=BASE_DEFAULT_LENGTH,
    propeller_length=PROPELLER_DEFAULT_LENGTH,
    propeller_radius=PROPELLER_DEFAULT_RADIUS,
    propeller_mass = PROPELLER_DEFAULT_MASS,
    propeller_inertia = PROPELLER_DEFAULT_INERTIA
) -> List[Link]:
    """
    Creates the links for a drone module.
    Args:
        module (DroneModule): The drone module.
    Returns:
        Tuple[Link, Link]: The base link and propeller link.
    """
    # Base link setup
    base_name = f"base_link_{module.id}"
    base_visual = Visual(
        origin=xyz_rpy_to_matrix([0, 0, 0, 0, 0, 0]),
        geometry=Geometry(cylinder=Cylinder(radius=module.radius, length=base_length)),
        material=Material(name="White", color=[1, 1, 1, 1.0]),
    )
    base_link_collision = Collision(
        name=base_name + "_collision",
        origin=xyz_rpy_to_matrix([0, 0, 0, 0, 0, 0]),
        geometry=Geometry(sphere=Sphere(radius=module.radius)),
    )
    base_link_inertial = Inertial(mass=BASE_DEFAULT_MASS, inertia=BASE_DEFAULT_INERTIA)
    base_link = Link(
        name=base_name,
        visuals=[base_visual],
        collisions=[base_link_collision],
        inertial=base_link_inertial,
    )
    # Propeller link
    if module.type not in [ModuleTypeEnum.BASE, ModuleTypeEnum.PAYLOAD]:
        propeller_name = f"propeller_link_{module.id}"
        propeller_visual = Visual(
            origin=xyz_rpy_to_matrix([0, 0, 0, 0, 0, 0]),
            geometry=Geometry(cylinder=Cylinder(radius=propeller_radius, length=propeller_length)),
            material=Material(name="Orange", color=[1, 0.423, 0.03921568627, 1.0]),
        )
        propeller_link_inertial = Inertial(
            mass=propeller_mass, inertia=propeller_inertia
        )
        propeller_link = Link(
            name=propeller_name,
            visuals=[propeller_visual],
            collisions=[],
            inertial=propeller_link_inertial,
        )
    else:
        propeller_link = None
    return (base_link, propeller_link)


def create_joints(
    drone: DroneConfig, base_length=BASE_DEFAULT_LENGTH, propeller_length=PROPELLER_DEFAULT_LENGTH
) -> Tuple[Joint, Joint]:
    """
    Creates the joints for a drone.

    Args:
        drone (DroneConfig): The drone.
        base_length (float): The length of the base.
        propeller_length (float): The length of the propeller.

    Returns:
        Tuple[Joint, Joint]: The base joint and propeller joint.
    """
    average_length = (base_length + propeller_length) / 2
    directed_edges = make_directed_edge_indexes(drone.edge_index)
    # Create joints
    base_joints = []
    propeller_joints = []
    for i, edge in enumerate(sorted(directed_edges)):
        # # Base joint
        source, target = edge
        base_joint_name = f"base_joint_{i}"
        base_joint = Joint(
            name=base_joint_name,
            parent=f"base_link_{source}",
            child=f"base_link_{target}",
            joint_type="fixed",
            origin=xyz_rpy_to_matrix(
                list(
                    tuple(
                        np.array(drone.modules[target].position)
                        - np.array(drone.modules[source].position)
                    )
                )
                + [get_z_position(drone.modules[target], average_length), 0, 0, 0]
            ),
        )
        base_joints.append(base_joint)
        # Propeller joint
        if drone.modules[target].type not in [
            ModuleTypeEnum.BASE,
            ModuleTypeEnum.PAYLOAD,
        ]:
            propeller_joint_name = f"propeller_joint_{i}"
            propeller_joint = Joint(
                name=propeller_joint_name,
                parent=f"base_link_{target}",
                child=f"propeller_link_{target}",
                joint_type="fixed",
                origin=xyz_rpy_to_matrix([0] * 6),
            )
        else:
            propeller_joint = None
        propeller_joints.append(propeller_joint)
    return base_joints, propeller_joints


def make_directed_edge_indexes(tensor: torch.Tensor) -> list:
    """Converts a tensor of edge indexes into a list of directed edges.

    Args:
        tensor (torch.Tensor): The tensor of edge indexes.

    Returns:
        list: A list of directed edges.
    """
    directed_edges = set()
    for i in range(tensor.size(1)):
        src, dest = tensor[0, i].item(), tensor[1, i].item()
        if src != dest:
            edge = (min(src, dest), max(src, dest))
            directed_edges.add(edge)
    return list(directed_edges)


def dronecfg_to_urdf(
    name: str,
    drone: DroneConfig,
    base_length: float = BASE_DEFAULT_LENGTH,
    propeller_length: float = PROPELLER_DEFAULT_LENGTH,
) -> URDF:
    """Converts a DroneConfig object into a URDF representation of it.

    Args:
        drone (DroneConfig): The DroneConfig object to convert.

    Returns:
        URDF: The URDF representation of the DroneConfig object from urdfpy.

    Raises:
        ValueError: If the DroneConfig object is invalid.
    """

    # Check if the DroneConfig object is valid
    if not isinstance(drone, DroneConfig):
        raise ValueError("drone must be a DroneConfig object")
    # Instantiate empty list of links and joints
    links = []
    joints = []
    # Add the base and propeller links
    base_links = []
    propeller_links = []
    for module in drone.modules:
        base_link, propeller_link = create_module_links(
            module, base_length=base_length, propeller_length=propeller_length
        )
        base_links.append(base_link)
        if propeller_link:
            propeller_links.append(propeller_link)
    links.extend(base_links)
    links.extend(propeller_links)
    # Add the base and propeller joints
    base_joints, propeller_joints = create_joints(drone)
    propeller_joints = [p for p in propeller_joints if p is not None]
    joints.extend(base_joints)
    joints.extend(propeller_joints)
    robot = URDF(name=name, links=links, joints=joints)
    return robot
