import matplotlib.pyplot as plt
from aerial_gym.robot_generator.models.drone_modules import DroneConfig, ModuleTypeEnum

def pĺot_drone(drone: DroneConfig):
    """Plots the drone's architecture
    
    Args:
        drone: DroneConfig object
    """
    _, ax = plt.subplots(figsize=(14, 11))
    for module in drone.modules:
        circle_id = module.id
        radius = module.radius
        x = module.position[0]
        y = module.position[1]
        print(f"Module {module.id} position: ({x}, {y})")
        circle_type = module.type
        # Set circle color based on circle_type
        color = 'blue' if circle_type == ModuleTypeEnum.BASE else 'green'
        # Plot the circle
        circle_plot = plt.Circle((x, y), radius, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(circle_plot)
        # Add text inside the circle with the circle_id
        ax.text(x, y, str(circle_id) + f"_{module.level.name}", color='white', ha='center', va='center', fontweight='bold')
    # Set aspect ratio to 'equal' for a more accurate representation of circles
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.title('Indexed Circles')
    plt.show()