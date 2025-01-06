import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
import matplotlib.widgets as widgets

# Environment Setup
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_aspect('equal')

# Obstacles (static and dynamic)
static_obstacles = [
    [(10, 20), (15, 20), (15, 25), (10, 25)],  # Obstacle 1
    [(25, 10), (30, 10), (30, 15), (25, 15)],  # Obstacle 2
    [(35, 30), (40, 30), (40, 35), (35, 35)],  # Obstacle 3
]

dynamic_obstacle = [(20, 40), (25, 40), (25, 45), (20, 45)]  # Dynamic obstacle
dynamic_obstacle_velocity = np.array([0.0, 0.5])  # Moving downward

# Robot and Target
robot_pos = np.array([5.0, 5.0])
target_pos = np.array([45.0, 45.0])
trail = []
reached_goal = False
robot_circle = None
trail_line = None
force_arrows = []  # To store force arrows for clearing

# Landmarks ("Les Amers")
landmarks = [
    np.array([15.0, 15.0]),  # Landmark 1
    np.array([30.0, 30.0]),  # Landmark 2
    np.array([40.0, 40.0]),  # Landmark 3
]
current_landmark_index = 0

# Forces parameters (Cartwright and Collet model-inspired)
k_attractive = 1.0  # Attractive force gain
k_repulsive = 1000.0  # Repulsive force gain
repulsive_range = 15.0  # Increased range for early obstacle detection
robot_speed = 0.5  # Robot movement speed
force_limit = 5.0  # Limit on the magnitude of forces

# Additional parameters for smooth navigation
smoothing_factor = 0.1  # Smoothing factor for force transitions
min_distance_to_obstacle = 2.0  # Minimum distance to consider for repulsive force
perturbation_strength = 0.1  # Strength of random perturbation

# Metrics
steps_taken = 0
start_time = None

def draw_obstacles():
    """Draw the static and dynamic obstacles."""
    for obstacle in static_obstacles:
        poly = Polygon(obstacle, closed=True, color='black')
        ax.add_patch(poly)
    dynamic_poly = Polygon(dynamic_obstacle, closed=True, color='gray')
    ax.add_patch(dynamic_poly)

def draw_landmarks():
    """Draw the landmarks ("Les Amers")."""
    for landmark in landmarks:
        ax.plot(landmark[0], landmark[1], 'yo', markersize=10, label='Landmark' if landmark[0] == landmarks[0][0] else "")

def init_plot():
    """Initialize the plot with static elements."""
    global robot_circle, trail_line
    draw_obstacles()
    draw_landmarks()
    robot_circle = Circle(robot_pos, 1, color='green')
    ax.add_patch(robot_circle)
    ax.plot(target_pos[0], target_pos[1], 'bo', label='Target')
    trail_line, = ax.plot([], [], 'g-', alpha=0.5, label='Trail')
    ax.legend(loc='upper left', labels=['Target', 'Trail', 'Landmark', 'Attractive Force', 'Repulsive Force', 'Resultant Force'], 
              handles=[
                  plt.Line2D([0], [0], color='blue', lw=2, label='Attractive Force'),
                  plt.Line2D([0], [0], color='red', lw=2, label='Repulsive Force'),
                  plt.Line2D([0], [0], color='green', lw=2, label='Resultant Force'),
                  plt.Line2D([0], [0], color='yellow', lw=2, label='Landmark')
              ])

def clear_force_arrows():
    """Remove existing force arrows from the plot."""
    global force_arrows
    for arrow in force_arrows:
        arrow.remove()
    force_arrows = []

def update_robot_position():
    """Update the robot's position and trail."""
    global robot_circle, trail_line

    # Update robot circle
    robot_circle.center = robot_pos

    # Update trail
    trail.append(robot_pos.copy())
    trail_array = np.array(trail)
    trail_line.set_data(trail_array[:, 0], trail_array[:, 1])

    # Clear previous arrows
    clear_force_arrows()

    # Draw forces
    attractive_force = compute_attractive_force()
    repulsive_force = compute_repulsive_force()
    total_force = attractive_force + repulsive_force

    # Draw attractive force arrow
    force_arrows.append(draw_force_arrow(robot_pos, attractive_force, color='blue', alpha=1.0, scale=1.5))

    # Draw repulsive force arrow
    force_arrows.append(draw_force_arrow(robot_pos, repulsive_force, color='red', alpha=1.0, scale=1.5))

    # Draw total force arrow
    force_arrows.append(draw_force_arrow(robot_pos, total_force, color='green', alpha=1.0, scale=1.5))

def draw_force_arrow(position, force, color, alpha=1.0, scale=1.5):
    """Draw a large and clear arrow representing a force vector linked to the robot."""
    arrow = ax.arrow(
        position[0], position[1],
        scale * force[0], scale * force[1],
        head_width=1.0, head_length=1.0,
        fc=color, ec=color, alpha=alpha
    )
    return arrow

def compute_attractive_force():
    """Compute the attractive force (Cartwright and Collet model-inspired)."""
    global current_landmark_index
    # Attract to the current landmark
    if current_landmark_index < len(landmarks):
        direction = landmarks[current_landmark_index] - robot_pos
    else:
        direction = target_pos - robot_pos
    distance = np.linalg.norm(direction)
    if distance > 0:
        return k_attractive * direction / distance
    return np.array([0.0, 0.0])

def compute_repulsive_force():
    """Compute the repulsive forces from obstacles (Cartwright and Collet model-inspired)."""
    repulsive_force = np.array([0.0, 0.0])
    for obstacle in static_obstacles + [dynamic_obstacle]:
        for vertex in obstacle:
            direction = robot_pos - np.array(vertex)
            distance = np.linalg.norm(direction)
            if 0 < distance < repulsive_range:
                # Smooth repulsive force calculation
                force_magnitude = k_repulsive * (1 / distance - 1 / repulsive_range) * (1 / distance**2)
                repulsive_force += force_magnitude * (direction / distance)
    return repulsive_force

def compute_forces():
    """Compute the total forces acting on the robot (Cartwright and Collet model-inspired)."""
    attractive_force = compute_attractive_force()
    repulsive_force = compute_repulsive_force()

    # Smooth the transition between forces
    total_force = (1 - smoothing_factor) * attractive_force + smoothing_factor * repulsive_force

    # Add a small random perturbation to avoid getting stuck
    perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, size=2)
    total_force += perturbation

    # Limit the total force
    if np.linalg.norm(total_force) > force_limit:
        total_force = force_limit * total_force / np.linalg.norm(total_force)

    return total_force

def move_robot():
    """Move the robot."""
    global reached_goal, steps_taken, current_landmark_index
    if current_landmark_index < len(landmarks):
        # Move toward the current landmark
        if np.linalg.norm(landmarks[current_landmark_index] - robot_pos) < robot_speed:
            current_landmark_index += 1  # Switch to the next landmark
    else:
        # Move toward the target
        if np.linalg.norm(target_pos - robot_pos) < robot_speed:
            reached_goal = True
            return
    force = compute_forces()
    robot_pos[:] += robot_speed * force / np.linalg.norm(force)
    steps_taken += 1

def move_dynamic_obstacle():
    """Move the dynamic obstacle."""
    global dynamic_obstacle
    for i in range(len(dynamic_obstacle)):
        dynamic_obstacle[i] = tuple(np.array(dynamic_obstacle[i]) + dynamic_obstacle_velocity)
    # Reverse direction if the obstacle hits the boundary
    if dynamic_obstacle[0][1] < 0 or dynamic_obstacle[2][1] > 50:
        dynamic_obstacle_velocity[:] *= -1

def stop_simulation(event):
    """Stop the simulation."""
    global running
    running = False

def start_simulation(event):
    """Start the simulation."""
    global running, start_time
    running = True
    start_time = plt.datetime.datetime.now()

# Buttons
stop_ax = plt.axes([0.7, 0.01, 0.1, 0.05])
stop_button = widgets.Button(stop_ax, 'Stop', color='red', hovercolor='darkred')
stop_button.on_clicked(stop_simulation)

start_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
start_button = widgets.Button(start_ax, 'Start', color='green', hovercolor='darkgreen')
start_button.on_clicked(start_simulation)

# Initialize plot
init_plot()
plt.ion()

# Simulation loop
running = False
while not reached_goal:
    if running:
        move_robot()
        move_dynamic_obstacle()
        update_robot_position()
        plt.pause(0.1)
    else:
        plt.pause(0.1)

if reached_goal:
    print("Le robot a atteint la cible !")
    print(f"Steps taken: {steps_taken}")
    if start_time:
        end_time = plt.datetime.datetime.now()
        print(f"Time elapsed: {(end_time - start_time).total_seconds()} seconds")

plt.show()