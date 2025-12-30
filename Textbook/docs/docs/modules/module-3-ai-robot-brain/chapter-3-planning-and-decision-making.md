---
sidebar_label: 'Chapter 3: Planning and Decision Making'
sidebar_position: 3
---

# Chapter 3: Planning and Decision Making

## Learning Objectives
- Understand different types of planning in robotics
- Implement path planning and motion planning algorithms
- Learn about decision-making frameworks for robots
- Integrate planning with perception and control systems
- Analyze planning algorithms for computational efficiency and optimality

## Overview of Robot Planning
Robot planning involves determining a sequence of actions to achieve desired goals. It bridges the gap between high-level objectives and low-level control commands, considering environmental constraints and robot capabilities.

### Planning Hierarchy
Robot planning typically occurs at multiple levels:

1. **Task Planning**: High-level goal decomposition and sequencing
2. **Motion Planning**: Pathfinding and trajectory generation
3. **Trajectory Planning**: Time-parameterized motion with dynamic constraints
4. **Control Execution**: Low-level actuator commands

```python
# Example: Hierarchical planning system
class HierarchicalPlanner:
    def __init__(self):
        self.task_planner = TaskPlanner()
        self.motion_planner = MotionPlanner()
        self.trajectory_planner = TrajectoryPlanner()
        self.controller = ControllerBase()

    def plan_to_goal(self, initial_state, goal_state, environment):
        # Task planning
        task_sequence = self.task_planner.plan_tasks(goal_state)

        # Motion planning for each task
        motion_paths = []
        current_state = initial_state

        for task in task_sequence:
            path = self.motion_planner.plan_path(current_state, task.goal, environment)
            motion_paths.append(path)
            current_state = path.end_state

        # Trajectory planning with timing
        trajectories = []
        for path in motion_paths:
            trajectory = self.trajectory_planner.plan_trajectory(path)
            trajectories.append(trajectory)

        return trajectories
```

### Planning Components
Key components of a planning system:

- **State Space**: The space of all possible configurations
- **Action Space**: Available actions or movements
- **Cost Function**: Measures the quality of different paths
- **Constraints**: Physical and environmental limitations
- **Optimization Criteria**: What to optimize (distance, time, energy)

## Types of Planning

### Motion Planning
Motion planning focuses on finding collision-free paths for the robot.

#### Configuration Space (C-Space)
Planning in the robot's joint space:

```python
import numpy as np
from scipy.spatial.distance import euclidean

class ConfigurationSpacePlanner:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.joint_limits = robot_model.joint_limits

    def is_collision_free(self, joint_angles, obstacles):
        """Check if a configuration is collision-free"""
        # Calculate robot's end-effector position
        end_effector_pos = self.robot_model.forward_kinematics(joint_angles)

        # Check collision with obstacles
        for obstacle in obstacles:
            if self.check_collision(end_effector_pos, obstacle):
                return False
        return True

    def plan_motion(self, start_joints, goal_joints, obstacles):
        """Plan motion in configuration space"""
        # Use RRT or other motion planning algorithm
        path = self.rrt_planning(start_joints, goal_joints, obstacles)
        return path

    def rrt_planning(self, start, goal, obstacles):
        """RRT algorithm implementation"""
        tree = [start]
        max_iterations = 1000

        for _ in range(max_iterations):
            # Sample random configuration
            random_config = self.sample_configuration()

            # Find nearest node in tree
            nearest_idx = self.find_nearest(tree, random_config)

            # Extend tree towards random configuration
            new_config = self.extend_towards(tree[nearest_idx], random_config)

            # Check collision
            if self.is_collision_free(new_config, obstacles):
                tree.append(new_config)

                # Check if goal is reached
                if self.is_near_goal(new_config, goal):
                    return self.reconstruct_path(tree, len(tree)-1)

        return None  # No path found
```

#### Cartesian Space Planning
Planning in 3D workspace:

```python
class CartesianPlanner:
    def __init__(self, workspace_bounds):
        self.bounds = workspace_bounds

    def plan_cartesian_path(self, start_pose, goal_pose, obstacles):
        """Plan path in Cartesian space"""
        # Convert to configuration space using inverse kinematics
        start_configs = self.inverse_kinematics(start_pose)
        goal_configs = self.inverse_kinematics(goal_pose)

        # Plan in configuration space
        config_path = self.plan_config_path(start_configs, goal_configs, obstacles)

        # Convert back to Cartesian path
        cartesian_path = self.config_to_cartesian(config_path)

        return cartesian_path

    def inverse_kinematics(self, pose):
        """Calculate inverse kinematics for pose"""
        # Multiple solutions possible for redundant robots
        solutions = self.calculate_ik_solutions(pose)
        return solutions
```

#### Task Space Planning
Planning for specific tasks or behaviors:

```python
class TaskSpacePlanner:
    def __init__(self):
        self.task_library = self.load_task_library()

    def plan_task(self, task_description, current_state):
        """Plan for a specific task"""
        # Parse task requirements
        task_requirements = self.parse_task(task_description)

        # Find appropriate planning strategy
        planner = self.select_planner(task_requirements)

        # Generate task-specific plan
        task_plan = planner.generate_plan(task_requirements, current_state)

        return task_plan

    def parse_task(self, task_description):
        """Parse natural language task description"""
        # Example: "Move object A to location B"
        # Extract: object, target_location, action_type
        pass
```

### Path Planning

#### Global Planning
Computing paths from start to goal:

```python
import heapq
from collections import defaultdict

class GlobalPathPlanner:
    def __init__(self, map_resolution=0.1):
        self.resolution = map_resolution

    def a_star(self, start, goal, occupancy_grid):
        """A* path planning algorithm"""
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        # Initialize open and closed sets
        open_set = [(0, start_grid)]
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                if self.is_occupied(neighbor, occupancy_grid):
                    continue

                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)

                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, a, b):
        """Heuristic function (Euclidean distance)"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def world_to_grid(self, world_coords):
        """Convert world coordinates to grid coordinates"""
        return (int(world_coords[0] / self.resolution),
                int(world_coords[1] / self.resolution))

    def get_neighbors(self, pos):
        """Get 8-connected neighbors"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((pos[0] + dx, pos[1] + dy))
        return neighbors
```

#### Local Planning
Reactive path following and obstacle avoidance:

```python
class LocalPathPlanner:
    def __init__(self, local_map_size=5.0, control_frequency=10.0):
        self.local_map_size = local_map_size
        self.control_frequency = control_frequency
        self.lookahead_distance = 1.0

    def follow_path(self, global_path, current_pose, local_obstacles):
        """Follow global path while avoiding local obstacles"""
        # Find current position on path
        current_idx = self.find_closest_point(global_path, current_pose)

        # Get look-ahead point
        lookahead_idx = min(current_idx + 10, len(global_path) - 1)
        lookahead_point = global_path[lookahead_idx]

        # Check for obstacles along path
        if self.path_has_obstacles(global_path[current_idx:lookahead_idx], local_obstacles):
            # Local obstacle avoidance
            avoidance_cmd = self.avoid_obstacles(current_pose, local_obstacles)
            return avoidance_cmd

        # Follow global path
        path_cmd = self.follow_global_path(current_pose, lookahead_point)
        return path_cmd

    def avoid_obstacles(self, current_pose, obstacles):
        """Local obstacle avoidance using VFH or similar"""
        # Vector Field Histogram approach
        histogram = self.create_histogram(obstacles, current_pose)

        # Find safe direction
        safe_direction = self.find_safe_direction(histogram)

        # Generate velocity command
        return self.generate_velocity_command(safe_direction)

    def create_histogram(self, obstacles, robot_pose):
        """Create polar histogram of obstacles"""
        # Divide space into sectors
        num_sectors = 72  # 5-degree resolution
        histogram = np.zeros(num_sectors)

        for obstacle in obstacles:
            # Calculate relative position
            rel_x = obstacle.x - robot_pose.x
            rel_y = obstacle.y - robot_pose.y

            # Calculate angle and distance
            angle = np.arctan2(rel_y, rel_x)
            distance = np.sqrt(rel_x**2 + rel_y**2)

            # Add to histogram
            sector = int((angle + np.pi) / (2 * np.pi) * num_sectors)
            if 0 <= sector < num_sectors:
                histogram[sector] += 1.0 / (distance + 0.1)  # Closer obstacles get higher values

        return histogram
```

#### Anytime Planning
Algorithms that improve solutions over time:

```python
class AnytimePlanner:
    def __init__(self, time_limit=5.0):
        self.time_limit = time_limit
        self.best_solution = None
        self.best_cost = float('inf')

    def anytime_search(self, start, goal, environment):
        """Anytime search algorithm"""
        start_time = time.time()

        # Initial quick solution
        initial_solution = self.get_quick_solution(start, goal, environment)
        self.best_solution = initial_solution
        self.best_cost = self.calculate_cost(initial_solution)

        # Continue improving solution until time runs out
        while time.time() - start_time < self.time_limit:
            # Try to improve current solution
            improved_solution = self.improve_solution(
                self.best_solution, start, goal, environment
            )

            if improved_solution:
                new_cost = self.calculate_cost(improved_solution)
                if new_cost < self.best_cost:
                    self.best_solution = improved_solution
                    self.best_cost = new_cost

        return self.best_solution

    def get_quick_solution(self, start, goal, environment):
        """Get initial solution quickly"""
        # Use greedy or simple heuristic-based approach
        return self.greedy_path(start, goal, environment)

    def improve_solution(self, current_solution, start, goal, environment):
        """Try to improve current solution"""
        # Local search, random perturbation, or other improvement method
        return self.local_search(current_solution, environment)
```

### Trajectory Planning
Generating time-parameterized paths with dynamic constraints:

```python
import numpy as np
from scipy.interpolate import splprep, splev

class TrajectoryPlanner:
    def __init__(self):
        self.max_velocity = 1.0  # m/s
        self.max_acceleration = 0.5  # m/s^2
        self.control_rate = 50.0  # Hz

    def plan_trajectory(self, path, initial_velocity=0.0, final_velocity=0.0):
        """Plan time-parameterized trajectory from path"""
        # Smooth the path
        smoothed_path = self.smooth_path(path)

        # Calculate time allocation
        time_allocation = self.calculate_time_allocation(smoothed_path)

        # Generate velocity profile
        velocity_profile = self.generate_velocity_profile(
            smoothed_path, time_allocation, initial_velocity, final_velocity
        )

        # Create trajectory
        trajectory = self.create_trajectory(smoothed_path, velocity_profile)

        return trajectory

    def smooth_path(self, path):
        """Smooth the path using spline interpolation"""
        if len(path) < 3:
            return path

        # Convert to numpy array
        path_array = np.array(path).T

        # Create spline
        tck, u = splprep([path_array[0], path_array[1]], s=0)

        # Evaluate at more points for smoothness
        u_new = np.linspace(0, 1, len(path) * 10)
        smoothed = splev(u_new, tck)

        return list(zip(smoothed[0], smoothed[1]))

    def calculate_time_allocation(self, path):
        """Calculate time allocation for each segment"""
        time_allocation = []
        total_time = 0.0

        for i in range(1, len(path)):
            distance = self.euclidean_distance(path[i-1], path[i])
            # Assume constant velocity for time estimation
            segment_time = distance / self.max_velocity
            total_time += segment_time
            time_allocation.append(segment_time)

        return time_allocation

    def generate_velocity_profile(self, path, time_allocation, initial_vel, final_vel):
        """Generate velocity profile with acceleration constraints"""
        velocities = [initial_vel]

        for i in range(len(path) - 1):
            # Calculate required acceleration to reach next point
            distance = self.euclidean_distance(path[i], path[i+1])
            time = time_allocation[i]

            # Try to maintain max velocity while respecting acceleration limits
            required_vel = min(self.max_velocity, distance / time if time > 0 else self.max_velocity)

            # Apply acceleration limits
            current_vel = velocities[-1]
            max_change = self.max_acceleration * time

            if required_vel > current_vel:
                new_vel = min(required_vel, current_vel + max_change)
            else:
                new_vel = max(required_vel, current_vel - max_change)

            velocities.append(new_vel)

        # Ensure final velocity constraint
        if len(velocities) > 1:
            velocities[-1] = final_vel

        return velocities

    def create_trajectory(self, path, velocities):
        """Create complete trajectory with position, velocity, and acceleration"""
        trajectory = []

        for i, (pos, vel) in enumerate(zip(path, velocities)):
            # Calculate acceleration
            acc = 0.0
            if i > 0:
                dt = 1.0 / self.control_rate  # Assuming constant time step
                acc = (velocities[i] - velocities[i-1]) / dt if dt > 0 else 0.0

            trajectory_point = {
                'position': pos,
                'velocity': vel,
                'acceleration': acc,
                'time': i / self.control_rate
            }
            trajectory.append(trajectory_point)

        return trajectory

    def euclidean_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
```

## Classical Planning Approaches

### Sampling-Based Methods

#### RRT (Rapidly-exploring Random Trees)
Probabilistically complete planning algorithm:

```python
import numpy as np
import random

class RRTPlanner:
    def __init__(self, bounds, step_size=0.5):
        self.bounds = bounds  # [(min_x, max_x), (min_y, max_y)]
        self.step_size = step_size
        self.tree = []
        self.parent = {}

    def plan(self, start, goal, obstacles, max_iterations=10000):
        """RRT planning algorithm"""
        self.tree = [start]
        self.parent[start] = None

        for iteration in range(max_iterations):
            # Sample random point
            if random.random() < 0.05:  # 5% chance to sample goal
                q_rand = goal
            else:
                q_rand = self.sample_free_space(obstacles)

            # Find nearest node in tree
            q_near = self.nearest_node(q_rand)

            # Extend towards random point
            q_new = self.extend(q_near, q_rand)

            # Check collision
            if not self.check_collision(q_near, q_new, obstacles):
                self.tree.append(q_new)
                self.parent[q_new] = q_near

                # Check if goal is reached
                if self.distance(q_new, goal) < self.step_size:
                    return self.extract_path(goal)

        return None  # No path found

    def sample_free_space(self, obstacles):
        """Sample a random point in free space"""
        while True:
            x = random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = random.uniform(self.bounds[1][0], self.bounds[1][1])
            point = (x, y)

            if not self.in_obstacle(point, obstacles):
                return point

    def nearest_node(self, point):
        """Find nearest node in tree to given point"""
        nearest = self.tree[0]
        min_dist = self.distance(nearest, point)

        for node in self.tree[1:]:
            dist = self.distance(node, point)
            if dist < min_dist:
                min_dist = dist
                nearest = node

        return nearest

    def extend(self, start, goal):
        """Extend from start towards goal by step size"""
        direction = np.array(goal) - np.array(start)
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return goal

        # Normalize direction and scale by step size
        direction = direction / distance * self.step_size
        new_point = tuple(np.array(start) + direction)

        return new_point

    def extract_path(self, goal):
        """Extract path from tree to goal"""
        path = []
        current = self.nearest_node(goal)  # Find node closest to goal

        while current is not None:
            path.append(current)
            current = self.parent[current]

        return path[::-1]  # Reverse to get start-to-goal path

    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def check_collision(self, p1, p2, obstacles):
        """Check if line segment p1-p2 collides with obstacles"""
        # Simple implementation: check multiple points along the segment
        steps = max(int(self.distance(p1, p2) / 0.1), 10)

        for i in range(steps + 1):
            t = i / steps
            point = (
                p1[0] + t * (p2[0] - p1[0]),
                p1[1] + t * (p2[1] - p1[1])
            )

            if self.in_obstacle(point, obstacles):
                return True

        return False

    def in_obstacle(self, point, obstacles):
        """Check if point is inside any obstacle"""
        for obstacle in obstacles:
            if (obstacle['min_x'] <= point[0] <= obstacle['max_x'] and
                obstacle['min_y'] <= point[1] <= obstacle['max_y']):
                return True
        return False
```

#### RRT*
Asymptotically optimal RRT variant:

```python
class RRTStarPlanner(RRTPlanner):
    def __init__(self, bounds, step_size=0.5, radius_factor=1.1):
        super().__init__(bounds, step_size)
        self.radius_factor = radius_factor
        self.costs = {}

    def plan(self, start, goal, obstacles, max_iterations=10000):
        """RRT* planning algorithm with optimality"""
        self.tree = [start]
        self.parent[start] = None
        self.costs[start] = 0

        for iteration in range(max_iterations):
            # Sample random point
            if random.random() < 0.05:  # 5% chance to sample goal
                q_rand = goal
            else:
                q_rand = self.sample_free_space(obstacles)

            # Find nearest node in tree
            q_near = self.nearest_node(q_rand)

            # Extend towards random point
            q_new = self.extend(q_near, q_rand)

            # Check collision
            if not self.check_collision(q_near, q_new, obstacles):
                # Find neighbors within radius
                neighbors = self.find_neighbors(q_new)

                # Select parent with minimum cost
                min_cost = float('inf')
                best_parent = q_near

                for neighbor in neighbors:
                    if not self.check_collision(neighbor, q_new, obstacles):
                        cost = self.costs[neighbor] + self.distance(neighbor, q_new)
                        if cost < min_cost:
                            min_cost = cost
                            best_parent = neighbor

                # Add new node to tree
                self.tree.append(q_new)
                self.parent[q_new] = best_parent
                self.costs[q_new] = self.costs[best_parent] + self.distance(best_parent, q_new)

                # Rewire tree
                self.rewire(q_new, neighbors, obstacles)

                # Check if goal is reached
                if self.distance(q_new, goal) < self.step_size:
                    return self.extract_path(goal)

        return None  # No path found

    def find_neighbors(self, point):
        """Find neighbors within a radius"""
        radius = self.radius_factor * self.step_size * (np.log(len(self.tree)) / len(self.tree)) ** (1/2)
        neighbors = []

        for node in self.tree:
            if self.distance(node, point) <= radius:
                neighbors.append(node)

        return neighbors

    def rewire(self, new_node, neighbors, obstacles):
        """Rewire tree to improve path quality"""
        for neighbor in neighbors:
            if (neighbor != self.parent[new_node] and
                not self.check_collision(new_node, neighbor, obstacles)):

                new_cost = self.costs[new_node] + self.distance(new_node, neighbor)

                if new_cost < self.costs[neighbor]:
                    self.parent[neighbor] = new_node
                    self.costs[neighbor] = new_cost
```

#### PRM (Probabilistic Roadmap)
Multi-query planning approach:

```python
class PRMPlanner:
    def __init__(self, bounds, num_samples=1000):
        self.bounds = bounds
        self.num_samples = num_samples
        self.roadmap = {}
        self.samples = []

    def build_roadmap(self, obstacles, connection_radius=1.0):
        """Build roadmap by sampling and connecting configurations"""
        # Sample free configurations
        self.samples = []
        while len(self.samples) < self.num_samples:
            sample = self.sample_free_space(obstacles)
            if sample:
                self.samples.append(sample)

        # Connect nearby configurations
        self.roadmap = {sample: [] for sample in self.samples}

        for i, sample in enumerate(self.samples):
            for j, other_sample in enumerate(self.samples[i+1:], i+1):
                if self.distance(sample, other_sample) <= connection_radius:
                    if not self.check_collision(sample, other_sample, obstacles):
                        self.roadmap[sample].append(other_sample)
                        self.roadmap[other_sample].append(sample)

    def plan(self, start, goal, obstacles):
        """Plan using pre-built roadmap"""
        # Connect start and goal to roadmap
        start_neighbors = self.find_nearby(start, obstacles)
        goal_neighbors = self.find_nearby(goal, obstacles)

        # Temporarily add start and goal to roadmap
        temp_roadmap = self.roadmap.copy()
        temp_roadmap[start] = start_neighbors
        for neighbor in start_neighbors:
            temp_roadmap[neighbor].append(start)

        temp_roadmap[goal] = goal_neighbors
        for neighbor in goal_neighbors:
            temp_roadmap[neighbor].append(goal)

        # Run shortest path algorithm (Dijkstra's)
        path = self.dijkstra_path(start, goal, temp_roadmap)

        return path

    def find_nearby(self, point, obstacles, k=5):
        """Find k nearest samples that can be connected"""
        distances = [(self.distance(point, sample), sample) for sample in self.samples]
        distances.sort()

        nearby = []
        for dist, sample in distances:
            if len(nearby) >= k:
                break
            if not self.check_collision(point, sample, obstacles):
                nearby.append(sample)

        return nearby

    def dijkstra_path(self, start, goal, roadmap):
        """Find shortest path using Dijkstra's algorithm"""
        import heapq

        distances = {node: float('inf') for node in roadmap}
        previous = {node: None for node in roadmap}
        distances[start] = 0

        pq = [(0, start)]

        while pq:
            current_dist, current = heapq.heappop(pq)

            if current == goal:
                break

            if current_dist > distances[current]:
                continue

            for neighbor in roadmap[current]:
                distance = self.distance(current, neighbor)
                new_dist = distances[current] + distance

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = previous[current]

        return path[::-1] if path[-1] == start else None
```

### Grid-Based Methods

#### A* Algorithm
Optimal pathfinding on discrete grids:

```python
class AStarGridPlanner:
    def __init__(self, grid_resolution=0.1):
        self.grid_resolution = grid_resolution

    def plan(self, start, goal, occupancy_grid):
        """A* path planning on grid map"""
        # Convert continuous coordinates to grid indices
        start_idx = self.world_to_grid(start)
        goal_idx = self.world_to_grid(goal)

        # Initialize data structures
        open_set = [(0, start_idx)]  # (f_score, position)
        heapq.heapify(open_set)

        came_from = {}
        g_score = {start_idx: 0}
        f_score = {start_idx: self.heuristic(start_idx, goal_idx)}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == goal_idx:
                return self.reconstruct_path(came_from, current)

            # Check all 8 neighbors
            for neighbor in self.get_8_connected_neighbors(current):
                if self.is_out_of_bounds(neighbor, occupancy_grid):
                    continue

                if self.is_occupied(neighbor, occupancy_grid):
                    continue

                tentative_g = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_idx)

                    # Add to open set if not already there
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def heuristic(self, pos1, pos2):
        """Manhattan distance heuristic (admissible)"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_8_connected_neighbors(self, pos):
        """Get 8-connected neighbors"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((pos[0] + dx, pos[1] + dy))
        return neighbors

    def world_to_grid(self, world_coords):
        """Convert world coordinates to grid indices"""
        return (int(world_coords[0] / self.grid_resolution),
                int(world_coords[1] / self.grid_resolution))

    def is_occupied(self, grid_pos, occupancy_grid):
        """Check if grid position is occupied"""
        x, y = grid_pos
        if x < 0 or y < 0 or x >= occupancy_grid.shape[0] or y >= occupancy_grid.shape[1]:
            return True
        return occupancy_grid[x, y] > 0.5  # Occupancy threshold

    def is_out_of_bounds(self, grid_pos, occupancy_grid):
        """Check if grid position is out of bounds"""
        x, y = grid_pos
        return x < 0 or y < 0 or x >= occupancy_grid.shape[0] or y >= occupancy_grid.shape[1]

    def distance(self, pos1, pos2):
        """Euclidean distance between grid positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
```

#### D* Algorithm
Dynamic replanning for changing environments:

```python
class DStarPlanner:
    def __init__(self, grid_resolution=0.1):
        self.grid_resolution = grid_resolution
        self.open_list = []
        self.rhs = {}  # Right-hand side values
        self.g = {}    # g-values
        self.priority_queue = []

    def initialize(self, start, goal, occupancy_grid):
        """Initialize D* algorithm"""
        self.start = self.world_to_grid(start)
        self.goal = self.world_to_grid(goal)
        self.occupancy_grid = occupancy_grid

        # Initialize all states
        for x in range(occupancy_grid.shape[0]):
            for y in range(occupancy_grid.shape[1]):
                pos = (x, y)
                self.rhs[pos] = float('inf')
                self.g[pos] = float('inf')

        # Set goal state
        self.rhs[self.goal] = 0
        self.update_vertex(self.goal)

    def compute_shortest_path(self):
        """Compute shortest path using D*"""
        while self.top_key() < self.calculate_key(self.start) or self.rhs[self.start] > self.g[self.start]:
            k_old = self.top_key()
            u = self.pop_from_open()

            if k_old < self.calculate_key(u):
                self.insert_to_open(u)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                # Update successors
                for s in self.get_neighbors(u):
                    self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u):
                    self.update_vertex(s)

    def update_vertex(self, u):
        """Update vertex u"""
        if u != self.goal:
            self.rhs[u] = min([self.c(u, s) + self.g[s] for s in self.get_neighbors(u)])

        if self.g[u] != self.rhs[u]:
            self.insert_to_open(u)

    def c(self, u, v):
        """Cost function between u and v"""
        if self.is_occupied(u, self.occupancy_grid) or self.is_occupied(v, self.occupancy_grid):
            return float('inf')
        return self.distance(u, v)

    def get_neighbors(self, pos):
        """Get 8-connected neighbors"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (pos[0] + dx, pos[1] + dy)
                if not self.is_out_of_bounds(neighbor, self.occupancy_grid):
                    neighbors.append(neighbor)
        return neighbors

    def calculate_key(self, s):
        """Calculate priority key for state s"""
        return (min(self.g[s], self.rhs[s]) + self.heuristic(s, self.start),
                min(self.g[s], self.rhs[s]))

    def heuristic(self, pos1, pos2):
        """Heuristic function"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def top_key(self):
        """Get top key from open list"""
        if not self.priority_queue:
            return (float('inf'), float('inf'))
        return self.priority_queue[0][0]

    def pop_from_open(self):
        """Pop from open list"""
        return heapq.heappop(self.priority_queue)[1]

    def insert_to_open(self, s):
        """Insert state s into open list"""
        key = self.calculate_key(s)
        heapq.heappush(self.priority_queue, (key, s))

    def world_to_grid(self, world_coords):
        """Convert world coordinates to grid indices"""
        return (int(world_coords[0] / self.grid_resolution),
                int(world_coords[1] / self.grid_resolution))

    def is_occupied(self, grid_pos, occupancy_grid):
        """Check if grid position is occupied"""
        x, y = grid_pos
        if x < 0 or y < 0 or x >= occupancy_grid.shape[0] or y >= occupancy_grid.shape[1]:
            return True
        return occupancy_grid[x, y] > 0.5

    def is_out_of_bounds(self, grid_pos, occupancy_grid):
        """Check if grid position is out of bounds"""
        x, y = grid_pos
        return x < 0 or y < 0 or x >= occupancy_grid.shape[0] or y >= occupancy_grid.shape[1]

    def distance(self, pos1, pos2):
        """Euclidean distance between grid positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
```

### Optimization-Based Methods

#### CHOMP (Covariant Hamiltonian Optimization for Motion Planning)
Trajectory optimization approach:

```python
class CHOMPPlanner:
    def __init__(self, max_iterations=200, learning_rate=0.01):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.obstacle_threshold = 0.5

    def plan(self, start, goal, path_initial, occupancy_grid, obstacles):
        """CHOMP trajectory optimization"""
        # Initialize trajectory from initial path
        trajectory = np.array(path_initial)

        for iteration in range(self.max_iterations):
            # Compute gradient of cost function
            gradient = self.compute_gradient(trajectory, occupancy_grid, obstacles)

            # Update trajectory
            trajectory = trajectory - self.learning_rate * gradient

            # Project to feasible space if needed
            trajectory = self.project_to_feasible(trajectory, obstacles)

            # Check convergence
            if np.linalg.norm(gradient) < 1e-6:
                break

        return trajectory.tolist()

    def compute_gradient(self, trajectory, occupancy_grid, obstacles):
        """Compute gradient of cost function"""
        gradient = np.zeros_like(trajectory)

        # Smoothness cost gradient
        smoothness_grad = self.compute_smoothness_gradient(trajectory)

        # Obstacle cost gradient
        obstacle_grad = self.compute_obstacle_gradient(trajectory, occupancy_grid, obstacles)

        # Combine gradients
        gradient = smoothness_grad + obstacle_grad

        return gradient

    def compute_smoothness_gradient(self, trajectory):
        """Compute gradient for smoothness cost"""
        grad = np.zeros_like(trajectory)

        # Gradient for second-order smoothness (minimize curvature)
        for i in range(1, len(trajectory) - 1):
            grad[i] = 2 * trajectory[i] - trajectory[i-1] - trajectory[i+1]

        return grad

    def compute_obstacle_gradient(self, trajectory, occupancy_grid, obstacles):
        """Compute gradient for obstacle avoidance"""
        grad = np.zeros_like(trajectory)

        for i, point in enumerate(trajectory):
            # Get obstacle forces at this point
            obstacle_force = self.compute_obstacle_force(point, occupancy_grid, obstacles)
            grad[i] = obstacle_force

        return grad

    def compute_obstacle_force(self, point, occupancy_grid, obstacles):
        """Compute repulsive force from obstacles"""
        force = np.array([0.0, 0.0])

        # For each obstacle, compute repulsive force
        for obstacle in obstacles:
            obs_center = np.array([(obstacle['min_x'] + obstacle['max_x']) / 2,
                                   (obstacle['min_y'] + obstacle['max_y']) / 2])
            distance = np.linalg.norm(point - obs_center)

            if distance < self.obstacle_threshold:
                # Repulsive force pointing away from obstacle
                direction = point - obs_center
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    force += direction * (1.0 / distance - 1.0 / self.obstacle_threshold)

        return force

    def project_to_feasible(self, trajectory, obstacles):
        """Project trajectory to feasible space"""
        # Ensure trajectory stays within bounds and avoids obstacles
        for i, point in enumerate(trajectory):
            # Simple projection: move away from obstacles if too close
            for obstacle in obstacles:
                if (obstacle['min_x'] <= point[0] <= obstacle['max_x'] and
                    obstacle['min_y'] <= point[1] <= obstacle['max_y']):
                    # Point is inside obstacle, move it outside
                    center = np.array([(obstacle['min_x'] + obstacle['max_x']) / 2,
                                       (obstacle['min_y'] + obstacle['max_y']) / 2])
                    direction = point - center
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        margin = 0.1  # Safety margin
                        new_point = center + direction * (self.obstacle_threshold + margin)
                        trajectory[i] = new_point

        return trajectory
```

## Decision-Making Frameworks

### Finite State Machines (FSM)

#### Basic FSM Implementation
```python
class StateMachine:
    def __init__(self):
        self.states = {}
        self.current_state = None
        self.transitions = {}

    def add_state(self, name, state_obj):
        """Add a state to the FSM"""
        self.states[name] = state_obj

    def add_transition(self, from_state, to_state, condition):
        """Add a transition between states"""
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        self.transitions[from_state].append({
            'to': to_state,
            'condition': condition
        })

    def set_initial_state(self, state_name):
        """Set the initial state"""
        self.current_state = state_name

    def update(self, sensor_data):
        """Update the FSM based on sensor data"""
        if self.current_state is None:
            return

        # Execute current state
        self.states[self.current_state].execute(sensor_data)

        # Check for transitions
        if self.current_state in self.transitions:
            for transition in self.transitions[self.current_state]:
                if transition['condition'](sensor_data):
                    self.current_state = transition['to']
                    self.states[self.current_state].enter()
                    break

class FSMState:
    def __init__(self, name):
        self.name = name

    def execute(self, sensor_data):
        """Execute state-specific logic"""
        pass

    def enter(self):
        """Called when entering the state"""
        pass

    def exit(self):
        """Called when exiting the state"""
        pass

# Example: Robot navigation FSM
class NavigationFSM(StateMachine):
    def __init__(self):
        super().__init__()

        # Define states
        self.add_state('explore', ExploreState('explore'))
        self.add_state('navigate', NavigateState('navigate'))
        self.add_state('avoid_obstacle', AvoidObstacleState('avoid_obstacle'))
        self.add_state('reach_goal', ReachGoalState('reach_goal'))

        # Define transitions
        self.add_transition('explore', 'navigate',
                           lambda data: data.get('goal_set', False))
        self.add_transition('navigate', 'avoid_obstacle',
                           lambda data: data.get('obstacle_detected', False))
        self.add_transition('avoid_obstacle', 'navigate',
                           lambda data: not data.get('obstacle_detected', False))
        self.add_transition('navigate', 'reach_goal',
                           lambda data: data.get('goal_reached', False))

        self.set_initial_state('explore')

class ExploreState(FSMState):
    def execute(self, sensor_data):
        # Random walk or systematic exploration
        print("Exploring environment...")

    def enter(self):
        print("Entering exploration mode")

class NavigateState(FSMState):
    def execute(self, sensor_data):
        # Follow planned path to goal
        print("Navigating to goal...")

    def enter(self):
        print("Starting navigation to goal")
```

#### Hierarchical FSM
Nested states for complex behaviors:

```python
class HierarchicalStateMachine:
    def __init__(self):
        self.root_state = CompositeState("root")
        self.current_state = self.root_state

    def add_substate(self, parent_state_name, substate):
        """Add a substate to a composite state"""
        parent = self.find_state(parent_state_name)
        if parent and isinstance(parent, CompositeState):
            parent.add_substate(substate)

    def find_state(self, state_name):
        """Find state by name (recursive search)"""
        return self.root_state.find_state_recursive(state_name)

    def update(self, context):
        """Update the hierarchical FSM"""
        self.current_state.execute(context)

class CompositeState:
    def __init__(self, name):
        self.name = name
        self.substates = {}
        self.active_substate = None
        self.transitions = {}

    def add_substate(self, state):
        """Add a substate"""
        self.substates[state.name] = state

    def execute(self, context):
        """Execute the active substate"""
        if self.active_substate:
            self.active_substate.execute(context)

    def set_active_substate(self, state_name):
        """Set the active substate"""
        if state_name in self.substates:
            self.active_substate = self.substates[state_name]

    def find_state_recursive(self, name):
        """Recursively find a state by name"""
        if self.name == name:
            return self

        for substate in self.substates.values():
            if substate.name == name:
                return substate
            if hasattr(substate, 'find_state_recursive'):
                result = substate.find_state_recursive(name)
                if result:
                    return result
        return None

# Example: Hierarchical navigation system
class NavigationSystem:
    def __init__(self):
        self.hsm = HierarchicalStateMachine()

        # Create high-level states
        self.movement_state = CompositeState("movement")
        self.manipulation_state = CompositeState("manipulation")

        # Create movement substates
        self.idle_substate = SimpleState("idle")
        self.forward_substate = SimpleState("forward")
        self.turn_substate = SimpleState("turn")

        # Add substates to movement state
        self.movement_state.add_substate(self.idle_substate)
        self.movement_state.add_substate(self.forward_substate)
        self.movement_state.add_substate(self.turn_substate)

        # Add high-level states to root
        self.hsm.root_state.add_substate(self.movement_state)
        self.hsm.root_state.add_substate(self.manipulation_state)

        # Set initial active state
        self.movement_state.set_active_substate("idle")

class SimpleState:
    def __init__(self, name):
        self.name = name

    def execute(self, context):
        """Execute state-specific behavior"""
        print(f"Executing {self.name} state")
        # State-specific logic here
```

### Behavior Trees

#### Basic Behavior Tree Structure
```python
class BehaviorNode:
    def __init__(self, name):
        self.name = name
        self.children = []
        self.status = None

    def tick(self, blackboard):
        """Execute the behavior and return status"""
        pass

    def add_child(self, child):
        """Add a child node"""
        self.children.append(child)

class SequenceNode(BehaviorNode):
    """Execute children in sequence until one fails"""
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status == 'FAILURE':
                self.status = 'FAILURE'
                return 'FAILURE'
        self.status = 'SUCCESS'
        return 'SUCCESS'

class SelectorNode(BehaviorNode):
    """Execute children until one succeeds"""
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status == 'SUCCESS':
                self.status = 'SUCCESS'
                return 'SUCCESS'
        self.status = 'FAILURE'
        return 'FAILURE'

class DecoratorNode(BehaviorNode):
    """Modify behavior of a single child"""
    def __init__(self, name, child=None):
        super().__init__(name)
        if child:
            self.children = [child]

    def tick(self, blackboard):
        if self.children:
            return self.children[0].tick(blackboard)
        return 'FAILURE'

class InverterNode(DecoratorNode):
    """Invert the result of the child"""
    def tick(self, blackboard):
        if not self.children:
            return 'FAILURE'

        result = self.children[0].tick(blackboard)
        if result == 'SUCCESS':
            return 'FAILURE'
        elif result == 'FAILURE':
            return 'SUCCESS'
        else:
            return result

class ConditionNode(BehaviorNode):
    """Check a condition and return SUCCESS or FAILURE"""
    def __init__(self, name, condition_func):
        super().__init__(name)
        self.condition_func = condition_func

    def tick(self, blackboard):
        if self.condition_func(blackboard):
            self.status = 'SUCCESS'
            return 'SUCCESS'
        else:
            self.status = 'FAILURE'
            return 'FAILURE'

class ActionNode(BehaviorNode):
    """Perform an action and return status"""
    def __init__(self, name, action_func):
        super().__init__(name)
        self.action_func = action_func

    def tick(self, blackboard):
        result = self.action_func(blackboard)
        self.status = result
        return result

# Example: Robot navigation behavior tree
def create_navigation_behavior_tree():
    """Create a behavior tree for robot navigation"""
    root = SelectorNode("root")

    # Emergency behaviors (higher priority)
    emergency_sequence = SequenceNode("emergency_check")
    emergency_sequence.add_child(ConditionNode("check_emergency",
                                             lambda bb: bb.get('emergency', False)))
    emergency_sequence.add_child(ActionNode("emergency_stop",
                                          lambda bb: emergency_stop(bb)))

    # Normal navigation behaviors
    navigation_sequence = SequenceNode("navigation")
    navigation_sequence.add_child(ConditionNode("goal_exists",
                                              lambda bb: bb.get('goal_set', False)))
    navigation_sequence.add_child(ActionNode("navigate_to_goal",
                                           lambda bb: navigate_to_goal(bb)))

    # Exploration behaviors
    exploration_sequence = SequenceNode("exploration")
    exploration_sequence.add_child(ConditionNode("explore_mode",
                                               lambda bb: bb.get('explore', False)))
    exploration_sequence.add_child(ActionNode("explore_environment",
                                            lambda bb: explore_environment(bb)))

    # Idle behavior
    idle_action = ActionNode("idle", lambda bb: idle(bb))

    # Build the tree
    root.add_child(emergency_sequence)
    root.add_child(navigation_sequence)
    root.add_child(exploration_sequence)
    root.add_child(idde_action)

    return root

def emergency_stop(blackboard):
    """Stop robot in emergency situation"""
    print("EMERGENCY STOP!")
    return 'SUCCESS'

def navigate_to_goal(blackboard):
    """Navigate to set goal"""
    goal = blackboard.get('goal')
    if goal:
        print(f"Navigating to goal: {goal}")
        # Navigation logic here
        return 'SUCCESS'
    return 'FAILURE'

def explore_environment(blackboard):
    """Explore the environment"""
    print("Exploring environment...")
    # Exploration logic here
    return 'RUNNING'  # Or SUCCESS if exploration is complete

def idle(blackboard):
    """Idle behavior"""
    print("Robot is idle")
    return 'SUCCESS'
```

#### Advanced Behavior Tree Features
```python
class ParallelNode(BehaviorNode):
    """Execute all children in parallel"""
    def __init__(self, name, success_policy='ALL', failure_policy='ONE'):
        super().__init__(name)
        self.success_policy = success_policy  # 'ALL' or 'ONE'
        self.failure_policy = failure_policy  # 'ALL' or 'ONE'

    def tick(self, blackboard):
        success_count = 0
        failure_count = 0
        running_count = 0

        for child in self.children:
            status = child.tick(blackboard)
            if status == 'SUCCESS':
                success_count += 1
            elif status == 'FAILURE':
                failure_count += 1
            else:
                running_count += 1

        # Check success condition
        if self.success_policy == 'ALL' and success_count == len(self.children):
            return 'SUCCESS'
        elif self.success_policy == 'ONE' and success_count > 0:
            return 'SUCCESS'

        # Check failure condition
        if self.failure_policy == 'ALL' and failure_count == len(self.children):
            return 'FAILURE'
        elif self.failure_policy == 'ONE' and failure_count > 0:
            return 'FAILURE'

        # Some are running
        if running_count > 0:
            return 'RUNNING'

        return 'RUNNING'

class Blackboard:
    """Shared memory for behavior tree"""
    def __init__(self):
        self.memory = {}

    def set(self, key, value):
        """Set a value in blackboard"""
        self.memory[key] = value

    def get(self, key, default=None):
        """Get a value from blackboard"""
        return self.memory.get(key, default)

    def update_sensor_data(self, sensor_data):
        """Update blackboard with sensor data"""
        for key, value in sensor_data.items():
            self.memory[key] = value
```

### Decision Networks

#### Markov Decision Process (MDP)
Sequential decision making under uncertainty:

```python
import numpy as np

class MDP:
    def __init__(self, states, actions, transition_probs, rewards, discount_factor=0.9):
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs  # P(s'|s,a)
        self.rewards = rewards  # R(s,a,s')
        self.discount_factor = discount_factor

    def value_iteration(self, max_iterations=1000, tolerance=1e-6):
        """Value iteration algorithm to find optimal policy"""
        # Initialize value function
        V = {s: 0.0 for s in self.states}

        for iteration in range(max_iterations):
            V_new = V.copy()

            for s in self.states:
                # Calculate value for each action
                action_values = []
                for a in self.actions:
                    value = 0.0
                    for s_prime in self.states:
                        prob = self.transition_probs.get((s, a, s_prime), 0.0)
                        reward = self.rewards.get((s, a, s_prime), 0.0)
                        value += prob * (reward + self.discount_factor * V[s_prime])
                    action_values.append(value)

                # Take maximum value
                V_new[s] = max(action_values) if action_values else 0.0

            # Check for convergence
            max_diff = max(abs(V_new[s] - V[s]) for s in self.states)
            if max_diff < tolerance:
                break

            V = V_new

        # Extract policy
        policy = {}
        for s in self.states:
            action_values = []
            for a in self.actions:
                value = 0.0
                for s_prime in self.states:
                    prob = self.transition_probs.get((s, a, s_prime), 0.0)
                    reward = self.rewards.get((s, a, s_prime), 0.0)
                    value += prob * (reward + self.discount_factor * V[s_prime])
                action_values.append((a, value))

            # Choose action with maximum value
            policy[s] = max(action_values, key=lambda x: x[1])[0]

        return policy, V

class RobotMDP(MDP):
    def __init__(self, grid_size, obstacles=None):
        # Define states (grid positions)
        states = [(x, y) for x in range(grid_size[0]) for y in range(grid_size[1])]

        # Define actions (move in 4 directions)
        actions = ['up', 'down', 'left', 'right']

        # Initialize transition probabilities and rewards
        transition_probs = {}
        rewards = {}

        for s in states:
            for a in actions:
                for s_prime in states:
                    # Calculate transition probability
                    intended_next = self.get_intended_next(s, a)
                    prob = self.calculate_transition_prob(s, a, s_prime, intended_next)
                    transition_probs[(s, a, s_prime)] = prob

                    # Calculate reward
                    reward = self.calculate_reward(s, a, s_prime, obstacles)
                    rewards[(s, a, s_prime)] = reward

        super().__init__(states, actions, transition_probs, rewards)

    def get_intended_next(self, state, action):
        """Get intended next state based on action"""
        x, y = state
        if action == 'up':
            return (x, y + 1)
        elif action == 'down':
            return (x, y - 1)
        elif action == 'left':
            return (x - 1, y)
        elif action == 'right':
            return (x + 1, y)
        return state

    def calculate_transition_prob(self, s, a, s_prime, intended_next):
        """Calculate transition probability with stochasticity"""
        if s_prime == intended_next:
            return 0.8  # 80% chance of going where intended
        elif s_prime == s:  # Bumped into wall/obstacle
            return 0.2  # 20% chance of staying in place
        else:
            return 0.0

    def calculate_reward(self, s, a, s_prime, obstacles):
        """Calculate reward for transition"""
        if obstacles and s_prime in obstacles:
            return -10  # Large negative reward for hitting obstacle
        elif s_prime == (self.grid_size[0]-1, self.grid_size[1]-1):  # Goal
            return 100  # Large positive reward for reaching goal
        else:
            return -1  # Small negative reward for each step (encourage efficiency)
```

#### Partially Observable MDP (POMDP)
Decision making with incomplete information:

```python
class POMDP:
    def __init__(self, states, actions, observations, transition_probs,
                 observation_probs, rewards, discount_factor=0.9):
        self.states = states
        self.actions = actions
        self.observations = observations
        self.transition_probs = transition_probs  # P(s'|s,a)
        self.observation_probs = observation_probs  # P(o|s',a)
        self.rewards = rewards  # R(s,a,s')
        self.discount_factor = discount_factor

    def update_belief_state(self, belief, action, observation):
        """Update belief state using Bayes rule"""
        new_belief = {}

        # Calculate P(s'|o,a,belief) = P(o|s',a) * sum_s P(s'|s,a) * b(s)
        for s_prime in self.states:
            # P(o|s',a)
            obs_prob = self.observation_probs.get((s_prime, action, observation), 0.0)

            # sum_s P(s'|s,a) * b(s)
            state_transition_sum = 0.0
            for s in self.states:
                trans_prob = self.transition_probs.get((s, action, s_prime), 0.0)
                state_prob = belief.get(s, 0.0)
                state_transition_sum += trans_prob * state_prob

            new_belief[s_prime] = obs_prob * state_transition_sum

        # Normalize
        total = sum(new_belief.values())
        if total > 0:
            for s in new_belief:
                new_belief[s] /= total

        return new_belief

class RobotPOMDP(POMDP):
    def __init__(self, grid_size):
        states = [(x, y) for x in range(grid_size[0]) for y in range(grid_size[1])]
        actions = ['up', 'down', 'left', 'right']
        observations = ['clear', 'obstacle_left', 'obstacle_right', 'obstacle_front']

        # Initialize probabilities (simplified for example)
        transition_probs = self.initialize_transition_probs(states, actions)
        observation_probs = self.initialize_observation_probs(states, actions, observations)
        rewards = self.initialize_rewards(states, actions)

        super().__init__(states, actions, observations, transition_probs,
                        observation_probs, rewards)

    def initialize_transition_probs(self, states, actions):
        """Initialize transition probabilities"""
        probs = {}
        for s in states:
            for a in actions:
                intended_next = self.get_intended_next(s, a)
                for s_prime in states:
                    if s_prime == intended_next:
                        probs[(s, a, s_prime)] = 0.8
                    elif s_prime == s:
                        probs[(s, a, s_prime)] = 0.2
                    else:
                        probs[(s, a, s_prime)] = 0.0
        return probs

    def initialize_observation_probs(self, states, actions, observations):
        """Initialize observation probabilities"""
        probs = {}
        for s in states:
            for a in actions:
                for o in observations:
                    # Simplified observation model
                    if o == 'clear':
                        probs[(s, a, o)] = 0.9 if not self.near_obstacle(s) else 0.1
                    else:
                        probs[(s, a, o)] = 0.1 if not self.near_obstacle(s) else 0.9
        return probs

    def initialize_rewards(self, states, actions):
        """Initialize rewards"""
        rewards = {}
        goal_state = (len(set(s[0] for s in states)) - 1, len(set(s[1] for s in states)) - 1)

        for s in states:
            for a in actions:
                for s_prime in states:
                    if s_prime == goal_state:
                        rewards[(s, a, s_prime)] = 100
                    else:
                        rewards[(s, a, s_prime)] = -1
        return rewards

    def near_obstacle(self, state):
        """Check if state is near an obstacle (simplified)"""
        # This would be based on actual environment
        return False
```

## Learning-Based Planning

### Reinforcement Learning

#### Q-Learning Implementation
```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1,
                 discount_factor=0.95, epsilon=0.1):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Initialize Q-table
        self.q_table = {}
        for state in state_space:
            self.q_table[state] = {action: 0.0 for action in action_space}

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(self.action_space)
        else:
            # Exploit: best known action
            q_values = self.q_table[state]
            return max(q_values, key=q_values.get)

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        current_q = self.q_table[state][action]

        # Get maximum Q-value for next state
        next_max_q = max(self.q_table[next_state].values())

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )

        self.q_table[state][action] = new_q

class RobotQLearning:
    def __init__(self, grid_size, learning_rate=0.1, discount_factor=0.95,
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Define state and action spaces
        self.state_space = [(x, y) for x in range(grid_size[0]) for y in range(grid_size[1])]
        self.action_space = ['up', 'down', 'left', 'right']

        # Initialize Q-learning agent
        self.agent = QLearningAgent(
            self.state_space, self.action_space,
            learning_rate, discount_factor, epsilon
        )

        # Define goal and obstacles
        self.goal = (grid_size[0]-1, grid_size[1]-1)
        self.obstacles = set()

    def discretize_state(self, continuous_state):
        """Convert continuous state to discrete state"""
        x, y = continuous_state
        grid_x = max(0, min(self.grid_size[0]-1, int(x)))
        grid_y = max(0, min(self.grid_size[1]-1, int(y)))
        return (grid_x, grid_y)

    def get_reward(self, state, next_state, action):
        """Calculate reward for state transition"""
        if next_state in self.obstacles:
            return -10  # Penalty for hitting obstacle
        elif next_state == self.goal:
            return 100  # Reward for reaching goal
        else:
            # Negative reward for distance from goal (encourage efficiency)
            distance_to_goal = abs(next_state[0] - self.goal[0]) + abs(next_state[1] - self.goal[1])
            return -distance_to_goal * 0.1

    def train(self, episodes=1000):
        """Train the Q-learning agent"""
        for episode in range(episodes):
            # Reset to start position
            state = (0, 0)
            total_reward = 0

            while state != self.goal:
                # Choose action
                action = self.agent.choose_action(state)

                # Execute action and get next state
                next_state = self.take_action(state, action)

                # Get reward
                reward = self.get_reward(state, next_state, action)
                total_reward += reward

                # Update Q-value
                self.agent.update_q_value(state, action, reward, next_state)

                # Move to next state
                state = next_state

                # Break if stuck in loop
                if len(self.state_space) > 100 and total_reward < -1000:
                    break

            # Decay epsilon
            self.agent.epsilon = max(
                self.min_epsilon,
                self.agent.epsilon * self.epsilon_decay
            )

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.agent.epsilon:.3f}")

    def take_action(self, state, action):
        """Execute action and return next state"""
        x, y = state

        if action == 'up':
            new_y = min(self.grid_size[1] - 1, y + 1)
        elif action == 'down':
            new_y = max(0, y - 1)
        elif action == 'left':
            new_x = max(0, x - 1)
        elif action == 'right':
            new_x = min(self.grid_size[0] - 1, x + 1)
        else:
            return state  # Invalid action

        new_state = (new_x, new_y) if 'new_x' in locals() else (x, new_y)

        # Don't move into obstacles
        if new_state in self.obstacles:
            return state

        return new_state

    def get_optimal_policy(self):
        """Get the optimal policy from the trained Q-table"""
        policy = {}
        for state in self.state_space:
            q_values = self.agent.q_table[state]
            policy[state] = max(q_values, key=q_values.get)
        return policy
```

#### Deep Q-Network (DQN)
Using neural networks for complex state spaces:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Neural networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Update target network
        self.update_target_network()

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())

    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class RobotDQNPlanner:
    def __init__(self, state_size, action_size):
        self.agent = DQNAgent(state_size, action_size)
        self.state_size = state_size
        self.action_size = action_size

    def train(self, env, episodes=1000):
        """Train the DQN agent in the environment"""
        scores = deque(maxlen=100)

        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [self.state_size])
            total_reward = 0

            for time in range(500):  # Max steps per episode
                action = self.agent.act(state)
                next_state, reward, done, _ = env.step(action)
                next_state = np.reshape(next_state, [self.state_size])

                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

                if len(self.agent.memory) > 32:
                    self.agent.replay()

            scores.append(total_reward)

            if e % 100 == 0:
                self.agent.update_target_network()
                print(f"episode: {e}, score: {total_reward}, e: {self.agent.epsilon:.2}")

        return scores
```

### Imitation Learning

#### Behavior Cloning
Learning from expert demonstrations:

```python
class BehaviorCloning:
    def __init__(self, state_size, action_size, learning_rate=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network
        self.policy_network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train(self, demonstrations, epochs=100, batch_size=32):
        """Train the policy using expert demonstrations"""
        states = torch.FloatTensor([demo[0] for demo in demonstrations]).to(self.device)
        actions = torch.FloatTensor([demo[1] for demo in demonstrations]).to(self.device)

        for epoch in range(epochs):
            # Shuffle the data
            indices = torch.randperm(len(states))
            states_shuffled = states[indices]
            actions_shuffled = actions[indices]

            # Process in batches
            for i in range(0, len(states), batch_size):
                batch_states = states_shuffled[i:i+batch_size]
                batch_actions = actions_shuffled[i:i+batch_size]

                # Forward pass
                predicted_actions = self.policy_network(batch_states)
                loss = self.criterion(predicted_actions, batch_actions)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict(self, state):
        """Predict action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy_network(state_tensor)
        return action.squeeze(0).cpu().numpy()

class ImitationLearningPlanner:
    def __init__(self, state_size, action_size):
        self.bc = BehaviorCloning(state_size, action_size)
        self.demonstrations = []

    def add_demonstration(self, state, action):
        """Add a state-action pair from expert demonstration"""
        self.demonstrations.append((state, action))

    def train_from_demonstrations(self, epochs=100):
        """Train the planner using collected demonstrations"""
        if len(self.demonstrations) < 10:
            print("Warning: Not enough demonstrations for training")
            return

        self.bc.train(self.demonstrations, epochs=epochs)

    def plan_action(self, current_state):
        """Plan action using trained policy"""
        return self.bc.predict(current_state)
```

## Multi-Robot Planning

### Coordination Strategies
```python
class MultiRobotPlanner:
    def __init__(self, num_robots, environment):
        self.num_robots = num_robots
        self.environment = environment
        self.robots = [self.create_robot(i) for i in range(num_robots)]

    def create_robot(self, robot_id):
        """Create a robot with planning capabilities"""
        return {
            'id': robot_id,
            'position': None,
            'goal': None,
            'planner': AStarGridPlanner(),
            'path': []
        }

    def centralized_planning(self, robot_goals):
        """Centralized planning for all robots"""
        all_paths = []

        for i, (robot, goal) in enumerate(zip(self.robots, robot_goals)):
            # Plan path for current robot
            path = robot['planner'].plan(robot['position'], goal, self.environment)

            # Check for conflicts with previously planned paths
            path = self.resolve_conflicts(path, all_paths, i)
            all_paths.append(path)

            robot['path'] = path
            robot['goal'] = goal

        return all_paths

    def resolve_conflicts(self, path, existing_paths, current_robot_idx):
        """Resolve conflicts between robot paths"""
        # Simple approach: delay if conflict detected
        for i, other_path in enumerate(existing_paths):
            if i != current_robot_idx:
                # Check for spatiotemporal conflicts
                conflict_time = self.find_conflict(path, other_path)
                if conflict_time is not None:
                    # Add delay to resolve conflict
                    path = self.add_delay(path, conflict_time)

        return path

    def find_conflict(self, path1, path2):
        """Find time of conflict between two paths"""
        min_len = min(len(path1), len(path2))

        for t in range(min_len):
            if path1[t] == path2[t]:  # Same location at same time
                return t
            # Check for head-on collisions
            if t > 0 and path1[t] == path2[t-1] and path1[t-1] == path2[t]:
                return t

        return None

    def add_delay(self, path, conflict_time):
        """Add delay to path to resolve conflict"""
        if conflict_time < len(path):
            delay_node = path[conflict_time-1] if conflict_time > 0 else path[0]
            return path[:conflict_time] + [delay_node] + path[conflict_time:]
        return path

    def decentralized_planning(self, robot_goals):
        """Decentralized planning with communication"""
        robot_paths = [None] * self.num_robots

        for i, (robot, goal) in enumerate(zip(self.robots, robot_goals)):
            # Plan path considering other robots' known paths
            path = self.plan_with_knowledge(robot['position'], goal, i, robot_paths)
            robot_paths[i] = path
            robot['path'] = path
            robot['goal'] = goal

        return robot_paths

    def plan_with_knowledge(self, start, goal, robot_idx, known_paths):
        """Plan path considering known paths of other robots"""
        # Create temporary occupancy grid with other robots' paths
        temp_grid = self.environment.copy()

        for i, path in enumerate(known_paths):
            if path is not None and i != robot_idx:
                # Mark other robots' paths as temporarily occupied
                for t, pos in enumerate(path):
                    if t < len(temp_grid) and pos[0] < len(temp_grid) and pos[1] < len(temp_grid[0]):
                        temp_grid[pos] = 1  # Mark as occupied

        # Plan path in modified environment
        return self.robots[robot_idx]['planner'].plan(start, goal, temp_grid)
```

## Integration with ROS 2

### Navigation Stack Integration
```python
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from action_msgs.msg import GoalStatus

class PlanningROS2Node(Node):
    def __init__(self):
        super().__init__('planning_node')

        # Action clients
        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

        # Publishers
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.goal_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10
        )

        # Planning components
        self.planner = AStarGridPlanner()
        self.current_map = None
        self.current_pose = None

    def plan_to_pose(self, goal_pose):
        """Plan path to goal pose"""
        if self.current_map is None or self.current_pose is None:
            self.get_logger().warn("Map or pose not available for planning")
            return False

        # Convert ROS PoseStamped to planning coordinates
        start = (self.current_pose.pose.position.x, self.current_pose.pose.position.y)
        goal = (goal_pose.pose.position.x, goal_pose.pose.position.y)

        # Plan path
        path = self.planner.plan(start, goal, self.current_map)

        if path:
            # Convert to ROS Path message
            ros_path = self.convert_to_ros_path(path)
            self.path_pub.publish(ros_path)

            # Send to navigation system
            return self.send_to_navigation(ros_path)

        return False

    def convert_to_ros_path(self, path):
        """Convert planning path to ROS Path message"""
        ros_path = Path()
        ros_path.header.stamp = self.get_clock().now().to_msg()
        ros_path.header.frame_id = 'map'

        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation

            ros_path.poses.append(pose)

        return ros_path

    def send_to_navigation(self, path):
        """Send path to navigation system"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = path.poses[-1]  # Use last pose as goal

        self.nav_to_pose_client.wait_for_server()
        future = self.nav_to_pose_client.send_goal_async(goal_msg)

        return future

    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_pose = msg.pose.pose

    def map_callback(self, msg):
        """Handle map updates"""
        # Convert OccupancyGrid to planning format
        self.current_map = self.ros_map_to_planning_format(msg)

    def ros_map_to_planning_format(self, ros_map):
        """Convert ROS occupancy grid to planning format"""
        width = ros_map.info.width
        height = ros_map.info.height
        resolution = ros_map.info.resolution

        # Reshape the data to 2D grid
        grid = np.array(ros_map.data).reshape((height, width))

        return grid
```

## Real-Time Considerations

### Efficient Planning Algorithms
```python
class RealTimePlanner:
    def __init__(self, max_planning_time=0.1):  # 100ms max planning time
        self.max_planning_time = max_planning_time

    def anytime_planning(self, start, goal, environment, time_budget):
        """Anytime planning within time budget"""
        start_time = time.time()

        # Quick initial solution
        initial_path = self.get_quick_path(start, goal, environment)

        best_path = initial_path
        best_cost = self.calculate_path_cost(initial_path) if initial_path else float('inf')

        # Continue improving if time allows
        while time.time() - start_time < time_budget:
            # Try to improve the current path
            improved_path = self.improve_path(best_path, start, goal, environment)

            if improved_path:
                new_cost = self.calculate_path_cost(improved_path)
                if new_cost < best_cost:
                    best_path = improved_path
                    best_cost = new_cost

        return best_path

    def get_quick_path(self, start, goal, environment):
        """Get a quick path using greedy or simple algorithm"""
        # Use a fast but potentially suboptimal algorithm
        return self.greedy_best_first_search(start, goal, environment)

    def improve_path(self, current_path, start, goal, environment):
        """Try to improve the current path"""
        # Local optimization: try to smooth the path
        return self.smooth_path(current_path, environment)

    def smooth_path(self, path, environment):
        """Smooth the path by removing unnecessary waypoints"""
        if len(path) < 3:
            return path

        smoothed_path = [path[0]]

        i = 0
        while i < len(path) - 1:
            # Try to connect current point to future points directly
            j = len(path) - 1
            while j > i:
                if self.is_line_clear(path[i], path[j], environment):
                    smoothed_path.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # No direct connection found, add next point
                i += 1
                if i < len(path):
                    smoothed_path.append(path[i])

        return smoothed_path

    def is_line_clear(self, start, end, environment):
        """Check if line between start and end is clear of obstacles"""
        # Bresenham's line algorithm or similar
        # Check multiple points along the line
        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
        if steps == 0:
            return True

        for t in range(steps + 1):
            t_norm = t / steps
            point_x = int(start[0] + t_norm * (end[0] - start[0]))
            point_y = int(start[1] + t_norm * (end[1] - start[1]))

            if self.is_occupied((point_x, point_y), environment):
                return False

        return True

    def calculate_path_cost(self, path):
        """Calculate the cost of a path"""
        if not path:
            return float('inf')

        total_cost = 0
        for i in range(1, len(path)):
            total_cost += self.distance(path[i-1], path[i])

        return total_cost
```

## Safety and Verification

### Safety-Constrained Planning
```python
class SafePlanner:
    def __init__(self):
        self.safety_margin = 0.5  # 50cm safety margin
        self.max_velocity = 1.0
        self.max_acceleration = 0.5

    def plan_with_safety_constraints(self, start, goal, environment, obstacles):
        """Plan path with safety constraints"""
        # Inflate obstacles by safety margin
        safe_environment = self.inflate_obstacles(environment, obstacles)

        # Plan path in safe environment
        path = self.plan_in_environment(start, goal, safe_environment)

        # Verify safety of planned path
        if self.verify_path_safety(path, obstacles):
            return path
        else:
            # Plan alternative path or return to safe state
            return self.plan_safe_alternative(start, goal, obstacles)

    def inflate_obstacles(self, environment, obstacles):
        """Inflate obstacles by safety margin"""
        inflated_obstacles = []

        for obstacle in obstacles:
            inflated = {
                'min_x': obstacle['min_x'] - self.safety_margin,
                'max_x': obstacle['max_x'] + self.safety_margin,
                'min_y': obstacle['min_y'] - self.safety_margin,
                'max_y': obstacle['max_y'] + self.safety_margin
            }
            inflated_obstacles.append(inflated)

        return inflated_obstacles

    def verify_path_safety(self, path, obstacles):
        """Verify that path maintains safety margins"""
        if not path:
            return False

        for point in path:
            for obstacle in obstacles:
                if (obstacle['min_x'] - self.safety_margin <= point[0] <= obstacle['max_x'] + self.safety_margin and
                    obstacle['min_y'] - self.safety_margin <= point[1] <= obstacle['max_y'] + self.safety_margin):
                    return False  # Path too close to obstacle

        return True

    def plan_safe_alternative(self, start, goal, obstacles):
        """Plan a safe alternative path"""
        # For example, return to a known safe location
        safe_zones = self.find_safe_zones(obstacles)
        if safe_zones:
            return self.plan_to_safe_zone(start, safe_zones[0])
        return None

    def find_safe_zones(self, obstacles):
        """Find zones that are guaranteed to be safe"""
        # This would depend on environment layout
        # For now, return some predefined safe zones
        return [(0, 0), (1, 1), (2, 2)]  # Example safe zones
```

## Best Practices for Planning Systems

### Performance Optimization
```python
class OptimizedPlanner:
    def __init__(self):
        self.planning_cache = {}
        self.last_environment = None
        self.last_path = None

    def plan_efficiently(self, start, goal, environment):
        """Plan efficiently using caching and incremental updates"""
        # Create environment hash for caching
        env_hash = hash(str(environment) + str(start) + str(goal))

        # Check cache first
        if env_hash in self.planning_cache:
            cached_path = self.planning_cache[env_hash]
            if self.is_path_still_valid(cached_path, environment):
                return cached_path

        # Plan new path
        path = self.plan(start, goal, environment)

        # Cache the result
        self.planning_cache[env_hash] = path

        return path

    def is_path_still_valid(self, path, environment):
        """Check if cached path is still valid for current environment"""
        if not path:
            return False

        # Check if obstacles have changed significantly along the path
        for point in path:
            if self.is_point_occupied(point, environment):
                return False

        return True

    def plan(self, start, goal, environment):
        """Plan path using appropriate algorithm based on environment characteristics"""
        # Choose algorithm based on environment size and complexity
        if self.is_small_environment(environment):
            return self.a_star_plan(start, goal, environment)
        else:
            return self.hierarchical_plan(start, goal, environment)

    def is_small_environment(self, environment):
        """Check if environment is small enough for global planning"""
        return environment.size < 1000  # Example threshold

    def a_star_plan(self, start, goal, environment):
        """Plan using A* for smaller environments"""
        planner = AStarGridPlanner()
        return planner.plan(start, goal, environment)

    def hierarchical_plan(self, start, goal, environment):
        """Plan using hierarchical approach for larger environments"""
        # Divide environment into regions
        regions = self.divide_into_regions(environment)

        # Plan high-level path between regions
        region_path = self.plan_region_path(start, goal, regions)

        # Plan detailed path within each region
        detailed_path = []
        for i, region in enumerate(region_path):
            if i == 0:
                region_start = start
            else:
                region_start = detailed_path[-1] if detailed_path else start

            if i == len(region_path) - 1:
                region_goal = goal
            else:
                region_goal = self.find_region_exit(region_start, region, regions[i+1])

            region_path_segment = self.a_star_plan(region_start, region_goal, region)
            detailed_path.extend(region_path_segment)

        return detailed_path
```

### Testing and Validation
```python
class PlanningValidator:
    def __init__(self):
        self.test_scenarios = []

    def validate_planner(self, planner, test_scenarios):
        """Validate planner performance across test scenarios"""
        results = {
            'success_rate': 0,
            'avg_path_cost': 0,
            'avg_planning_time': 0,
            'safety_violations': 0
        }

        successful_plans = 0
        total_cost = 0
        total_time = 0
        safety_violations = 0

        for scenario in test_scenarios:
            start_time = time.time()

            # Plan path
            path = planner.plan(scenario['start'], scenario['goal'], scenario['environment'])
            planning_time = time.time() - start_time

            if path:
                successful_plans += 1
                total_cost += self.calculate_path_cost(path)

                # Check safety
                if not self.check_path_safety(path, scenario['environment']):
                    safety_violations += 1
            else:
                # Plan failed
                total_cost += float('inf')

            total_time += planning_time

        if successful_plans > 0:
            results['success_rate'] = successful_plans / len(test_scenarios)
            results['avg_path_cost'] = total_cost / successful_plans
            results['avg_planning_time'] = total_time / len(test_scenarios)
        else:
            results['success_rate'] = 0

        results['safety_violations'] = safety_violations

        return results

    def calculate_path_cost(self, path):
        """Calculate total cost of path"""
        if not path:
            return float('inf')

        total_cost = 0
        for i in range(1, len(path)):
            total_cost += self.distance(path[i-1], path[i])

        return total_cost

    def check_path_safety(self, path, environment):
        """Check if path is safe"""
        if not path:
            return False

        for point in path:
            if self.is_point_in_collision(point, environment):
                return False

        return True

    def distance(self, p1, p2):
        """Calculate distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def is_point_in_collision(self, point, environment):
        """Check if point is in collision"""
        # Implementation depends on environment representation
        return False
```

## Summary

Planning and decision-making systems are crucial for autonomous robot operation. They enable robots to:

1. **Navigate efficiently** through complex environments using various path planning algorithms
2. **Make intelligent decisions** using frameworks like FSMs, behavior trees, and decision networks
3. **Learn and adapt** through reinforcement learning and imitation learning techniques
4. **Coordinate effectively** in multi-robot scenarios
5. **Operate safely** with proper safety constraints and verification

The choice of planning approach depends on the specific application requirements, including computational constraints, environment complexity, and safety considerations. Modern robotics increasingly combines classical planning methods with learning-based approaches to achieve robust and adaptive behavior.

## Next Steps
In the next chapter, we'll explore learning and adaptation in AI robot brains. We'll dive into machine learning techniques that enable robots to improve their performance over time, adapt to new situations, and acquire new skills through experience.