---
sidebar_label: 'Chapter 4: AI Agent Integration with ROS 2'
sidebar_position: 4
---

# Chapter 4: AI Agent Integration with ROS 2

## Learning Objectives
- Connect Python AI agents to ROS 2 controllers using rclpy
- Bridge AI decision-making with robotic actions
- Implement state management between AI and robotics
- Handle real-time constraints in AI-robot interaction

## Introduction to rclpy
rclpy is the Python client library for ROS 2. It provides the standard interface for Python programs to interact with ROS 2, allowing AI agents to communicate with robotic systems.

### Key Components
- **Node**: The basic execution unit in ROS 2
- **Publisher/Subscriber**: For asynchronous topic communication
- **Client/Server**: For synchronous service communication
- **Action Client/Server**: For long-running action communication

## Connecting AI Agents to ROS 2
To integrate AI agents with ROS 2, you need to:

1. Initialize rclpy in your AI application
2. Create a ROS 2 node within your AI agent
3. Set up publishers/subscribers for sensor and actuator data
4. Implement proper shutdown procedures

### Code Example: Basic AI Agent Integration
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import time

class SimpleAIAgent(Node):
    def __init__(self):
        super().__init__('simple_ai_agent')

        # Create subscriber for laser scan data
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        # Create publisher for robot commands
        self.cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Timer for AI decision making
        self.ai_timer = self.create_timer(0.1, self.ai_decision_loop)

        # State variables
        self.laser_data = None
        self.last_command_time = time.time()

        self.get_logger().info('Simple AI Agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = np.array(msg.ranges)
        self.get_logger().debug(f'Laser data received: {len(self.laser_data)} points')

    def ai_decision_loop(self):
        """Main AI decision-making loop"""
        if self.laser_data is not None:
            # Simple obstacle avoidance logic
            command = self.simple_obstacle_avoidance(self.laser_data)
            self.cmd_pub.publish(command)
            self.last_command_time = time.time()

    def simple_obstacle_avoidance(self, laser_data):
        """Simple AI logic for obstacle avoidance"""
        cmd = Twist()

        # Check for obstacles in front (within 1 meter)
        front_ranges = laser_data[330:30]  # Handle wrap-around
        if len(front_ranges) == 0:  # Handle case where angle range is invalid
            front_ranges = np.concatenate([laser_data[330:], laser_data[:30]])

        min_front_dist = np.min(front_ranges[np.isfinite(front_ranges)])

        if min_front_dist < 0.5:  # Obstacle too close
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn right
        else:
            cmd.linear.x = 0.5  # Move forward
            cmd.angular.z = 0.0

        return cmd

def main(args=None):
    rclpy.init(args=args)
    ai_agent = SimpleAIAgent()

    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        ai_agent.get_logger().info('AI agent interrupted by user')
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on Basic Integration:
- Always initialize rclpy before creating nodes
- Use appropriate message types for your robot's sensors and actuators
- Implement a timer-based decision loop for consistent AI processing
- Handle sensor data asynchronously with callbacks

### Code Example: Advanced AI Agent with State Management
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import json
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class RobotState:
    """Data class to maintain robot state"""
    position: Optional[list] = None
    orientation: Optional[list] = None
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    last_update_time: float = 0.0
    battery_level: float = 100.0
    task_status: str = "idle"
    goal_pose: Optional[list] = None

class AdvancedAIAgent(Node):
    def __init__(self):
        super().__init__('advanced_ai_agent')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/ai_status', 10)

        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)

        # Timer for AI processing
        self.ai_timer = self.create_timer(0.05, self.ai_decision_loop)  # 20 Hz

        # State management
        self.state = RobotState()
        self.ai_memory = {}  # For storing learned information

        # Parameters
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)

        self.get_logger().info('Advanced AI Agent initialized')

    def odom_callback(self, msg):
        """Update robot position and velocity"""
        self.state.position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ]

        self.state.orientation = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]

        self.state.linear_velocity = np.sqrt(
            msg.twist.twist.linear.x**2 +
            msg.twist.twist.linear.y**2
        )
        self.state.angular_velocity = msg.twist.twist.angular.z
        self.state.last_update_time = time.time()

    def laser_callback(self, msg):
        """Process laser scan data"""
        # Store laser data in state for later use
        self.laser_ranges = np.array(msg.ranges)

        # Update safety status based on obstacles
        min_range = np.min(self.laser_ranges[np.isfinite(self.laser_ranges)])
        safety_distance = self.get_parameter('safety_distance').value

        if min_range < safety_distance:
            self.state.task_status = "obstacle_detected"
        else:
            self.state.task_status = "safe"

    def goal_callback(self, msg):
        """Receive navigation goal"""
        self.state.goal_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ]
        self.state.task_status = "navigating"
        self.get_logger().info(f'New goal received: {self.state.goal_pose}')

    def ai_decision_loop(self):
        """Main AI decision-making loop with state management"""
        # Publish current AI status
        status_msg = String()
        status_msg.data = f"State: {self.state.task_status}, Pos: {self.state.position}"
        self.status_pub.publish(status_msg)

        # Make decisions based on current state
        if self.state.task_status == "navigating" and self.state.goal_pose:
            command = self.navigate_to_goal()
        elif self.state.task_status == "obstacle_detected":
            command = self.avoid_obstacle()
        else:
            command = self.explore_or_idle()

        # Publish command
        self.cmd_pub.publish(command)

    def navigate_to_goal(self):
        """Navigation AI logic"""
        cmd = Twist()

        if self.state.position and self.state.goal_pose:
            # Calculate direction to goal
            dx = self.state.goal_pose[0] - self.state.position[0]
            dy = self.state.goal_pose[1] - self.state.position[1]
            distance_to_goal = np.sqrt(dx**2 + dy**2)

            if distance_to_goal > 0.2:  # Not at goal yet
                # Simple proportional controller
                cmd.linear.x = min(0.3, distance_to_goal * 0.5)

                # Calculate angle to goal
                target_angle = np.arctan2(dy, dx)
                # Assuming we have orientation information to calculate current angle
                cmd.angular.z = target_angle * 0.5
            else:
                # Reached goal
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.state.task_status = "goal_reached"

        return cmd

    def avoid_obstacle(self):
        """Obstacle avoidance AI logic"""
        cmd = Twist()

        if hasattr(self, 'laser_ranges'):
            # Find safest direction (largest gap)
            safe_ranges = self.laser_ranges[np.isfinite(self.laser_ranges)]
            if len(safe_ranges) > 0:
                cmd.linear.x = 0.0  # Stop moving forward
                cmd.angular.z = 0.5  # Turn right to avoid obstacle

        return cmd

    def explore_or_idle(self):
        """Default behavior when no specific task"""
        cmd = Twist()
        cmd.linear.x = 0.2  # Slow forward movement
        cmd.angular.z = 0.0
        return cmd

def main(args=None):
    rclpy.init(args=args)
    ai_agent = AdvancedAIAgent()

    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        ai_agent.get_logger().info('Advanced AI agent interrupted')
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on Advanced Integration:
- Use dataclasses for structured state management
- Implement proper parameter handling for configuration
- Maintain separate callbacks for different sensor types
- Use state variables to track robot status and task progress

## State Management in AI Agents

### Code Example: State Machine for AI Agent
```python
from enum import Enum

class AgentState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING_OBSTACLE = "avoiding_obstacle"
    EXECUTING_TASK = "executing_task"
    EMERGENCY_STOP = "emergency_stop"

class StatefulAIAgent(Node):
    def __init__(self):
        super().__init__('stateful_ai_agent')

        # Initialize state
        self.current_state = AgentState.IDLE
        self.previous_state = None

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # State transition timer
        self.state_timer = self.create_timer(0.1, self.state_machine)

        # State-specific data
        self.state_data = {
            'navigation_goal': None,
            'obstacle_direction': None,
            'task_progress': 0.0
        }

    def laser_callback(self, msg):
        """Update sensor data used for state transitions"""
        self.laser_data = np.array(msg.ranges)

    def state_machine(self):
        """State machine logic for AI agent"""
        # Determine next state based on current state and sensor data
        next_state = self.determine_next_state()

        if next_state != self.current_state:
            self.transition_state(next_state)

        # Execute current state behavior
        self.execute_current_state()

    def determine_next_state(self):
        """Determine the next state based on conditions"""
        if hasattr(self, 'laser_data'):
            min_dist = np.min(self.laser_data[np.isfinite(self.laser_data)])

            if min_dist < 0.3:  # Emergency situation
                return AgentState.EMERGENCY_STOP
            elif min_dist < 0.5:  # Obstacle detected
                return AgentState.AVOIDING_OBSTACLE

        return self.current_state

    def transition_state(self, new_state):
        """Handle state transition"""
        self.get_logger().info(f'State transition: {self.current_state.value} -> {new_state.value}')
        self.previous_state = self.current_state
        self.current_state = new_state

        # Perform state-specific initialization
        if new_state == AgentState.EMERGENCY_STOP:
            self.emergency_stop_procedure()
        elif new_state == AgentState.AVOIDING_OBSTACLE:
            self.obstacle_avoidance_init()

    def execute_current_state(self):
        """Execute behavior for current state"""
        if self.current_state == AgentState.IDLE:
            self.execute_idle()
        elif self.current_state == AgentState.NAVIGATING:
            self.execute_navigation()
        elif self.current_state == AgentState.AVOIDING_OBSTACLE:
            self.execute_obstacle_avoidance()
        elif self.current_state == AgentState.EXECUTING_TASK:
            self.execute_task()
        elif self.current_state == AgentState.EMERGENCY_STOP:
            self.execute_emergency_stop()

    def emergency_stop_procedure(self):
        """Initialize emergency stop state"""
        self.get_logger().warn('EMERGENCY STOP ACTIVATED')
        # Additional emergency procedures can be added here

    def execute_emergency_stop(self):
        """Execute emergency stop behavior"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def execute_idle(self):
        """Execute idle behavior"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def execute_obstacle_avoidance(self):
        """Execute obstacle avoidance behavior"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.5  # Turn to avoid obstacle
        self.cmd_pub.publish(cmd)
```

### Notes on State Management:
- Use enums for clear state definitions
- Implement state transition logic with proper initialization
- Maintain state-specific data for complex behaviors
- Log state transitions for debugging

## Real-time Considerations

### Code Example: Real-time AI Processing with Threading
```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import time

class RealTimeAIAgent(Node):
    def __init__(self):
        super().__init__('real_time_ai_agent')

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # Queues for data processing
        self.sensor_queue = queue.Queue(maxsize=10)
        self.command_queue = queue.Queue(maxsize=10)

        # Threading for AI processing
        self.ai_thread = threading.Thread(target=self.ai_processing_loop, daemon=True)
        self.ai_thread.start()

        # Timer for real-time command execution
        self.cmd_timer = self.create_timer(0.02, self.execute_commands)  # 50 Hz

        # Performance monitoring
        self.last_ai_process_time = time.time()

    def laser_callback(self, msg):
        """Non-blocking sensor data collection"""
        try:
            self.sensor_queue.put_nowait(msg)
        except queue.Full:
            # Drop old data if queue is full
            try:
                self.sensor_queue.get_nowait()
                self.sensor_queue.put_nowait(msg)
            except queue.Empty:
                pass

    def ai_processing_loop(self):
        """AI processing in separate thread"""
        while rclpy.ok():
            try:
                # Get sensor data
                sensor_msg = self.sensor_queue.get(timeout=0.1)

                # Process with AI
                start_time = time.time()
                command = self.process_with_ai(sensor_msg)
                process_time = time.time() - start_time

                # Put command in queue
                try:
                    self.command_queue.put_nowait(command)
                except queue.Full:
                    pass  # Drop command if queue is full

                # Monitor processing time
                if process_time > 0.05:  # More than 50ms
                    self.get_logger().warn(f'AI processing took {process_time:.3f}s')

            except queue.Empty:
                continue

    def process_with_ai(self, sensor_msg):
        """AI processing logic"""
        # Convert sensor data to appropriate format
        laser_data = np.array(sensor_msg.ranges)

        # AI decision making
        cmd = Twist()
        min_dist = np.min(laser_data[np.isfinite(laser_data)])

        if min_dist < 0.5:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5
        else:
            cmd.linear.x = 0.5
            cmd.angular.z = 0.0

        return cmd

    def execute_commands(self):
        """Execute commands at real-time rate"""
        try:
            # Get latest command from queue
            while not self.command_queue.empty():
                cmd = self.command_queue.get_nowait()

            # Publish latest command
            self.cmd_pub.publish(cmd)
        except queue.Empty:
            # No new commands, use default
            cmd = Twist()
            self.cmd_pub.publish(cmd)
```

### Notes on Real-time Processing:
- Use threading to separate AI processing from ROS message handling
- Implement queues to manage data flow between threads
- Monitor processing time to ensure real-time performance
- Use non-blocking operations to avoid deadlocks

## URDF Integration

### Code Example: URDF-based Robot Model Access
```python
from rclpy.qos import QoSProfile
from std_msgs.msg import String

class URDFIntelligentAgent(Node):
    def __init__(self):
        super().__init__('urdf_intelligent_agent')

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # URDF access
        self.urdf_model = self.load_urdf_model()

        # AI decision timer
        self.ai_timer = self.create_timer(0.1, self.ai_decision_loop)

        # Joint limits from URDF
        self.joint_limits = self.extract_joint_limits()

    def load_urdf_model(self):
        """Load URDF model for kinematic calculations"""
        # In practice, you might use a URDF parser library
        # For this example, we'll simulate loading
        self.get_logger().info('Loading URDF model...')

        # This would typically use a URDF parser
        # urdf_parser = URDFParser()
        # return urdf_parser.parse_from_parameter_server(self)

        # For now, return a placeholder
        return {"joints": [], "links": []}

    def extract_joint_limits(self):
        """Extract joint limits from URDF for AI safety"""
        # This would extract limits from the URDF model
        limits = {}
        # Example: limits['joint_name'] = {'min': -1.57, 'max': 1.57}
        return limits

    def joint_state_callback(self, msg):
        """Monitor joint states and enforce URDF-based constraints"""
        for i, name in enumerate(msg.name):
            if name in self.joint_limits:
                limit = self.joint_limits[name]
                position = msg.position[i]

                # Check if joint is approaching limits
                if position < limit['min'] or position > limit['max']:
                    self.get_logger().warn(f'Joint {name} near limit: {position}')

    def ai_decision_loop(self):
        """AI decisions considering URDF constraints"""
        cmd = Twist()

        # AI logic that considers robot's physical constraints
        # based on URDF model
        cmd.linear.x = 0.3
        cmd.angular.z = 0.2

        # Check if command would violate constraints
        safe_cmd = self.enforce_physical_constraints(cmd)
        self.cmd_pub.publish(safe_cmd)

    def enforce_physical_constraints(self, cmd):
        """Enforce physical constraints based on URDF"""
        # This would implement kinematic/dynamic constraints
        # from the URDF model
        return cmd  # Simplified for example
```

### Notes on URDF Integration:
- Use URDF models to understand robot's physical constraints
- Extract joint limits and kinematic information
- Enforce safety constraints in AI decision-making
- Monitor joint states to prevent violations

## Integration with Machine Learning Models

### Code Example: ML Model Integration
```python
import tensorflow as tf  # or pytorch
import numpy as np

class MLIntelligentAgent(Node):
    def __init__(self):
        super().__init__('ml_intelligent_agent')

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)

        # Load ML model
        self.ml_model = self.load_model()

        # AI processing timer
        self.ai_timer = self.create_timer(0.1, self.ml_decision_loop)

        # Data buffer for ML input
        self.sensor_buffer = []

    def load_model(self):
        """Load pre-trained ML model"""
        try:
            # Load TensorFlow model
            model = tf.keras.models.load_model('/path/to/model')
            self.get_logger().info('ML model loaded successfully')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return None

    def laser_callback(self, msg):
        """Process laser data for ML model"""
        # Convert laser scan to appropriate format
        laser_array = np.array(msg.ranges)
        # Replace invalid values with max range
        laser_array = np.nan_to_num(laser_array, nan=np.max(laser_array))

        # Add to buffer for ML processing
        self.sensor_buffer.append(laser_array)

        # Keep only recent data
        if len(self.sensor_buffer) > 10:
            self.sensor_buffer.pop(0)

    def ml_decision_loop(self):
        """ML-based decision making"""
        if self.ml_model is not None and len(self.sensor_buffer) > 0:
            # Prepare input for model
            recent_data = np.array([self.sensor_buffer[-1]])  # Use latest

            try:
                # Run inference
                prediction = self.ml_model.predict(recent_data, verbose=0)

                # Convert prediction to robot command
                cmd = self.convert_prediction_to_command(prediction[0])
                self.cmd_pub.publish(cmd)
            except Exception as e:
                self.get_logger().error(f'ML inference failed: {e}')

    def convert_prediction_to_command(self, prediction):
        """Convert ML model output to robot command"""
        cmd = Twist()
        cmd.linear.x = float(prediction[0])  # Linear velocity
        cmd.angular.z = float(prediction[1])  # Angular velocity
        return cmd
```

### Notes on ML Integration:
- Load models efficiently to avoid blocking ROS operations
- Preprocess sensor data to match model expectations
- Handle model inference errors gracefully
- Consider model performance requirements

## Best Practices

### Code Example: Complete AI Agent Template
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import LaserScan, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np
import json
import time
from enum import Enum

class AIState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    AVOIDING = "avoiding"
    ERROR = "error"

class ProductionAIAgent(Node):
    def __init__(self):
        super().__init__('production_ai_agent')

        # QoS profiles for reliability
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile)
        self.status_pub = self.create_publisher(String, '/ai_agent_status', qos_profile)

        # Subscribers
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_profile)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, qos_profile)

        # Parameters
        self.declare_parameter('ai_loop_rate', 20)  # Hz
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('max_linear_speed', 0.5)

        # State management
        self.state = AIState.IDLE
        self.safety_distance = self.get_parameter('safety_distance').value
        self.max_speed = self.get_parameter('max_linear_speed').value

        # Timers
        loop_rate = self.get_parameter('ai_loop_rate').value
        self.ai_timer = self.create_timer(1.0/loop_rate, self.ai_loop)

        # Data storage
        self.laser_data = None
        self.joint_data = None

        self.get_logger().info('Production AI Agent initialized')

    def laser_callback(self, msg):
        """Process laser scan data with error handling"""
        try:
            self.laser_data = np.array(msg.ranges)
            self.laser_data = np.nan_to_num(self.laser_data, nan=np.inf)
        except Exception as e:
            self.get_logger().error(f'Laser callback error: {e}')

    def joint_callback(self, msg):
        """Process joint state data"""
        try:
            self.joint_data = {
                'names': msg.name,
                'positions': msg.position,
                'velocities': msg.velocity,
                'effort': msg.effort
            }
        except Exception as e:
            self.get_logger().error(f'Joint callback error: {e}')

    def ai_loop(self):
        """Main AI processing loop"""
        try:
            # Update status
            status_msg = String()
            status_msg.data = json.dumps({
                'state': self.state.value,
                'timestamp': time.time(),
                'laser_valid': self.laser_data is not None
            })
            self.status_pub.publish(status_msg)

            # Make decisions based on current state
            if self.laser_data is not None:
                cmd = self.make_decision()
                self.cmd_pub.publish(cmd)

        except Exception as e:
            self.get_logger().error(f'AI loop error: {e}')
            self.state = AIState.ERROR
            # Emergency stop
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

    def make_decision(self):
        """Main decision-making logic"""
        cmd = Twist()

        # Safety check first
        min_distance = np.min(self.laser_data[np.isfinite(self.laser_data)])

        if min_distance < self.safety_distance:
            # Emergency stop
            self.state = AIState.AVOIDING
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn away from obstacle
        else:
            # Normal operation
            self.state = AIState.NAVIGATING
            cmd.linear.x = min(self.max_speed, 0.5)
            cmd.angular.z = 0.0

        return cmd

def main(args=None):
    rclpy.init(args=args)
    ai_agent = ProductionAIAgent()

    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        ai_agent.get_logger().info('Production AI agent interrupted')
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices Summary:
- Use proper QoS profiles for reliable communication
- Implement comprehensive error handling
- Monitor and log system status
- Use parameters for configuration
- Implement safety checks in decision-making
- Separate concerns with modular design

## Summary
This chapter covered the essential techniques for integrating AI agents with ROS 2 robotic systems, enabling intelligent decision-making in robotic applications. We explored state management, real-time processing, URDF integration, and machine learning model integration.

## Next Steps
This concludes Module 1. In the next module, we'll explore digital twin simulation for robotics.