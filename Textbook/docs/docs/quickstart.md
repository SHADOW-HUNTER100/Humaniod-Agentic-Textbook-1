---
sidebar_position: 3
---

# Quickstart Guide

Get up and running with the Physical AI development environment in 10 minutes.

## Prerequisites

Before starting, ensure you have:
- Completed the [Installation Guide](installation.md)
- At least 32GB RAM and a modern NVIDIA GPU (RTX 4070 Ti or better)
- ROS 2 Humble Hawksbill properly installed and sourced
- NVIDIA Isaac Sim installed and licensed

## Hello, Physical AI!

Let's create your first Physical AI simulation that demonstrates basic robot control and perception.

### 1. Set Up Your Environment

First, activate your development environment:

```bash
# Activate Python environment
source ~/physical-ai-env/bin/activate

# Source ROS 2
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Set project root
export PHYSICAL_AI_ROOT=$(pwd)
```

### 2. Create a Simple Robot Controller

Create a basic ROS 2 node that moves a simulated robot:

```python
# hello_physical_ai.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time


class HelloPhysicalAI(Node):
    def __init__(self):
        super().__init__('hello_physical_ai')

        # Create publisher for robot movement
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Create subscriber for laser scanner
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Timer for periodic movement
        self.timer = self.create_timer(0.1, self.move_robot)
        self.get_logger().info('Hello, Physical AI! Node started.')

        self.obstacle_detected = False

    def scan_callback(self, msg):
        """Process laser scan data to detect obstacles"""
        # Check for obstacles in front of robot (within 1 meter)
        if len(msg.ranges) > 0:
            front_range = msg.ranges[len(msg.ranges) // 2]  # Middle of scan
            self.obstacle_detected = front_range < 1.0  # Obstacle within 1m

    def move_robot(self):
        """Send movement commands to robot"""
        cmd = Twist()

        if self.obstacle_detected:
            # Rotate in place if obstacle detected
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Rotate counter-clockwise
            self.get_logger().info('Obstacle detected! Rotating...')
        else:
            # Move forward if path is clear
            cmd.linear.x = 0.3  # Move forward at 0.3 m/s
            cmd.angular.z = 0.0
            self.get_logger().info('Moving forward...')

        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    hello_node = HelloPhysicalAI()

    try:
        rclpy.spin(hello_node)
    except KeyboardInterrupt:
        pass
    finally:
        hello_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 3. Run the Simulation

Now run the simulation with your controller:

```bash
# Terminal 1: Start Gazebo simulation
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Run your controller (after sourcing environment)
python hello_physical_ai.py
```

You should see the robot navigating the virtual environment, avoiding obstacles using its laser scanner!

## Physical AI Concepts

### Embodied Intelligence

Physical AI systems embody intelligence in physical form. Unlike traditional AI that operates in digital spaces, Physical AI understands and interacts with the physical world through:

- **Perception**: Using sensors to understand the environment
- **Action**: Executing physical movements and manipulations
- **Learning**: Adapting behavior based on physical interactions

### Digital Twin Integration

Your simulation connects to a digital twin environment:

```bash
# Launch Isaac Sim with digital twin
isaac-sim --/isaac/omniverse/app=apps/omni.isaac.sim.python.kit \
          --/isaac/omniverse/user=YOUR_USERNAME \
          --/isaac/omniverse/pass=YOUR_PASSWORD
```

### Vision-Language-Action Pipeline

Physical AI systems integrate vision, language, and action:

```python
# Example VLA integration
from perception.vision_processor import VisionProcessor
from nlp.language_interpreter import LanguageInterpreter
from action.executor import ActionExecutor

# Process visual input
vision_processor = VisionProcessor()
objects = vision_processor.detect_objects(camera_feed)

# Interpret language command
language_interpreter = LanguageInterpreter()
action_plan = language_interpreter.parse_command("Pick up the red cube")

# Execute action
action_executor = ActionExecutor()
action_executor.execute_plan(objects, action_plan)
```

## Running Advanced Simulations

### 1. Launch Humanoid Simulation

```bash
# Launch Unitree Go2 simulation
ros2 launch unitree_ros unitree_go2_sim.launch.py
```

### 2. Connect to NVIDIA Isaac Sim

```bash
# Start Isaac Sim with Physical AI environment
./start_isaac_sim.sh --scenario=physical_ai_demo
```

### 3. Train AI Models

```bash
# Train a simple perception model
python -m training.train_perception \
    --dataset-path /path/to/simulation/data \
    --model-type resnet50 \
    --epochs 100
```

## Next Steps

Congratulations! You've created your first Physical AI simulation. Next, explore:

1. **[Physical AI Concepts](physical-ai/concepts.md)** - Deep dive into embodied intelligence
2. Continue with the **[Tutorial Basics](tutorial-basics/create-a-document.md)** to learn more about documentation

## Troubleshooting

### Common Issues

- **Simulation Performance**: Reduce graphics quality or close other applications
- **GPU Memory**: Ensure sufficient VRAM for Isaac Sim (at least 8GB recommended)
- **ROS Communication**: Verify ROS_DOMAIN_ID consistency across terminals
- **Sensor Data**: Check sensor topics are publishing data with `ros2 topic echo`

Continue with **[Physical AI Concepts](physical-ai/concepts.md)** to deepen your understanding.