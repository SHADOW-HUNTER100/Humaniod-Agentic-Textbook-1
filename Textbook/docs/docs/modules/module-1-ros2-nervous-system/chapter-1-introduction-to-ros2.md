---
sidebar_label: 'Chapter 1: Introduction to ROS 2'
sidebar_position: 1
---

# Chapter 1: Introduction to ROS 2

## Learning Objectives
- Understand the fundamental concepts of ROS 2
- Learn about the ROS 2 architecture and its components
- Explore the differences between ROS 1 and ROS 2
- Set up your ROS 2 environment

## What is ROS 2?
Robot Operating System 2 (ROS 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.

### Notes:
- ROS 2 is not just an update to ROS 1, but a complete redesign addressing fundamental issues
- It provides better security, real-time support, and multi-robot system capabilities
- ROS 2 supports multiple DDS implementations (Fast DDS, Cyclone DDS, RTI Connext DDS)

## Key Concepts
- **Nodes**: Processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous communication patterns for long-running tasks
- **Packages**: Organizational unit for code and resources

### Code Example: Basic ROS 2 Node Structure
```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello ROS 2 World!')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on the Code:
- `rclpy.init()` initializes the ROS 2 client library
- `MinimalNode()` inherits from Node class to create a ROS 2 node
- `get_logger().info()` provides logging functionality
- `rclpy.spin()` keeps the node alive and processing callbacks
- Always call `destroy_node()` and `rclpy.shutdown()` for proper cleanup

## ROS 2 Architecture
ROS 2 uses a DDS (Data Distribution Service) implementation for communication between nodes. This provides better support for real-time systems, embedded systems, and multi-robot systems compared to ROS 1.

### Architecture Components:
- **RMW (ROS Middleware)**: Abstraction layer for different DDS implementations
- **rclpy/rclcpp**: Client libraries for Python/C++
- **ros2cli**: Command-line tools for system introspection and control

### Code Example: Creating a Simple Publisher
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()

    try:
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on Publisher Code:
- `create_publisher()` creates a publisher with message type, topic name, and QoS queue size
- `create_timer()` creates a timer that calls the callback function at specified intervals
- Message objects must be created and populated before publishing
- QoS (Quality of Service) settings control message delivery guarantees

## Differences between ROS 1 and ROS 2

### Key Improvements in ROS 2:
- **DDS-based communication**: Provides better real-time support and multi-robot systems
- **Improved security**: Built-in security features and authentication
- **Better cross-platform support**: Works on Windows, Linux, and macOS
- **Lifecycle management**: Better node lifecycle management
- **Quality of Service (QoS)**: Configurable reliability and performance settings

### Code Example: Quality of Service Settings
```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Create a QoS profile with specific settings
qos_profile = QoSProfile(
    depth=10,  # History depth
    reliability=QoSReliabilityPolicy.RELIABLE,  # Reliable delivery
    history=QoSHistoryPolicy.KEEP_LAST  # Keep last N messages
)

# Use the QoS profile when creating a publisher
publisher = node.create_publisher(String, 'topic', qos_profile)
```

### Notes on QoS Settings:
- **Reliability**: RELIABLE (all messages delivered) vs BEST_EFFORT (try to deliver)
- **History**: KEEP_ALL (store all messages) vs KEEP_LAST (store N most recent)
- **Depth**: Number of messages to store in the history
- QoS settings must match between publishers and subscribers for proper communication

## Setting up ROS 2 Environment

### Installation Notes:
- ROS 2 supports multiple distributions (Humble Hawksbill, Iron Irwini, Jazzy Jalisco)
- Choose the appropriate distribution based on your Ubuntu version
- Always source the setup script before using ROS 2 commands

### Command Examples:
```bash
# Source the ROS 2 setup script
source /opt/ros/humble/setup.bash

# Create a new workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build the workspace
colcon build

# Source the workspace
source install/setup.bash
```

## Next Steps
In the next chapter, we'll dive deeper into ROS 2 nodes and topics.