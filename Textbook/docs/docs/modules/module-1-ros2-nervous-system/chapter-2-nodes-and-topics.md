---
sidebar_label: 'Chapter 2: ROS 2 Nodes and Topics'
sidebar_position: 2
---

# Chapter 2: ROS 2 Nodes and Topics

## Learning Objectives
- Create and run ROS 2 nodes
- Understand topic-based communication
- Implement publishers and subscribers
- Use ROS 2 command-line tools

## Nodes in ROS 2
A node is a process that performs computation. Nodes are combined together to form a complete robot application. In ROS 2, nodes are designed to be as lightweight as possible.

### Creating a Node
To create a node, you need to:
1. Initialize the ROS client library
2. Create a node object
3. Define the node's functionality
4. Spin the node to keep it alive

### Code Example: Complete Node Implementation
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class CustomNode(Node):
    def __init__(self):
        super().__init__('custom_node')

        # Create a publisher
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create a timer to periodically publish messages
        self.timer = self.create_timer(0.5, self.timer_callback)

        # Counter for messages
        self.counter = 0

        # Log node creation
        self.get_logger().info('CustomNode has been started')

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello from custom node: {self.counter}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = CustomNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on Node Implementation:
- Always call `super().__init__()` with a unique node name
- Use `create_publisher()` to create topic publishers
- Use `create_timer()` for periodic tasks
- Implement proper error handling and cleanup
- Use `get_logger()` for debugging and status messages

## Topics and Messages
Topics are named buses over which nodes exchange messages. Messages are data structures that are passed between nodes. ROS 2 uses a publish-subscribe communication pattern for topics.

### Publishers and Subscribers
- **Publishers** send messages to topics
- **Subscribers** receive messages from topics
- Multiple nodes can publish and subscribe to the same topic

### Code Example: Publisher and Subscriber Pair
```python
# publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Talker(Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    talker = Talker()

    try:
        rclpy.spin(talker)
    except KeyboardInterrupt:
        pass
    finally:
        talker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

```python
# listener.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    listener = Listener()

    try:
        rclpy.spin(listener)
    except KeyboardInterrupt:
        pass
    finally:
        listener.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on Publisher/Subscriber Implementation:
- Publisher and subscriber must use the same topic name
- Message types must match between publisher and subscriber
- QoS settings should be compatible for proper communication
- Always handle the case where subscription object might be unused (the comment line)

## Advanced Node Features

### Parameters in Nodes
Nodes can accept parameters for configuration:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ParameterizedNode(Node):
    def __init__(self):
        super().__init__('parameterized_node')

        # Declare parameters with default values
        self.declare_parameter('message_prefix', 'Hello')
        self.declare_parameter('publish_rate', 1.0)

        # Get parameter values
        self.prefix = self.get_parameter('message_prefix').value
        self.rate = self.get_parameter('publish_rate').value

        # Create publisher and timer
        self.publisher = self.create_publisher(String, 'parameterized_topic', 10)
        self.timer = self.create_timer(1.0/self.rate, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'{self.prefix} World: {self.counter}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = ParameterizedNode()

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

### Notes on Parameters:
- Use `declare_parameter()` to define parameters with default values
- Access parameters using `get_parameter()` method
- Parameters can be set at runtime using command line or parameter files
- Parameters provide flexibility without recompiling code

## Command-Line Tools
ROS 2 provides several command-line tools:

### Node Tools
- `ros2 node list` - List active nodes
- `ros2 node info <node_name>` - Get detailed information about a node
- `ros2 run <package_name> <executable_name>` - Run a node directly

### Topic Tools
- `ros2 topic list` - List active topics
- `ros2 topic echo <topic_name>` - Print messages from a topic
- `ros2 topic info <topic_name>` - Get information about a topic
- `ros2 topic pub <topic_name> <msg_type> <args>` - Publish a message to a topic

### Code Example: Using Command Line Tools
```bash
# Terminal 1: Run the publisher
ros2 run demo_nodes_py talker

# Terminal 2: Run the subscriber
ros2 run demo_nodes_py listener

# Terminal 3: Monitor the topic
ros2 topic echo /chatter std_msgs/msg/String

# Terminal 4: List all topics
ros2 topic list

# Terminal 5: Get topic info
ros2 topic info /chatter
```

### Notes on Command Line Tools:
- Use `ros2 topic hz <topic_name>` to check message frequency
- Use `ros2 topic bw <topic_name>` to check bandwidth usage
- Use `ros2 node list` to see which nodes are running
- Use `ros2 run` to execute nodes from packages

## Quality of Service (QoS) Settings

### Code Example: QoS Configuration
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String

class QoSPublisher(Node):
    def __init__(self):
        super().__init__('qos_publisher')

        # Create a custom QoS profile
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        self.publisher = self.create_publisher(
            String,
            'qos_chatter',
            qos_profile
        )

        self.timer = self.create_timer(0.5, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'QoS Message: {self.counter}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.data}')
        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    node = QoSPublisher()

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

### Notes on QoS Settings:
- RELIABLE: Guarantees message delivery (use for critical data)
- BEST_EFFORT: Attempts delivery but doesn't guarantee it (use for streaming data)
- KEEP_ALL: Stores all messages up to available memory
- KEEP_LAST: Stores only the most recent messages (specify depth)

## Best Practices

### Node Design
- Keep nodes focused on a single responsibility
- Use descriptive names for nodes, topics, and parameters
- Implement proper error handling and logging
- Always call cleanup methods in finally blocks

### Topic Communication
- Use meaningful topic names that reflect their purpose
- Match QoS settings between publishers and subscribers
- Consider message frequency and bandwidth usage
- Use appropriate message types for your data

## Next Steps
In the next chapter, we'll explore services and actions in ROS 2.