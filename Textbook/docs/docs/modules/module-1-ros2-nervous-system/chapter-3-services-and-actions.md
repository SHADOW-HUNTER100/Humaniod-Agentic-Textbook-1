---
sidebar_label: 'Chapter 3: ROS 2 Services and Actions'
sidebar_position: 3
---

# Chapter 3: ROS 2 Services and Actions

## Learning Objectives
- Understand service-based communication in ROS 2
- Implement client-server communication patterns
- Work with ROS 2 actions for long-running tasks
- Compare services vs actions vs topics

## Services in ROS 2
Services provide a request/response communication pattern in ROS 2. Unlike topics which are asynchronous, services are synchronous - the client waits for a response from the server.

### Service Architecture
- **Service Server**: Provides a service by processing requests and returning responses
- **Service Client**: Sends requests to a service server and waits for responses
- **Service Interface**: Defines the request and response message types

### Creating a Service
Service interfaces are defined using `.srv` files that specify both the request and response message formats:

```
# Request (part before the three dashes)
string name
int32 age
---
# Response (part after the three dashes)
bool success
string message
```

### Code Example: Service Server Implementation
```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback
        )

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(
            f'Incoming request\na: {request.a} b: {request.b}'
        )
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()

    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        minimal_service.get_logger().info('Service interrupted by user')
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on Service Server:
- `create_service()` creates a service server with the specified service type, name, and callback
- The callback function receives both request and response objects
- Always return the response object from the callback
- Service names should be descriptive and follow naming conventions

### Code Example: Service Client Implementation
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def send_request(self, a, b):
        request = AddTwoInts.Request()
        request.a = a
        request.b = b
        self.future = self.cli.call_async(request)
        return self.future

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()

    # Send a request
    future = minimal_client.send_request(1, 2)

    try:
        rclpy.spin_until_future_complete(minimal_client, future)
        if future.result() is not None:
            response = future.result()
            minimal_client.get_logger().info(
                f'Result of add_two_ints: {response.sum}'
            )
        else:
            minimal_client.get_logger().info('Service call failed')
    except KeyboardInterrupt:
        minimal_client.get_logger().info('Client interrupted by user')
    finally:
        minimal_client.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on Service Client:
- `create_client()` creates a client for the specified service
- Use `wait_for_service()` to ensure the service is available before making calls
- `call_async()` makes non-blocking service calls
- Use `spin_until_future_complete()` to wait for the response

## Actions in ROS 2
Actions are designed for long-running tasks that may take seconds, minutes, or hours to complete. They provide feedback during execution and can be canceled.

### Action Architecture
- **Goal**: Request to start an action
- **Feedback**: Status updates during execution
- **Result**: Final outcome of the action
- **Cancel**: Request to stop an action

### Code Example: Action Server Implementation
```python
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup())

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1]
            )

            self.get_logger().info(f'Feedback: {feedback_msg.sequence}')
            goal_handle.publish_feedback(feedback_msg)

            time.sleep(1)  # Simulate work

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')

        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()

    try:
        executor = MultiThreadedExecutor()
        rclpy.spin(fibonacci_action_server, executor=executor)
    except KeyboardInterrupt:
        fibonacci_action_server.get_logger().info('Action server interrupted')
    finally:
        fibonacci_action_server.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on Action Server:
- Action servers require a callback function that handles the goal execution
- Use `publish_feedback()` to send progress updates
- Check `is_cancel_requested` to handle cancellation requests
- Call `succeed()`, `abort()`, or `canceled()` to complete the goal
- Use `ReentrantCallbackGroup` for concurrent processing

### Code Example: Action Client Implementation
```python
import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()

    action_client.send_goal(10)

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        action_client.get_logger().info('Action client interrupted')

if __name__ == '__main__':
    main()
```

### Notes on Action Client:
- Use `send_goal_async()` for non-blocking goal sending
- Implement `feedback_callback` to receive progress updates
- Use `get_result_async()` to get the final result
- Handle goal acceptance/rejection in the response callback

## Service vs Topic vs Action Comparison

### When to Use Each Communication Pattern

| Pattern | Use Case | Characteristics |
|---------|----------|-----------------|
| Topics | Continuous data streaming | Async, broadcast, one-way |
| Services | Request/response | Sync, one-to-one, immediate |
| Actions | Long-running tasks | Async, with feedback/cancel |

### Code Example: Comparing Communication Patterns
```python
# Topic Publisher (continuous data)
class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')
        self.publisher = self.create_publisher(String, 'sensor_data', 10)
        self.timer = self.create_timer(0.1, self.publish_sensor_data)

    def publish_sensor_data(self):
        msg = String()
        msg.data = f'Sensor reading: {time.time()}'
        self.publisher.publish(msg)

# Service Server (request/response)
class QueryServer(Node):
    def __init__(self):
        super().__init__('query_server')
        self.srv = self.create_service(
            Trigger, 'get_status', self.status_callback)

    def status_callback(self, request, response):
        response.success = True
        response.message = 'System is operational'
        return response

# Action Server (long-running task)
class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')
        self._action_server = ActionServer(
            self, NavigateToPose, 'navigate_to_pose', self.execute_nav_callback)

    def execute_nav_callback(self, goal_handle):
        # Long-running navigation task with feedback
        for step in range(goal_handle.request.target_pose.steps):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return NavigateToPose.Result()

            # Navigate one step
            feedback_msg = NavigateToPose.Feedback()
            feedback_msg.current_pose = get_current_pose()
            goal_handle.publish_feedback(feedback_msg)

            time.sleep(0.5)

        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.pose = get_current_pose()
        return result
```

### Notes on Communication Pattern Selection:
- Use topics for sensor data, status updates, and continuous monitoring
- Use services for quick queries, configuration changes, and immediate responses
- Use actions for navigation, manipulation, and other time-consuming operations

## Integration with AI Agents
ROS 2 services and actions are particularly useful for AI agents that need to coordinate with robotic systems:

### Code Example: AI Agent Using Services
```python
class AIAgent(Node):
    def __init__(self):
        super().__init__('ai_agent')

        # Service client for sensor queries
        self.sensor_client = self.create_client(
            GetSensorData, 'get_sensor_data')

        # Action client for navigation
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        # Timer to run AI decision-making loop
        self.ai_timer = self.create_timer(1.0, self.ai_decision_loop)

    def ai_decision_loop(self):
        # Query sensor data
        future = self.sensor_client.call_async(GetSensorData.Request())

        # Process sensor data and make decisions
        # This would typically involve AI algorithms
        if self.should_navigate():
            self.send_navigation_goal()

    def send_navigation_goal(self):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.calculate_target_pose()

        self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback)

    def nav_feedback_callback(self, feedback_msg):
        # Process navigation feedback
        # Adjust AI behavior based on progress
        pass

def main(args=None):
    rclpy.init(args=args)
    ai_agent = AIAgent()

    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        ai_agent.get_logger().info('AI agent interrupted')
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()
```

### Notes on AI Integration:
- Services for quick sensor queries and state checks
- Actions for complex robot behaviors like navigation or manipulation
- Proper error handling for failed service calls or action cancellations
- Integration with AI decision-making loops

## Best Practices

### Service Best Practices
- Use descriptive service names that indicate their purpose
- Implement proper timeout handling in clients
- Validate request parameters in service servers
- Return meaningful error responses

### Action Best Practices
- Provide regular feedback during long-running operations
- Handle cancellation requests gracefully
- Use appropriate goal IDs for tracking multiple concurrent goals
- Implement proper error recovery mechanisms

## Summary
Services and actions complement the topic-based communication by providing synchronous and long-running communication patterns respectively. Services are ideal for request/response interactions, while actions are perfect for tasks that require feedback and cancellation capabilities.

## Next Steps
In the next chapter, we'll explore how to connect Python AI agents to ROS 2 controllers using rclpy.