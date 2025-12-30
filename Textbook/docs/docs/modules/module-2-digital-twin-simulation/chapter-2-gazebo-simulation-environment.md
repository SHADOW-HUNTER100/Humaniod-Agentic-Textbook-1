---
sidebar_label: 'Chapter 2: Gazebo Simulation Environment'
sidebar_position: 2
---

# Chapter 2: Gazebo Simulation Environment

## Learning Objectives
- Set up and configure Gazebo for robotics simulation
- Create robot models and environments in Gazebo
- Implement physics-based simulation with realistic parameters
- Integrate Gazebo with ROS 2 for robot control

## Introduction to Gazebo
Gazebo is a physics-based simulation environment that provides realistic rendering, physics simulation, and sensor simulation capabilities. It's widely used in robotics for testing algorithms, training AI models, and validating robot behaviors before deployment.

### Notes on Gazebo Architecture:
- Gazebo uses a client-server architecture
- The server handles physics simulation and rendering
- Clients can connect to visualize and control the simulation
- Multiple physics engines are supported (ODE, Bullet, DART)

### Code Example: Basic Gazebo Launch File
```xml
<?xml version="1.0"?>
<launch>
  <!-- Set Gazebo arguments -->
  <arg name="world" default="empty"/>
  <arg name="paused" default="false"/>
  <arg name="use_sim_time" default="true"/>
  <arg name="gui" default="true"/>
  <arg name="headless" default="false"/>
  <arg name="debug" default="false"/>

  <!-- Launch Gazebo server -->
  <node name="gazebo" pkg="gazebo_ros" type="gzserver" respawn="false"
        args="$(arg world) -u $(arg paused) $(arg debug)" output="screen"/>

  <!-- Launch Gazebo client (GUI) -->
  <node name="gazebo_gui" pkg="gazebo_ros" type="gzclient" respawn="false"
        args="$(arg gui)" output="screen" if="$(arg gui)"/>
</launch>
```

### Notes on Launch Files:
- Use `gzserver` for headless simulation
- Use `gzclient` for GUI visualization
- The `use_sim_time` parameter synchronizes simulation time with ROS
- Different physics engines can be specified via command-line arguments

## Key Features of Gazebo

### Physics Simulation
Gazebo provides realistic physics simulation:

```xml
<!-- Physics configuration in world file -->
<sdf version="1.6">
  <world name="default">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>
  </world>
</sdf>
```

### Notes on Physics Configuration:
- `max_step_size`: Smaller values provide more accurate but slower simulation
- `real_time_factor`: 1.0 means simulation runs at real-time speed
- `real_time_update_rate`: Higher rates provide smoother simulation
- Gravity can be modified for different environments (Moon, Mars, etc.)

### Sensor Simulation Capabilities
Gazebo supports various sensors with realistic noise models:

```xml
<!-- Camera sensor configuration -->
<sensor name="camera" type="camera">
  <camera name="head">
    <horizontal_fov>1.3962634</horizontal_fov>
    <image>
      <width>800</width>
      <height>600</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### Notes on Sensor Configuration:
- Each sensor type has specific parameters
- Update rates affect computational load
- Visualize flag controls whether sensor is shown in GUI
- Noise models can be added for realistic simulation

## Setting Up Gazebo with ROS 2

### Installation and Setup
```bash
# Install Gazebo Garden (or appropriate version)
sudo apt update
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros-control

# Verify installation
gazebo --version
```

### Code Example: Complete Robot URDF with Gazebo Integration
```xml
<?xml version="1.0"?>
<robot name="turtlebot3_burger" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://turtlebot3_description/meshes/bases/burger_base.stl" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="light_black">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="package://turtlebot3_description/meshes/bases/burger_base.stl" scale="0.001 0.001 0.001"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.432"/>
      <inertia ixx="0.00023" ixy="0.0" ixz="0.0" iyy="0.00023" iyz="0.0" izz="0.0002"/>
    </inertial>
  </link>

  <!-- Wheel joints and links -->
  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left_link"/>
    <origin xyz="0 0.1 0" rpy="-1.57 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <link name="wheel_left_link">
    <visual>
      <geometry>
        <mesh filename="package://turtlebot3_description/meshes/wheels/left_tire.stl" scale="0.001 0.001 0.001"/>
      </geometry>
      <material name="dark">
        <color rgba="0.2 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.016" radius="0.033"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>/tb3</namespace>
        <remapping>cmd_vel:=cmd_vel</remapping>
        <remapping>odom:=odom</remapping>
      </ros>
      <left_joint>wheel_left_joint</left_joint>
      <right_joint>wheel_right_joint</right_joint>
      <wheel_separation>0.160</wheel_separation>
      <wheel_diameter>0.066</wheel_diameter>
      <max_wheel_torque>20.0</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
    </plugin>
  </gazebo>

  <!-- Camera sensor -->
  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <pose>0.0 0.0 0.0 0.0 0.0 0.0</pose>
      <camera name="head">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>/tb3</namespace>
          <remapping>image_raw:=camera/image_raw</remapping>
          <remapping>camera_info:=camera/camera_info</remapping>
        </ros>
        <camera_name>camera</camera_name>
        <image_topic_name>image_raw</image_topic_name>
        <camera_info_topic_name>camera_info</camera_info_topic_name>
        <frame_name>camera_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

### Notes on URDF Integration:
- Each physical link needs visual, collision, and inertial properties
- Joints define how links connect and move relative to each other
- Gazebo plugins provide ROS interfaces for control and sensing
- Sensor configurations define realistic sensor behavior

### Code Example: ROS 2 Node for Gazebo Control
```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np

class GazeboController(Node):
    def __init__(self):
        super().__init__('gazebo_controller')

        # Publishers for robot control
        self.cmd_vel_pub = self.create_publisher(Twist, '/tb3/cmd_vel', 10)

        # Subscribers for sensor data
        self.laser_sub = self.create_subscription(
            LaserScan, '/tb3/scan', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/tb3/odom', self.odom_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/tb3/camera/image_raw', self.image_callback, 10)

        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)

        # Internal state
        self.current_odom = None
        self.current_scan = None
        self.current_image = None
        self.cv_bridge = CvBridge()

        # Control parameters
        self.linear_speed = 0.2
        self.angular_speed = 0.5

        self.get_logger().info('Gazebo Controller initialized')

    def laser_callback(self, msg):
        """Process laser scan data from Gazebo"""
        self.current_scan = msg
        self.get_logger().debug(f'Laser ranges: {len(msg.ranges)} points')

    def odom_callback(self, msg):
        """Process odometry data from Gazebo"""
        self.current_odom = msg
        pos = msg.pose.pose.position
        self.get_logger().debug(f'Position: ({pos.x:.2f}, {pos.y:.2f})')

    def image_callback(self, msg):
        """Process camera image from Gazebo"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.current_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def control_loop(self):
        """Main control loop for robot in Gazebo"""
        cmd = Twist()

        if self.current_scan is not None:
            # Simple obstacle avoidance using laser data
            min_range = min(self.current_scan.ranges)

            if min_range < 0.5:  # Obstacle too close
                cmd.linear.x = 0.0
                cmd.angular.z = self.angular_speed
            else:
                cmd.linear.x = self.linear_speed
                cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)

    def move_to_goal(self, goal_x, goal_y):
        """Navigate to a specific goal position"""
        if self.current_odom is not None:
            current_pos = self.current_odom.pose.pose.position
            dx = goal_x - current_pos.x
            dy = goal_y - current_pos.y
            distance = np.sqrt(dx**2 + dy**2)

            if distance > 0.1:  # Not at goal yet
                cmd = Twist()
                cmd.linear.x = min(0.2, distance * 0.5)  # Proportional control
                cmd.angular.z = np.arctan2(dy, dx) * 0.5
                self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    controller = GazeboController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Gazebo Controller interrupted')
    finally:
        # Stop the robot before shutting down
        cmd = Twist()
        controller.cmd_vel_pub.publish(cmd)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on Gazebo Control:
- Use appropriate topic names that match Gazebo plugin configurations
- Implement proper error handling for sensor data processing
- Consider simulation timing when designing control loops
- Always stop the robot before shutting down to prevent simulation errors

## Creating Robot Models in Gazebo

### Code Example: Custom Robot Model with Multiple Sensors
```xml
<?xml version="1.0"?>
<robot name="custom_robot">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.1"/>
      <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
  </link>

  <!-- Camera link -->
  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.1"/>
  </joint>

  <link name="camera_link">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- LiDAR link -->
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0.2 0 0.2"/>
  </joint>

  <link name="lidar_link">
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Gazebo plugins for sensors and actuators -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="camera_link">
    <sensor name="camera" type="camera">
      <pose>0 0 0 0 0 0</pose>
      <camera name="custom_camera">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <ros>
          <namespace>/custom_robot</namespace>
          <remapping>image_raw:=camera/image_raw</remapping>
        </ros>
        <frame_name>camera_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="lidar_link">
    <sensor name="lidar" type="ray">
      <pose>0 0 0 0 0 0</pose>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1.0</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.1</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros>
          <namespace>/custom_robot</namespace>
          <remapping>scan:=scan</remapping>
        </ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>lidar_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Differential drive plugin -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <ros>
        <namespace>/custom_robot</namespace>
      </ros>
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.4</wheel_separation>
      <wheel_diameter>0.15</wheel_diameter>
      <max_wheel_torque>20.0</max_wheel_torque>
      <max_wheel_acceleration>1.0</max_wheel_acceleration>
      <publish_odom>true</publish_odom>
      <publish_odom_tf>true</publish_odom_tf>
      <publish_wheel_tf>true</publish_wheel_tf>
    </plugin>
  </gazebo>
</robot>
```

### Notes on Custom Robot Models:
- Proper inertial properties are crucial for realistic physics
- Sensor noise models make simulation more realistic
- Use fixed joints for sensors that don't move relative to base
- Proper frame naming is important for TF transforms

## Physics Configuration

### Code Example: Advanced Physics Configuration
```xml
<!-- Custom world file with advanced physics settings -->
<sdf version="1.6">
  <world name="custom_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>

      <!-- ODE-specific parameters -->
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom environment objects -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="table_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 0.5 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 0.5 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Notes on Physics Configuration:
- Smaller step sizes provide more accurate simulation but are computationally expensive
- The ERP (Error Reduction Parameter) affects how quickly constraint errors are corrected
- CFM (Constraint Force Mixing) affects constraint stiffness
- Adjust parameters based on simulation requirements and performance

## Sensor Simulation

### Code Example: Advanced Sensor Configuration with Noise Models
```xml
<!-- IMU sensor with realistic noise -->
<sensor name="imu_sensor" type="imu">
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <pose>0 0 0 0 0 0</pose>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.0000075</bias_mean>
          <bias_stddev>0.0000008</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.1</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <ros>
      <namespace>/custom_robot</namespace>
      <remapping>imu:=imu/data</remapping>
    </ros>
    <frame_name>imu_link</frame_name>
    <body_name>imu_link</body_name>
  </plugin>
</sensor>

<!-- GPS sensor with noise -->
<sensor name="gps_sensor" type="gps">
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <pose>0 0 0 0 0 0</pose>
  <plugin name="gps_plugin" filename="libgazebo_ros_gps.so">
    <ros>
      <namespace>/custom_robot</namespace>
      <remapping>navsat/fix:=gps/fix</remapping>
    </ros>
    <frame_name>gps_link</frame_name>
    <update_rate>10</update_rate>
    <gps>
      <position_noise>0.1</position_noise>
      <velocity_noise>0.01</velocity_noise>
    </gps>
  </plugin>
</sensor>
```

### Code Example: ROS 2 Node for Processing Multiple Gazebo Sensors
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, NavSatFix
from geometry_msgs.msg import Twist
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
from collections import deque

class MultiSensorGazeboNode(Node):
    def __init__(self):
        super().__init__('multi_sensor_gazebo_node')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/custom_robot/cmd_vel', 10)

        # Subscribers for multiple sensors
        self.laser_sub = self.create_subscription(
            LaserScan, '/custom_robot/scan', self.laser_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/custom_robot/imu/data', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/custom_robot/gps/fix', self.gps_callback, 10)

        # Timer for sensor fusion and control
        self.fusion_timer = self.create_timer(0.05, self.sensor_fusion_loop)

        # Data storage
        self.laser_data = None
        self.imu_data = None
        self.gps_data = None
        self.sensor_history = deque(maxlen=100)

        # State estimation
        self.estimated_position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.estimated_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion
        self.estimated_velocity = np.array([0.0, 0.0, 0.0])  # vx, vy, vz

        self.get_logger().info('Multi-Sensor Gazebo Node initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg
        self.get_logger().debug(f'Laser: {len(msg.ranges)} ranges')

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg
        # Extract orientation and angular velocity
        orientation = msg.orientation
        angular_velocity = msg.angular_velocity
        linear_acceleration = msg.linear_acceleration

        self.get_logger().debug(
            f'IMU: ({orientation.x:.3f}, {orientation.y:.3f}, {orientation.z:.3f}, {orientation.w:.3f})'
        )

    def gps_callback(self, msg):
        """Process GPS data"""
        self.gps_data = msg
        self.get_logger().debug(
            f'GPS: ({msg.latitude:.6f}, {msg.longitude:.6f}, {msg.altitude:.2f})'
        )

    def sensor_fusion_loop(self):
        """Fusion of multiple sensor data for state estimation"""
        # Store current sensor readings
        current_reading = {
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'laser': self.laser_data,
            'imu': self.imu_data,
            'gps': self.gps_data
        }
        self.sensor_history.append(current_reading)

        # Simple sensor fusion for state estimation
        if self.imu_data is not None:
            # Use IMU for orientation estimation
            self.estimated_orientation = np.array([
                self.imu_data.orientation.x,
                self.imu_data.orientation.y,
                self.imu_data.orientation.z,
                self.imu_data.orientation.w
            ])

        # If GPS is available, use it for position correction
        if self.gps_data is not None:
            # Convert GPS to local coordinates (simplified)
            # In practice, you'd use proper coordinate transformation
            self.estimated_position[0] = self.gps_data.longitude * 111320.0  # Rough conversion
            self.estimated_position[1] = self.gps_data.latitude * 111320.0
            self.estimated_position[2] = self.gps_data.altitude

        # Generate control command based on sensor fusion
        cmd = self.generate_control_command()
        self.cmd_vel_pub.publish(cmd)

    def generate_control_command(self):
        """Generate control command based on fused sensor data"""
        cmd = Twist()

        if self.laser_data is not None:
            # Get front distance (simplified)
            front_ranges = self.laser_data.ranges[:10] + self.laser_data.ranges[-10:]
            front_distance = min([r for r in front_ranges if r > 0 and r < float('inf')], default=10.0)

            if front_distance < 0.5:  # Obstacle detected
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # Turn
            else:
                cmd.linear.x = 0.3  # Move forward
                cmd.angular.z = 0.0

        return cmd

def main(args=None):
    rclpy.init(args=args)
    node = MultiSensorGazeboNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Multi-Sensor node interrupted')
    finally:
        # Stop robot
        cmd = Twist()
        node.cmd_vel_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Notes on Multi-Sensor Integration:
- Combine data from multiple sensors for robust state estimation
- Consider sensor update rates when designing fusion algorithms
- Implement proper error handling for missing sensor data
- Use appropriate coordinate frame transformations

## Performance Optimization

### Code Example: Efficient Gazebo Simulation Configuration
```xml
<!-- Optimized world configuration -->
<sdf version="1.6">
  <world name="optimized_world">
    <!-- Optimized physics settings -->
    <physics type="ode">
      <max_step_size>0.01</max_step_size>  <!-- Larger step for performance -->
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>100.0</real_time_update_rate>  <!-- Lower update rate -->
      <gravity>0 0 -9.8</gravity>
      <ode>
        <solver>
          <type>quick</type>
          <iters>20</iters>  <!-- Balance between accuracy and speed -->
          <sor>1.3</sor>
        </solver>
      </ode>
    </physics>

    <!-- Efficient model configuration -->
    <model name="efficient_robot">
      <!-- Use simple collision geometries -->
      <link name="base_link">
        <collision>
          <geometry>
            <box><size>0.5 0.3 0.2</size></box>  <!-- Simple box instead of complex mesh -->
          </geometry>
        </collision>
        <visual>
          <geometry>
            <mesh filename="package://model/mesh.stl"/>
          </geometry>
        </visual>
        <inertial>
          <mass value="10.0"/>
          <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
        </inertial>
      </link>

      <!-- Limit sensor update rates -->
      <sensor name="camera" type="camera">
        <update_rate>10</update_rate>  <!-- Lower rate for performance -->
        <camera name="head">
          <image>
            <width>320</width>  <!-- Smaller resolution -->
            <height>240</height>
            <format>R8G8B8</format>
          </image>
        </camera>
      </sensor>
    </model>
  </world>
</sdf>
```

### Notes on Performance Optimization:
- Use simpler collision geometries when possible
- Reduce sensor update rates for better performance
- Optimize mesh resolutions for visual elements
- Balance physics accuracy with simulation speed

## Best Practices

### Code Example: Gazebo Best Practices Implementation
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time

class GazeboBestPracticesNode(Node):
    def __init__(self):
        super().__init__('gazebo_best_practices')

        # Use appropriate QoS profiles for simulation
        qos_profile = QoSProfile(depth=10)
        qos_profile.durability = QoSDurabilityPolicy.VOLATILE

        # Publishers with proper QoS
        self.cmd_pub = self.create_publisher(Twist, '/robot/cmd_vel', qos_profile)
        self.status_pub = self.create_publisher(String, '/robot/status', qos_profile)

        # Subscribers with proper QoS
        self.scan_sub = self.create_subscription(
            LaserScan, '/robot/scan', self.scan_callback, qos_profile)

        # Parameters for configuration
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('control_frequency', 20)

        # Initialize parameters
        self.safety_distance = self.get_parameter('safety_distance').value
        self.max_speed = self.get_parameter('max_linear_speed').value

        # Control timer with specified frequency
        control_freq = self.get_parameter('control_frequency').value
        self.control_timer = self.create_timer(1.0/control_freq, self.control_loop)

        # State variables
        self.scan_data = None
        self.last_control_time = time.time()
        self.emergency_stop = False

        self.get_logger().info('Gazebo Best Practices Node initialized')

    def scan_callback(self, msg):
        """Process scan data with validation"""
        try:
            # Validate scan data
            if len(msg.ranges) == 0:
                self.get_logger().warn('Received empty scan data')
                return

            # Check for invalid ranges
            valid_ranges = [r for r in msg.ranges if 0 < r < float('inf')]
            if not valid_ranges:
                self.get_logger().warn('No valid ranges in scan data')
                return

            self.scan_data = msg
        except Exception as e:
            self.get_logger().error(f'Scan callback error: {e}')

    def control_loop(self):
        """Main control loop with error handling"""
        try:
            # Publish status
            status_msg = String()
            status_msg.data = f"Running - Last update: {time.time():.2f}"
            self.status_pub.publish(status_msg)

            # Generate and publish command
            cmd = self.generate_safe_command()
            self.cmd_pub.publish(cmd)

            self.last_control_time = time.time()

        except Exception as e:
            self.get_logger().error(f'Control loop error: {e}')
            self.emergency_stop = True
            self.emergency_stop_procedure()

    def generate_safe_command(self):
        """Generate safe command based on sensor data"""
        cmd = Twist()

        if self.emergency_stop:
            return cmd  # Stop command

        if self.scan_data is not None:
            # Find minimum distance
            min_distance = min([r for r in self.scan_data.ranges
                              if 0 < r < float('inf')], default=float('inf'))

            if min_distance < self.safety_distance:
                # Emergency stop
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # Turn away from obstacle
            else:
                # Normal operation
                cmd.linear.x = min(self.max_speed, 0.3)
                cmd.angular.z = 0.0

        return cmd

    def emergency_stop_procedure(self):
        """Emergency stop procedure"""
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self.get_logger().error('EMERGENCY STOP ACTIVATED')

def main(args=None):
    rclpy.init(args=args)
    node = GazeboBestPracticesNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Gazebo Best Practices Node interrupted')
    finally:
        # Always stop the robot safely
        cmd = Twist()
        node.cmd_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices Summary:
- Use appropriate QoS profiles for simulation data
- Implement comprehensive error handling
- Validate sensor data before processing
- Implement emergency stop procedures
- Use parameters for easy configuration
- Always stop robot safely before shutdown

## Next Steps
In the next chapter, we'll explore Unity integration for high-fidelity visualization.