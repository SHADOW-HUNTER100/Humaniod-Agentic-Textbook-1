---
sidebar_label: 'Chapter 3: Unity Integration'
sidebar_position: 3
---

# Chapter 3: Unity Integration

## Learning Objectives
- Understand Unity's role in robotics simulation
- Set up Unity for high-fidelity robot simulation
- Integrate Unity with ROS 2 using ROS# or similar tools
- Create realistic environments and sensor simulation

## Introduction to Unity for Robotics
Unity is a powerful game engine that provides high-fidelity 3D rendering capabilities. In robotics, Unity serves as a platform for creating photorealistic simulation environments that can be used for training computer vision algorithms, testing perception systems, and creating immersive visualization experiences.

### Notes on Unity's Advantages:
- **Photorealistic rendering**: High-quality PBR materials and lighting
- **Real-time performance**: Optimized for real-time rendering
- **Extensive asset ecosystem**: Large library of 3D models and environments
- **Cross-platform support**: Deploy to multiple platforms
- **Scripting flexibility**: C# scripting with extensive APIs

### Code Example: Basic Unity ROS 2 Setup
```csharp
using ROS2;
using UnityEngine;
using System.Collections;
using std_msgs = RosMessageTypes.Std;

public class UnityROS2Setup : MonoBehaviour
{
    [Header("ROS 2 Connection Settings")]
    public string rosMasterUri = "http://localhost:11311";
    public string rosHostname = "localhost";
    public string topicName = "unity_data";

    private ROS2UnityComponent ros2Unity;
    private Publisher<std_msgs.Msg.String> publisher;
    private Subscriber<std_msgs.Msg.String> subscriber;

    void Start()
    {
        InitializeROS2Connection();
    }

    private void InitializeROS2Connection()
    {
        // Get or add the ROS2UnityComponent
        ros2Unity = GetComponent<ROS2UnityComponent>();

        if (ros2Unity == null)
        {
            ros2Unity = gameObject.AddComponent<ROS2UnityComponent>();
        }

        // Configure ROS 2 settings
        ros2Unity.ROS2UnitySettings.protocol = ROS2ProjectProtocol.TCPv0;
        ros2Unity.ROS2UnitySettings.domainId = 0; // Default domain

        // Initialize the connection
        ros2Unity.Init();

        // Wait for connection to be established
        StartCoroutine(WaitForConnection());
    }

    private IEnumerator WaitForConnection()
    {
        // Wait until ROS 2 is ready
        while (!ros2Unity.Ok())
        {
            Debug.Log("Waiting for ROS 2 connection...");
            yield return new WaitForSeconds(1.0f);
        }

        Debug.Log("ROS 2 connection established!");

        // Create publisher and subscriber after connection is ready
        SetupPublisher();
        SetupSubscriber();
    }

    private void SetupPublisher()
    {
        publisher = ros2Unity.node.CreatePublisher<std_msgs.Msg.String>(topicName);
        Debug.Log($"Publisher created for topic: {topicName}");
    }

    private void SetupSubscriber()
    {
        string commandTopic = "robot_commands";
        subscriber = ros2Unity.node.CreateSubscription<std_msgs.Msg.String>(
            commandTopic,
            CommandCallback
        );
        Debug.Log($"Subscriber created for topic: {commandTopic}");
    }

    private void CommandCallback(std_msgs.Msg.String msg)
    {
        Debug.Log($"Received command: {msg.Data}");
        ProcessRobotCommand(msg.Data);
    }

    private void ProcessRobotCommand(string command)
    {
        // Process the command received from ROS 2
        switch (command.ToLower())
        {
            case "move_forward":
                MoveRobot(Vector3.forward);
                break;
            case "turn_left":
                RotateRobot(-90f);
                break;
            case "turn_right":
                RotateRobot(90f);
                break;
            default:
                Debug.LogWarning($"Unknown command: {command}");
                break;
        }
    }

    private void MoveRobot(Vector3 direction)
    {
        transform.Translate(direction * Time.deltaTime * 2.0f);
    }

    private void RotateRobot(float angle)
    {
        transform.Rotate(Vector3.up, angle);
    }

    void Update()
    {
        // Send periodic status updates
        if (publisher != null && Time.time % 2.0f < Time.deltaTime)
        {
            var statusMsg = new std_msgs.Msg.String();
            statusMsg.Data = $"Robot position: {transform.position}";
            publisher.Publish(statusMsg);
        }
    }

    void OnDestroy()
    {
        // Clean up ROS 2 connections
        if (ros2Unity != null)
        {
            ros2Unity.Shutdown();
        }
    }
}
```

### Notes on Unity ROS 2 Setup:
- Use `ROS2UnityComponent` for managing ROS 2 connections
- Implement connection waiting with coroutines
- Always clean up connections in `OnDestroy`
- Use appropriate message types for data exchange

## Unity Robotics Ecosystem

### Unity Robotics Hub
The Unity Robotics Hub provides centralized access to robotics packages:

```csharp
// Example of using Unity Robotics Hub packages
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.Core;
using UnityEngine;

public class RoboticsHubExample : MonoBehaviour
{
    [SerializeField] private RosTopic publisherTopic;
    [SerializeField] private RosTopic subscriberTopic;

    private ROSConnection ros;
    private float publishFrequency = 10.0f; // Hz
    private float lastPublishTime = 0.0f;

    void Start()
    {
        // Get the ROS connection
        ros = ROSConnection.GetOrCreateInstance();

        // Register topics
        ros.RegisterPublisher<std_msgs.Msg.String>(publisherTopic);
        ros.RegisterSubscriber<std_msgs.Msg.String>(subscriberTopic, OnMessageReceived);
    }

    void Update()
    {
        // Publish messages at specified frequency
        if (Time.time - lastPublishTime > 1.0f / publishFrequency)
        {
            PublishRobotData();
            lastPublishTime = Time.time;
        }
    }

    private void PublishRobotData()
    {
        var robotData = new std_msgs.Msg.String
        {
            Data = $"Robot data at time: {Time.time}"
        };

        ros.Publish(publisherTopic, robotData);
    }

    private void OnMessageReceived(std_msgs.Msg.String msg)
    {
        Debug.Log($"Received message: {msg.Data}");
        ProcessReceivedMessage(msg.Data);
    }

    private void ProcessReceivedMessage(string data)
    {
        // Handle the received message
        if (data.Contains("command"))
        {
            ExecuteRobotCommand(data);
        }
    }

    private void ExecuteRobotCommand(string command)
    {
        Debug.Log($"Executing command: {command}");
        // Implement command execution logic here
    }
}
```

### Unity Machine Learning Agents (ML-Agents)
ML-Agents enables training AI in Unity environments:

```csharp
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class RobotAgent : Agent
{
    [Header("Robot Settings")]
    public float moveSpeed = 5.0f;
    public float rotationSpeed = 180.0f;

    [Header("Environment")]
    public Transform target;
    public float maxDistance = 10.0f;

    private Rigidbody rb;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        // Reset robot position
        transform.position = new Vector3(
            Random.Range(-maxDistance, maxDistance),
            0.5f,
            Random.Range(-maxDistance, maxDistance)
        );

        // Reset target position
        target.position = new Vector3(
            Random.Range(-maxDistance, maxDistance),
            0.5f,
            Random.Range(-maxDistance, maxDistance)
        );
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Add observations for the ML agent
        sensor.AddObservation(transform.position);           // Robot position
        sensor.AddObservation(target.position);             // Target position
        sensor.AddObservation(rb.velocity);                 // Robot velocity
        sensor.AddObservation(transform.rotation);          // Robot rotation
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Process actions from the ML agent
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        float rotate = actions.ContinuousActions[2];

        // Apply movement
        Vector3 movement = new Vector3(moveX, 0, moveZ) * moveSpeed * Time.deltaTime;
        rb.MovePosition(rb.position + movement);

        // Apply rotation
        transform.Rotate(Vector3.up, rotate * rotationSpeed * Time.deltaTime);

        // Calculate distance to target
        float distanceToTarget = Vector3.Distance(transform.position, target.position);

        // Set rewards based on distance
        if (distanceToTarget < 1.0f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else
        {
            SetReward(-distanceToTarget * 0.01f); // Negative reward for distance
        }

        // End episode if robot goes too far
        if (transform.position.magnitude > maxDistance * 2)
        {
            SetReward(-1.0f);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // Manual control for testing
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Horizontal"); // Move X
        continuousActionsOut[1] = Input.GetAxis("Vertical");   // Move Z
        continuousActionsOut[2] = Input.GetKey(KeyCode.Q) ? -1f :
                  Input.GetKey(KeyCode.E) ? 1f : 0f;           // Rotate
    }
}
```

### Unity Perception Package
The Perception package enables synthetic data generation:

```csharp
using UnityEngine;
using Unity.Perception.GroundTruth;
using Unity.Perception.Randomization.Parameters;
using Unity.Perception.Randomization.Samplers;

public class PerceptionSetup : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera perceptionCamera;
    public float captureFrequency = 1.0f;

    [Header("Randomization")]
    public CategoricalParameter materialParameter;
    public FloatParameter lightingParameter;

    private float lastCaptureTime = 0.0f;
    private SemanticSegmentationLabeler labeler;

    void Start()
    {
        SetupPerceptionCamera();
        SetupRandomization();
    }

    private void SetupPerceptionCamera()
    {
        if (perceptionCamera == null)
        {
            perceptionCamera = GetComponent<Camera>();
        }

        // Add semantic segmentation labeler
        labeler = perceptionCamera.gameObject.AddComponent<SemanticSegmentationLabeler>();

        // Configure camera for perception
        perceptionCamera.depthTextureMode = DepthTextureMode.Depth;
        perceptionCamera.allowDynamicResolution = true;
    }

    private void SetupRandomization()
    {
        // Set up domain randomization parameters
        if (materialParameter != null)
        {
            materialParameter.values = new[] { "MaterialA", "MaterialB", "MaterialC" };
        }

        if (lightingParameter != null)
        {
            lightingParameter.min = 0.5f;
            lightingParameter.max = 2.0f;
        }
    }

    void Update()
    {
        // Capture data at specified frequency
        if (Time.time - lastCaptureTime > 1.0f / captureFrequency)
        {
            CapturePerceptionData();
            lastCaptureTime = Time.time;
        }
    }

    private void CapturePerceptionData()
    {
        // Trigger data capture
        var captureData = new PerceptionOutput
        {
            timestamp = Time.time,
            cameraPose = perceptionCamera.transform.position,
            semanticSegmentation = GetSemanticSegmentation()
        };

        // Process captured data
        ProcessPerceptionCapture(captureData);
    }

    private Texture2D GetSemanticSegmentation()
    {
        // Get semantic segmentation texture
        return labeler.GetSemanticSegmentationTexture();
    }

    private void ProcessPerceptionCapture(PerceptionOutput data)
    {
        Debug.Log($"Captured perception data at {data.timestamp}");
        // Save or process the captured data
    }
}

[System.Serializable]
public struct PerceptionOutput
{
    public float timestamp;
    public Vector3 cameraPose;
    public Texture2D semanticSegmentation;
}
```

## Setting Up Unity with ROS 2

### Unity ROS TCP Connector Integration
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    [Header("Robot Configuration")]
    public float linearSpeed = 2.0f;
    public float angularSpeed = 90.0f; // degrees per second

    [Header("ROS Topics")]
    public string cmdVelTopic = "/cmd_vel";
    public string odomTopic = "/odom";
    public string laserScanTopic = "/scan";

    private ROSConnection ros;
    private Twist currentCommand;
    private Vector3 robotPosition;
    private Quaternion robotRotation;

    void Start()
    {
        // Initialize ROS connection
        ros = ROSConnection.GetOrCreateInstance();

        // Subscribe to command topic
        ros.Subscribe<Twist>(cmdVelTopic, ProcessCommand);

        // Initialize robot state
        currentCommand = new Twist();
        robotPosition = transform.position;
        robotRotation = transform.rotation;

        // Start publishing odometry
        InvokeRepeating("PublishOdometry", 0.0f, 0.1f); // 10 Hz
    }

    void Update()
    {
        // Update robot position based on current command
        UpdateRobotPosition();
    }

    private void ProcessCommand(Twist cmd)
    {
        currentCommand = cmd;
    }

    private void UpdateRobotPosition()
    {
        // Apply linear velocity
        Vector3 linearVelocity = transform.forward * (float)currentCommand.linear.x;
        robotPosition += linearVelocity * Time.deltaTime;

        // Apply angular velocity (around Y axis)
        float angularVelocity = (float)currentCommand.angular.z;
        robotRotation *= Quaternion.Euler(0, angularVelocity * Time.deltaTime * Mathf.Rad2Deg, 0);

        // Update transform
        transform.position = robotPosition;
        transform.rotation = robotRotation;
    }

    private void PublishOdometry()
    {
        // Create odometry message
        var odomMsg = new RosMessageTypes.Nav.OdometryMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg { sec = (int)Time.time, nanosec = 0 },
                frame_id = "odom"
            },
            child_frame_id = "base_link",
            pose = new geometry_msgs.PoseWithCovarianceMsg
            {
                pose = new geometry_msgs.PoseMsg
                {
                    position = new geometry_msgs.PointMsg
                    {
                        x = transform.position.x,
                        y = transform.position.y,
                        z = transform.position.z
                    },
                    orientation = new geometry_msgs.QuaternionMsg
                    {
                        x = transform.rotation.x,
                        y = transform.rotation.y,
                        z = transform.rotation.z,
                        w = transform.rotation.w
                    }
                },
                covariance = new double[36] // Initialize covariance matrix
            },
            twist = new geometry_msgs.TwistWithCovarianceMsg
            {
                twist = new geometry_msgs.TwistMsg
                {
                    linear = new geometry_msgs.Vector3Msg
                    {
                        x = (float)currentCommand.linear.x,
                        y = (float)currentCommand.linear.y,
                        z = (float)currentCommand.linear.z
                    },
                    angular = new geometry_msgs.Vector3Msg
                    {
                        x = (float)currentCommand.angular.x,
                        y = (float)currentCommand.angular.y,
                        z = (float)currentCommand.angular.z
                    }
                },
                covariance = new double[36] // Initialize covariance matrix
            }
        };

        ros.Publish(odomTopic, odomMsg);
    }
}
```

### Code Example: Advanced Unity-ROS Integration with Multiple Sensors
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using System.Collections.Generic;

public class AdvancedUnityRobot : MonoBehaviour
{
    [Header("Sensors Configuration")]
    public Camera frontCamera;
    public Camera depthCamera;
    public Transform lidarSensor;
    public Transform imuSensor;

    [Header("Topics")]
    public string imageTopic = "/camera/image_raw";
    public string depthTopic = "/camera/depth/image_raw";
    public string lidarTopic = "/scan";
    public string imuTopic = "/imu/data";

    private ROSConnection ros;
    private float sensorUpdateRate = 30.0f; // Hz
    private float lastSensorUpdate = 0.0f;

    // Sensor data buffers
    private RenderTexture cameraRenderTexture;
    private RenderTexture depthRenderTexture;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Initialize cameras if not set
        if (frontCamera == null) frontCamera = GetComponent<Camera>();
        if (depthCamera == null) depthCamera = GetComponent<Camera>();

        // Create render textures for sensor simulation
        InitializeRenderTextures();

        // Start sensor data publishing
        InvokeRepeating("PublishAllSensors", 0.0f, 1.0f / sensorUpdateRate);
    }

    private void InitializeRenderTextures()
    {
        int width = 640;
        int height = 480;

        // Create render texture for RGB camera
        cameraRenderTexture = new RenderTexture(width, height, 24);
        cameraRenderTexture.Create();

        // Create render texture for depth camera
        depthRenderTexture = new RenderTexture(width, height, 24);
        depthRenderTexture.Create();

        // Assign to cameras
        if (frontCamera != null)
        {
            frontCamera.targetTexture = cameraRenderTexture;
        }
        if (depthCamera != null)
        {
            depthCamera.targetTexture = depthRenderTexture;
        }
    }

    private void PublishAllSensors()
    {
        // Publish camera image
        if (frontCamera != null)
        {
            PublishCameraImage();
        }

        // Publish depth image
        if (depthCamera != null)
        {
            PublishDepthImage();
        }

        // Publish LiDAR data
        PublishLidarData();

        // Publish IMU data
        PublishImuData();
    }

    private void PublishCameraImage()
    {
        // Read pixels from render texture
        RenderTexture.active = cameraRenderTexture;
        Texture2D texture2D = new Texture2D(cameraRenderTexture.width, cameraRenderTexture.height, TextureFormat.RGB24, false);
        texture2D.ReadPixels(new Rect(0, 0, cameraRenderTexture.width, cameraRenderTexture.height), 0, 0);
        texture2D.Apply();

        // Convert to ROS image message
        var imageMsg = new ImageMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg { sec = (int)Time.time, nanosec = 0 },
                frame_id = "camera_frame"
            },
            height = (uint)texture2D.height,
            width = (uint)texture2D.width,
            encoding = "rgb8",
            is_bigendian = 0,
            step = (uint)(texture2D.width * 3), // 3 bytes per pixel for RGB
            data = texture2D.GetRawTextureData<byte>().ToArray()
        };

        ros.Publish(imageTopic, imageMsg);
        Destroy(texture2D); // Clean up temporary texture
    }

    private void PublishDepthImage()
    {
        // Similar to camera image but for depth data
        RenderTexture.active = depthRenderTexture;
        Texture2D depthTexture = new Texture2D(depthRenderTexture.width, depthRenderTexture.height, TextureFormat.RGB24, false);
        depthTexture.ReadPixels(new Rect(0, 0, depthRenderTexture.width, depthRenderTexture.height), 0, 0);
        depthTexture.Apply();

        // Convert depth data to appropriate format
        var depthMsg = new ImageMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg { sec = (int)Time.time, nanosec = 0 },
                frame_id = "depth_frame"
            },
            height = (uint)depthTexture.height,
            width = (uint)depthTexture.width,
            encoding = "32FC1", // 32-bit float, single channel
            is_bigendian = 0,
            step = (uint)(depthTexture.width * 4), // 4 bytes per float
            data = new byte[depthTexture.width * depthTexture.height * 4] // Placeholder
        };

        ros.Publish(depthTopic, depthMsg);
        Destroy(depthTexture);
    }

    private void PublishLidarData()
    {
        // Simulate LiDAR data using raycasting
        int numRays = 360;
        float[] ranges = new float[numRays];
        float angleIncrement = 2 * Mathf.PI / numRays;

        for (int i = 0; i < numRays; i++)
        {
            float angle = i * angleIncrement;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            // Perform raycast in this direction
            RaycastHit hit;
            if (Physics.Raycast(lidarSensor.position, lidarSensor.TransformDirection(direction), out hit, 10.0f))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = 10.0f; // Max range
            }
        }

        var lidarMsg = new LaserScanMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg { sec = (int)Time.time, nanosec = 0 },
                frame_id = "lidar_frame"
            },
            angle_min = -Mathf.PI,
            angle_max = Mathf.PI,
            angle_increment = angleIncrement,
            time_increment = 0.0f,
            scan_time = 0.1f,
            range_min = 0.1f,
            range_max = 10.0f,
            ranges = ranges,
            intensities = new float[numRays] // Initialize with zeros
        };

        ros.Publish(lidarTopic, lidarMsg);
    }

    private void PublishImuData()
    {
        var imuMsg = new ImuMsg
        {
            header = new std_msgs.HeaderMsg
            {
                stamp = new builtin_interfaces.TimeMsg { sec = (int)Time.time, nanosec = 0 },
                frame_id = "imu_frame"
            },
            orientation = new geometry_msgs.QuaternionMsg
            {
                x = transform.rotation.x,
                y = transform.rotation.y,
                z = transform.rotation.z,
                w = transform.rotation.w
            },
            angular_velocity = new geometry_msgs.Vector3Msg
            {
                x = Random.Range(-0.1f, 0.1f), // Add some noise
                y = Random.Range(-0.1f, 0.1f),
                z = Random.Range(-0.1f, 0.1f)
            },
            linear_acceleration = new geometry_msgs.Vector3Msg
            {
                x = Random.Range(-0.5f, 0.5f), // Add some noise
                y = Random.Range(-0.5f, 0.5f),
                z = Physics.gravity.y + Random.Range(-0.1f, 0.1f)
            }
        };

        ros.Publish(imuTopic, imuMsg);
    }
}
```

### Notes on Advanced Integration:
- Use RenderTextures for efficient camera simulation
- Implement proper resource cleanup for textures
- Add realistic noise models to sensor data
- Use appropriate data types for different sensor modalities

## Creating Realistic Environments

### Code Example: Procedural Environment Generation
```csharp
using UnityEngine;
using System.Collections.Generic;

public class ProceduralEnvironment : MonoBehaviour
{
    [Header("Environment Settings")]
    public int gridSize = 20;
    public float cellSize = 2.0f;
    public GameObject[] obstaclePrefabs;
    public Material[] floorMaterials;

    [Header("Lighting")]
    public Light mainLight;
    public float minLightIntensity = 0.5f;
    public float maxLightIntensity = 1.5f;

    [Header("Randomization")]
    public bool randomizeEnvironment = true;
    public float obstacleDensity = 0.2f;

    private List<GameObject> spawnedObjects = new List<GameObject>();

    void Start()
    {
        if (randomizeEnvironment)
        {
            GenerateRandomEnvironment();
        }
        else
        {
            GenerateFixedEnvironment();
        }

        RandomizeLighting();
    }

    private void GenerateRandomEnvironment()
    {
        for (int x = 0; x < gridSize; x++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                Vector3 position = new Vector3(
                    x * cellSize - (gridSize * cellSize) / 2,
                    0,
                    z * cellSize - (gridSize * cellSize) / 2
                );

                // Add floor with random material
                GameObject floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
                floor.transform.position = position;
                floor.transform.localScale = Vector3.one * cellSize / 10; // Plane is 10x10 units
                floor.GetComponent<Renderer>().material = floorMaterials[
                    Random.Range(0, floorMaterials.Length)
                ];

                // Add obstacles based on density
                if (Random.value < obstacleDensity)
                {
                    SpawnRandomObstacle(position);
                }
            }
        }
    }

    private void SpawnRandomObstacle(Vector3 basePosition)
    {
        if (obstaclePrefabs.Length == 0) return;

        // Offset slightly from grid center
        Vector3 position = basePosition + new Vector3(
            Random.Range(-cellSize * 0.3f, cellSize * 0.3f),
            0,
            Random.Range(-cellSize * 0.3f, cellSize * 0.3f)
        );

        GameObject obstacle = Instantiate(
            obstaclePrefabs[Random.Range(0, obstaclePrefabs.Length)],
            position,
            Quaternion.Euler(0, Random.Range(0, 360), 0)
        );

        // Randomize scale
        float scale = Random.Range(0.5f, 2.0f);
        obstacle.transform.localScale *= scale;

        spawnedObjects.Add(obstacle);
    }

    private void RandomizeLighting()
    {
        if (mainLight != null)
        {
            mainLight.intensity = Random.Range(minLightIntensity, maxLightIntensity);

            // Randomize light color temperature
            float temperature = Random.Range(4000f, 8000f);
            mainLight.color = GetColorForTemperature(temperature);
        }
    }

    private Color GetColorForTemperature(float temperature)
    {
        // Simplified color temperature to RGB conversion
        temperature /= 100f;

        float r, g, b;

        if (temperature <= 66)
        {
            r = 255;
            g = temperature;
            g = 99.4708025861f * Mathf.Log(g) - 161.1195681661f;
        }
        else
        {
            r = temperature - 60;
            r = 329.698727446f * Mathf.Pow(r, -0.1332047592f);
            g = temperature - 60;
            g = 288.1221695283f * Mathf.Pow(g, -0.0755148492f);
        }

        if (temperature >= 66)
        {
            b = 255;
        }
        else if (temperature <= 19)
        {
            b = 0;
        }
        else
        {
            b = temperature - 10;
            b = 138.5177312231f * Mathf.Log(b) - 305.0447927307f;
        }

        return new Color(
            Mathf.Clamp01(r / 255f),
            Mathf.Clamp01(g / 255f),
            Mathf.Clamp01(b / 255f)
        );
    }

    private void GenerateFixedEnvironment()
    {
        // Create a fixed environment layout
        // This could be a specific maze, room layout, etc.
    }

    void OnDestroy()
    {
        // Clean up spawned objects
        foreach (GameObject obj in spawnedObjects)
        {
            if (obj != null)
            {
                DestroyImmediate(obj);
            }
        }
        spawnedObjects.Clear();
    }
}
```

### Notes on Environment Generation:
- Use procedural generation for varied training environments
- Randomize lighting conditions for robust perception
- Include diverse obstacle types and materials
- Implement proper cleanup to avoid memory leaks

## Sensor Simulation in Unity

### Code Example: Advanced Sensor Simulation with Noise Models
```csharp
using UnityEngine;
using System.Collections.Generic;

public class AdvancedSensorSimulation : MonoBehaviour
{
    [Header("Camera Sensor")]
    public Camera sensorCamera;
    [Range(0.0f, 1.0f)] public float cameraNoiseLevel = 0.1f;
    [Range(0.0f, 1.0f)] public float cameraBlurAmount = 0.05f;

    [Header("LiDAR Sensor")]
    public Transform lidarOrigin;
    public int lidarRays = 360;
    public float lidarRange = 10.0f;
    [Range(0.0f, 0.1f)] public float lidarNoise = 0.02f;

    [Header("IMU Sensor")]
    [Range(0.0f, 1.0f)] public float imuNoise = 0.05f;
    [Range(0.0f, 1.0f)] public float imuDrift = 0.001f;

    // Internal state
    private float imuDriftAccumulator = 0f;
    private Vector3 lastPosition = Vector3.zero;
    private Quaternion lastRotation = Quaternion.identity;

    void Start()
    {
        if (sensorCamera == null)
            sensorCamera = GetComponent<Camera>();
    }

    void Update()
    {
        // Update IMU drift
        imuDriftAccumulator += imuDrift * Time.deltaTime;
    }

    public float[] GetNoisyLidarData()
    {
        float[] ranges = new float[lidarRays];
        float angleIncrement = 2 * Mathf.PI / lidarRays;

        for (int i = 0; i < lidarRays; i++)
        {
            float angle = i * angleIncrement;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));

            RaycastHit hit;
            if (Physics.Raycast(lidarOrigin.position, lidarOrigin.TransformDirection(direction), out hit, lidarRange))
            {
                // Add noise to the distance measurement
                float noisyDistance = hit.distance + Random.Range(-lidarNoise, lidarNoise);
                ranges[i] = Mathf.Clamp(noisyDistance, 0.1f, lidarRange);
            }
            else
            {
                ranges[i] = lidarRange; // Max range
            }
        }

        return ranges;
    }

    public Vector3 GetNoisyIMUData()
    {
        // Calculate linear acceleration from movement
        Vector3 velocity = (transform.position - lastPosition) / Time.deltaTime;
        Vector3 acceleration = (velocity - (lastPosition - lastPosition)) / Time.deltaTime; // This is a simplified approach

        // Add noise and drift
        Vector3 noisyAcceleration = acceleration +
            new Vector3(
                Random.Range(-imuNoise, imuNoise) + imuDriftAccumulator,
                Random.Range(-imuNoise, imuNoise) + imuDriftAccumulator,
                Random.Range(-imuNoise, imuNoise) + imuDriftAccumulator
            );

        // Update state for next frame
        lastPosition = transform.position;

        return noisyAcceleration;
    }

    public Quaternion GetNoisyOrientation()
    {
        // Add noise to orientation
        Vector3 noise = new Vector3(
            Random.Range(-imuNoise, imuNoise),
            Random.Range(-imuNoise, imuNoise),
            Random.Range(-imuNoise, imuNoise)
        ) * Mathf.Deg2Rad;

        Quaternion noisyRotation = transform.rotation * Quaternion.Euler(noise);

        return noisyRotation;
    }

    public Texture2D GetNoisyCameraImage()
    {
        // Create a temporary render texture
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = sensorCamera.targetTexture;

        // Read pixels from the camera
        Texture2D image = new Texture2D(sensorCamera.targetTexture.width, sensorCamera.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, sensorCamera.targetTexture.width, sensorCamera.targetTexture.height), 0, 0);
        image.Apply();

        // Apply noise to the image
        ApplyNoiseToImage(image, cameraNoiseLevel);

        // Restore render texture
        RenderTexture.active = currentRT;

        return image;
    }

    private void ApplyNoiseToImage(Texture2D image, float noiseLevel)
    {
        Color[] pixels = image.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            // Add Gaussian noise
            float noiseX = Random.Range(-noiseLevel, noiseLevel);
            float noiseY = Random.Range(-noiseLevel, noiseLevel);
            float noiseZ = Random.Range(-noiseLevel, noiseLevel);

            pixels[i] = new Color(
                Mathf.Clamp01(pixels[i].r + noiseX),
                Mathf.Clamp01(pixels[i].g + noiseY),
                Mathf.Clamp01(pixels[i].b + noiseZ),
                pixels[i].a
            );
        }

        image.SetPixels(pixels);
        image.Apply();
    }

    // Method to get depth image with noise
    public float[,] GetNoisyDepthImage(int width = 320, int height = 240)
    {
        float[,] depthData = new float[width, height];

        // For simplicity, this is a placeholder
        // In a real implementation, you would use a depth shader or raycasting
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                // Simulate depth data with noise
                depthData[x, y] = Random.Range(0.1f, 10.0f) + Random.Range(-0.05f, 0.05f);
            }
        }

        return depthData;
    }
}
```

### Notes on Sensor Simulation:
- Add realistic noise models to sensor data
- Implement drift for IMU sensors
- Use appropriate coordinate transformations
- Consider sensor limitations and ranges

## Performance Considerations

### Code Example: Optimized Sensor Simulation
```csharp
using UnityEngine;
using System.Collections.Generic;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;

public class OptimizedSensorSimulation : MonoBehaviour
{
    [Header("Performance Settings")]
    public int sensorUpdateRate = 30; // Hz
    public bool useMultithreading = true;
    public int maxLidarRays = 1080;

    // Sensor data buffers
    private float[] lidarRanges;
    private bool[] lidarRayHits;
    private Vector3[] lidarRayDirections;

    // Performance optimization
    private float lastSensorUpdate = 0.0f;
    private int[] activeRayIndices;

    void Start()
    {
        InitializeSensorBuffers();
    }

    private void InitializeSensorBuffers()
    {
        lidarRanges = new float[maxLidarRays];
        lidarRayHits = new bool[maxLidarRays];
        lidarRayDirections = new Vector3[maxLidarRays];
        activeRayIndices = new int[maxLidarRays];

        // Pre-calculate ray directions for efficiency
        float angleIncrement = 2 * Mathf.PI / maxLidarRays;
        for (int i = 0; i < maxLidarRays; i++)
        {
            float angle = i * angleIncrement;
            lidarRayDirections[i] = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            activeRayIndices[i] = i;
        }
    }

    void Update()
    {
        // Update sensors at specified rate
        if (Time.time - lastSensorUpdate > 1.0f / sensorUpdateRate)
        {
            UpdateSensors();
            lastSensorUpdate = Time.time;
        }
    }

    private void UpdateSensors()
    {
        if (useMultithreading)
        {
            // Use Unity's Job System for multithreaded raycasting
            UpdateLidarWithJobs();
        }
        else
        {
            // Fallback to regular raycasting
            UpdateLidarRegular();
        }
    }

    private void UpdateLidarWithJobs()
    {
        // Create a job for raycasting
        var raycastJob = new RaycastJob
        {
            origins = new NativeArray<Vector3>(maxLidarRays, Allocator.TempJob),
            directions = new NativeArray<Vector3>(maxLidarRays, Allocator.TempJob),
            ranges = new NativeArray<float>(maxLidarRays, Allocator.TempJob),
            rayCount = maxLidarRays,
            maxDistance = 10.0f
        };

        // Fill arrays with data
        for (int i = 0; i < maxLidarRays; i++)
        {
            raycastJob.origins[i] = transform.position;
            raycastJob.directions[i] = transform.TransformDirection(lidarRayDirections[i]);
        }

        // Schedule the job
        var jobHandle = raycastJob.Schedule(maxLidarRays, 10);
        jobHandle.Complete();

        // Copy results back
        for (int i = 0; i < maxLidarRays; i++)
        {
            lidarRanges[i] = raycastJob.ranges[i];
        }

        // Dispose of native arrays
        raycastJob.origins.Dispose();
        raycastJob.directions.Dispose();
        raycastJob.ranges.Dispose();
    }

    private void UpdateLidarRegular()
    {
        for (int i = 0; i < maxLidarRays; i++)
        {
            Vector3 direction = transform.TransformDirection(lidarRayDirections[i]);
            RaycastHit hit;

            if (Physics.Raycast(transform.position, direction, out hit, 10.0f))
            {
                lidarRanges[i] = hit.distance;
            }
            else
            {
                lidarRanges[i] = 10.0f; // Max range
            }
        }
    }

    [BurstCompile]
    struct RaycastJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Vector3> origins;
        [ReadOnly] public NativeArray<Vector3> directions;
        public NativeArray<float> ranges;
        public int rayCount;
        public float maxDistance;

        public void Execute(int index)
        {
            if (index < rayCount)
            {
                RaycastHit hit;
                if (Physics.Raycast(origins[index], directions[index], out hit, maxDistance))
                {
                    ranges[index] = hit.distance;
                }
                else
                {
                    ranges[index] = maxDistance;
                }
            }
        }
    }

    // Get processed sensor data
    public float[] GetLidarData()
    {
        // Add noise and return processed data
        float[] noisyData = new float[lidarRanges.Length];
        for (int i = 0; i < lidarRanges.Length; i++)
        {
            noisyData[i] = lidarRanges[i] + Random.Range(-0.02f, 0.02f); // Add noise
        }
        return noisyData;
    }

    void OnDestroy()
    {
        // Clean up if needed
    }
}
```

### Notes on Performance Optimization:
- Use Unity's Job System for multithreaded sensor processing
- Limit sensor update rates based on requirements
- Use object pooling for frequently created objects
- Consider LOD (Level of Detail) for complex environments

## Best Practices

### Code Example: Unity-ROS Integration Best Practices
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using System.Collections.Generic;

public class UnityROSBestPractices : MonoBehaviour
{
    [Header("Connection Settings")]
    public string rosMasterUri = "http://localhost:11311";
    public float connectionTimeout = 10.0f;
    public bool autoReconnect = true;

    [Header("Topic Configuration")]
    public List<TopicConfig> topics = new List<TopicConfig>();

    private ROSConnection ros;
    private float connectionAttemptTime;
    private bool isConnected = false;

    [System.Serializable]
    public class TopicConfig
    {
        public string topicName;
        public string messageType;
        public bool isPublisher;
        public float publishRate = 10.0f;
    }

    void Start()
    {
        InitializeROSConnection();
    }

    private void InitializeROSConnection()
    {
        try
        {
            ros = ROSConnection.GetOrCreateInstance();
            ros.OnConnected += OnConnectionEstablished;
            ros.OnDisconnected += OnConnectionLost;

            // Attempt connection
            connectionAttemptTime = Time.time;
            ros.Initialize(rosMasterUri);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to initialize ROS connection: {e.Message}");
        }
    }

    private void OnConnectionEstablished()
    {
        isConnected = true;
        Debug.Log("ROS connection established successfully");

        // Register all configured topics
        foreach (var topic in topics)
        {
            if (topic.isPublisher)
            {
                RegisterPublisher(topic);
            }
            else
            {
                RegisterSubscriber(topic);
            }
        }
    }

    private void OnConnectionLost()
    {
        isConnected = false;
        Debug.LogWarning("ROS connection lost");

        if (autoReconnect)
        {
            Invoke(nameof(RetryConnection), 2.0f);
        }
    }

    private void RegisterPublisher(TopicConfig config)
    {
        // Register publisher based on message type
        switch (config.messageType)
        {
            case "std_msgs/String":
                ros.RegisterPublisher<std_msgs.StringMsg>(config.topicName);
                break;
            case "geometry_msgs/Twist":
                ros.RegisterPublisher<geometry_msgs.TwistMsg>(config.topicName);
                break;
            default:
                Debug.LogWarning($"Unsupported message type: {config.messageType}");
                break;
        }
    }

    private void RegisterSubscriber(TopicConfig config)
    {
        // Register subscriber based on message type
        switch (config.messageType)
        {
            case "std_msgs/String":
                ros.RegisterSubscriber<std_msgs.StringMsg>(config.topicName, HandleStringMessage);
                break;
            case "geometry_msgs/Twist":
                ros.RegisterSubscriber<geometry_msgs.TwistMsg>(config.topicName, HandleTwistMessage);
                break;
            default:
                Debug.LogWarning($"Unsupported message type: {config.messageType}");
                break;
        }
    }

    private void HandleStringMessage(std_msgs.StringMsg msg)
    {
        Debug.Log($"Received string message: {msg.data}");
        ProcessStringCommand(msg.data);
    }

    private void HandleTwistMessage(geometry_msgs.TwistMsg msg)
    {
        Debug.Log($"Received twist message: linear=({msg.linear.x}, {msg.linear.y}, {msg.linear.z})");
        ProcessTwistCommand(msg);
    }

    private void ProcessStringCommand(string command)
    {
        // Process string command
        switch (command.ToLower())
        {
            case "reset":
                ResetEnvironment();
                break;
            case "pause":
                Time.timeScale = 0f;
                break;
            case "resume":
                Time.timeScale = 1f;
                break;
        }
    }

    private void ProcessTwistCommand(geometry_msgs.TwistMsg twist)
    {
        // Process twist command for robot movement
        Vector3 linear = new Vector3((float)twist.linear.x, (float)twist.linear.y, (float)twist.linear.z);
        Vector3 angular = new Vector3((float)twist.angular.x, (float)twist.angular.y, (float)twist.angular.z);

        // Apply movement to robot
        transform.Translate(linear * Time.deltaTime);
        transform.Rotate(angular * Mathf.Rad2Deg * Time.deltaTime);
    }

    private void ResetEnvironment()
    {
        // Reset the environment to initial state
        transform.position = Vector3.zero;
        transform.rotation = Quaternion.identity;
        Time.timeScale = 1f;
    }

    private void RetryConnection()
    {
        if (!isConnected && Time.time - connectionAttemptTime < connectionTimeout)
        {
            InitializeROSConnection();
        }
    }

    void OnDestroy()
    {
        if (ros != null)
        {
            ros.OnConnected -= OnConnectionEstablished;
            ros.OnDisconnected -= OnConnectionLost;
            ros.Disconnect();
        }
    }
}
```

### Best Practices Summary:
- Implement proper connection management with timeouts
- Use appropriate error handling and recovery
- Register topics only after connection is established
- Implement graceful disconnection handling
- Use message validation to ensure data integrity
- Implement rate limiting for sensor data publishing

## Next Steps
In the next chapter, we'll explore sensor simulation in detail, covering both Gazebo and Unity implementations.