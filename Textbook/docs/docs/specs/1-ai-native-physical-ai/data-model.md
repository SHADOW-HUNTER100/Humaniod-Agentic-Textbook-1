# Data Model: AI-Native Software Development & Physical AI

## Core Entities

### Research Paper
- **Fields**:
  - id: UUID
  - title: string
  - abstract: text (max 5000 chars)
  - word_count: integer (5000-7000 range)
  - status: enum (draft, review, approved, published)
  - created_date: datetime
  - last_modified: datetime
  - authors: array of Author objects
  - citations: array of Citation objects
- **Validation**:
  - Word count must be between 5000-7000
  - At least 15 sources with 50%+ peer-reviewed
  - Flesch-Kincaid score between 10-12
- **Relationships**: Contains multiple Section entities

### Section
- **Fields**:
  - id: UUID
  - title: string
  - content: text
  - section_type: enum (overview, foundation, analysis, synthesis, appendix)
  - order: integer
  - readability_score: float (10.0-12.0 range)
  - created_date: datetime
  - last_modified: datetime
- **Validation**:
  - Content must pass plagiarism check (0% tolerance)
  - Must cite at least 1 source per 500 words
- **Relationships**: Belongs to Research Paper, contains multiple Citation entities

### Citation
- **Fields**:
  - id: UUID
  - source_type: enum (journal, conference, preprint, whitepaper, technical_report)
  - title: string
  - authors: array of strings
  - publication_date: date
  - doi: string (optional)
  - url: string (optional)
  - is_peer_reviewed: boolean
  - citation_text: string (APA 7th format)
  - retrieved_date: date
- **Validation**:
  - Must be properly formatted in APA 7th edition
  - Peer reviewed flag must match source type
- **Relationships**: Belongs to Section or Research Paper

### Author
- **Fields**:
  - id: UUID
  - first_name: string
  - last_name: string
  - institution: string
  - email: string (valid email format)
  - orcid: string (optional)
- **Validation**:
  - Email must be valid format
  - Institution must be non-empty
- **Relationships**: Associated with Research Paper

### Simulation Environment
- **Fields**:
  - id: UUID
  - name: string
  - description: text
  - platform: enum (gazebo, isaac_sim, unity)
  - physics_engine: string
  - complexity_level: enum (low, medium, high)
  - computational_requirements: object (gpu, ram, cpu)
- **Validation**:
  - Computational requirements must be realistic
  - Platform must be one of supported options
- **Relationships**: Used by multiple Robot Model entities

### Robot Model
- **Fields**:
  - id: UUID
  - name: string
  - manufacturer: string
  - model_type: enum (humanoid, quadruped, wheeled, custom)
  - urdf_path: string
  - joint_count: integer
  - sensor_config: array of Sensor objects
  - control_framework: enum (ros2, custom, hybrid)
- **Validation**:
  - URDF path must be valid
  - Joint count must be positive
- **Relationships**: Associated with Simulation Environment, uses multiple Sensor entities

### Sensor
- **Fields**:
  - id: UUID
  - name: string
  - type: enum (camera, lidar, imu, depth, microphone, force_torque)
  - specifications: object (resolution, range, accuracy, etc.)
  - ros_topic: string
  - update_rate: float (Hz)
- **Validation**:
  - Update rate must be positive
  - ROS topic must follow ROS naming conventions
- **Relationships**: Belongs to Robot Model

### ROS Package
- **Fields**:
  - id: UUID
  - package_name: string
  - version: string (semantic versioning)
  - dependencies: array of strings
  - description: text
  - nodes: array of Node objects
  - launch_files: array of strings
- **Validation**:
  - Package name must follow ROS naming conventions
  - Version must follow semantic versioning
- **Relationships**: Part of ROS Ecosystem

### Reproducibility Package
- **Fields**:
  - id: UUID
  - name: string
  - description: text
  - components: array of strings (docker_images, datasets, configs, scripts)
  - docker_images: array of strings
  - dataset_links: array of strings
  - setup_instructions: text
  - validation_script: string
- **Validation**:
  - All components must be accessible
  - Validation script must exist and be executable
- **Relationships**: Associated with Research Paper