---
sidebar_position: 2
---

# Installation Guide

This guide walks you through setting up the development environment for AI-Native Software Development & Physical AI project.

## System Requirements

### Minimum Specifications
- **Operating System**: Ubuntu 22.04 LTS or Windows 11 with WSL2
- **CPU**: Intel i7 / AMD Ryzen 7 or better
- **RAM**: 32GB minimum, 64GB recommended
- **GPU**: NVIDIA RTX 4070 Ti or better with CUDA support
- **Storage**: 500GB SSD minimum

### Recommended Specifications
- **CPU**: Intel i9 / AMD Ryzen 9
- **RAM**: 64GB or more
- **GPU**: NVIDIA RTX 4080, RTX 4090, or RTX 6000 Ada
- **Storage**: 1TB NVMe SSD

## Software Dependencies

### 1. ROS 2 Humble Hawksbill
```bash
# For Ubuntu
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt install ros-humble-desktop
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

### 2. NVIDIA Isaac Sim
Download from NVIDIA Developer Zone and follow installation guide for your GPU. Requires NVIDIA GPU drivers 535+ and CUDA 12.0+.

### 3. Python Environment
```bash
# Install Python 3.11+
sudo apt install python3.11 python3.11-venv python3.11-dev

# Create virtual environment
python3.11 -m venv ~/physical-ai-env
source ~/physical-ai-env/bin/activate
pip install --upgrade pip setuptools wheel
```

### 4. LaTeX and PDF Generation
```bash
# For research paper generation
sudo apt install texlive-full biber
```

### 5. Node.js and Docusaurus
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

## Project Setup

### 1. Clone the Repository
```bash
git clone https://github.com/humanod-agents/humanod-agents-book.git
cd humanod-agents-book
```

### 2. Install Python Dependencies
```bash
source ~/physical-ai-env/bin/activate  # Activate virtual environment
pip install -r requirements.txt
```

### 3. Set Up ROS 2 Workspace
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 4. Install Citation Management
```bash
# Install Zotero for citation management
sudo snap install zotero --edge --classic
```

## Verification

### 1. Verify ROS 2 Installation
```bash
source /opt/ros/humble/setup.bash
ros2 --version
```

### 2. Verify Python Dependencies
```bash
source ~/physical-ai-env/bin/activate
python -c "import rclpy; print('ROS 2 Python bindings OK')"
python -c "import torch; print('PyTorch OK')"
```

### 3. Verify Simulation Environment
```bash
# For Isaac Sim, run:
isaac-sim --/isaac/omniverse/user=YOUR_USERNAME --/isaac/omniverse/pass=YOUR_PASSWORD
```

## Hardware Setup

### 1. Unitree Robots (Go2, G1, etc.)
Connect to your Unitree robot and verify communication:
```bash
# Example for Unitree Go2
ros2 run unitree_ros unitree_driver
```

### 2. Intel RealSense Cameras
Verify camera detection:
```bash
realsense-viewer
```

### 3. NVIDIA Jetson Kits
Flash Jetson with appropriate image and install Isaac ROS packages:
```bash
# Follow NVIDIA Isaac ROS installation guide
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
```

## Environment Configuration

Create a configuration file for your setup:

```bash
# Create config directory
mkdir -p ~/physical-ai-config

# Create environment configuration
cat > ~/physical-ai-config/env.sh << EOF
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
export ISAAC_SIM_PATH="/path/to/isaac-sim"
export PHYSICAL_AI_PROJECT_ROOT="$(pwd)"
EOF

# Add to your shell profile
echo "source ~/physical-ai-config/env.sh" >> ~/.bashrc
```

## Running Tests

### 1. Unit Tests
```bash
source ~/physical-ai-env/bin/activate
python -m pytest tests/unit/
```

### 2. Simulation Tests
```bash
# Run Gazebo simulation smoke test
ros2 launch your_robot_bringup gazebo.launch.py
```

### 3. Integration Tests
```bash
# Run end-to-end tests
python -m tests.integration.simulation_smoke_test
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Ensure NVIDIA drivers are up to date and match CUDA version requirements
2. **ROS 2 Communication**: Check that ROS_DOMAIN_ID is consistent across all terminals
3. **Simulation Performance**: Close unnecessary applications and ensure sufficient GPU resources
4. **Python Dependencies**: Use virtual environment to avoid conflicts

Continue with the **[Quickstart Guide](quickstart.md)** to run your first Physical AI simulation.