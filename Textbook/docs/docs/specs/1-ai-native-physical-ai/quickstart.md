# Quickstart Guide: AI-Native Software Development & Physical AI

## Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04 LTS (recommended) or Windows 11 with WSL2
- **CPU**: Intel i7 / AMD Ryzen 7 or better
- **RAM**: 32GB minimum, 64GB recommended
- **GPU**: NVIDIA RTX 4070 Ti or better with CUDA support
- **Storage**: 500GB SSD minimum

### Software Dependencies
1. **ROS 2 Humble Hawksbill**
   ```bash
   sudo apt update && sudo apt install software-properties-common
   sudo add-apt-repository universe
   sudo apt update && sudo apt install curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   sudo apt update && sudo apt install ros-humble-desktop
   ```

2. **NVIDIA Isaac Sim** (requires NVIDIA Developer account)
   - Download from NVIDIA Developer Zone
   - Follow installation guide for your GPU

3. **Python 3.11+**
   ```bash
   sudo apt install python3.11 python3.11-venv python3.11-dev
   ```

4. **Node.js 18+** (for Docusaurus)
   ```bash
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt install -y nodejs
   ```

5. **LaTeX** (for PDF generation)
   ```bash
   sudo apt install texlive-full biber
   ```

## Setup Process

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/humanod-agents-book.git
cd humanod-agents-book
```

### 2. Create Python Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 3. Install Project Dependencies
```bash
pip install -r requirements.txt  # Create this file with your dependencies
```

### 4. Set Up ROS 2 Workspace
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 5. Install Citation Management
```bash
# Install Zotero connector for LaTeX
sudo snap install zotero --edge --classic
```

## Running the Project

### 1. Start ROS 2 Environment
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### 2. Launch Simulation Environment
```bash
# For Gazebo simulation
ros2 launch your_robot_bringup gazebo.launch.py

# For Isaac Sim
isaac-sim --/isaac/omniverse/user=YOUR_USERNAME --/isaac/omniverse/pass=YOUR_PASSWORD
```

### 3. Run Research Paper Pipeline
```bash
# Build the paper
cd src/paper
make pdf  # or your build command

# Run validation checks
python -m scripts.validation.citation_check
python -m scripts.validation.plagiarism_check
python -m scripts.validation.readability_check
```

### 4. Build Documentation Site
```bash
cd docs
npm install
npm run build
npm run serve  # for local preview
```

## Testing

### Unit Tests
```bash
# Python unit tests
python -m pytest tests/unit/

# ROS 2 tests
colcon test --packages-select your_package_name
```

### Integration Tests
```bash
# Simulation smoke tests
python -m tests.integration.simulation_smoke_test

# VLA end-to-end demo
python -m tests.integration.vla_demo_test
```

## Reproducibility Package

### Create Package
```bash
python -m scripts.reproducibility.create_package
```

### Validate Package
```bash
python -m scripts.reproducibility.validate
```

## Common Commands

### Check Quality Gates
```bash
# Citation compliance
python -m scripts.quality.citation_compliance

# Plagiarism check
python -m scripts.quality.plagiarism_check

# Readability check (Flesch-Kincaid 10-12)
python -m scripts.quality.readability_check

# Peer-reviewed percentage
python -m scripts.quality.peer_reviewed_percentage
```

### Build Deliverables
```bash
# Build PDF paper
make paper-pdf

# Build Docusaurus site
make docs-site

# Create reproducibility package
make reproducibility-package

# Build all
make all
```

## Troubleshooting

### ROS 2 Issues
- Ensure proper environment sourcing: `source /opt/ros/humble/setup.bash`
- Check ROS_DISTRO: `echo $ROS_DISTRO` (should be humble)

### Isaac Sim Issues
- Verify NVIDIA GPU drivers are up to date
- Check CUDA version compatibility
- Ensure Isaac Sim license is active

### Simulation Performance
- Monitor GPU utilization: `nvidia-smi`
- Adjust simulation quality settings for performance
- Close unnecessary applications during intensive simulations