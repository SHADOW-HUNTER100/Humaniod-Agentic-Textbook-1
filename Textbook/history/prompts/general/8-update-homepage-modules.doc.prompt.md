---
id: 8
title: "Update Homepage with Module Links"
stage: "doc"
date_iso: "2025-12-09"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "master"
user: "Home"
command: "update homepage modules"
labels: ["documentation", "docusaurus", "homepage", "modules", "ui"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml: |
  - D:\Hackhton 1.3\humaniod-agentic-book\docs\src\components\HomepageFeatures\index.tsx
tests_yaml: |
  - Module descriptions updated
  - Layout maintained with 3-column format
  - Feature content replaced with module information
---

# Update Homepage with Module Links

## Prompt Text

```
remove this in home page Easy to Use
Docusaurus was designed from the ground up to be easily installed and used to get your website up and running quickly.

Focus on What Matters
Docusaurus lets you focus on your docs, and we'll do the chores. Go ahead and move your docs into the docs directory.

Powered by React
Extend or customize your website layout by reusing React. Docusaurus can be extended while reusing the same header and footer.and add shortcut link of modules in there places
```

## Response Text

Updated the homepage features section to replace the default Docusaurus features with the Physical AI curriculum modules:

1. Module 1: Robotic Nervous System (ROS 2) - Middleware for robot control with ROS 2 Nodes, Topics, Services, Python-to-ROS bridging, and URDF for humanoids

2. Module 2: Digital Twin (Gazebo & Unity) - Physics simulation, environment building, sensor simulation, and high-fidelity rendering

3. Module 3: AI-Robot Brain (NVIDIA Isaac™) - Advanced perception, NVIDIA Isaac Sim, VSLAM, navigation, and VLA integration

The layout maintains the 3-column format for proper display, with Vision-Language-Action (VLA) concepts included as part of Module 3 to maintain the proper layout structure.

## Outcome

The homepage now showcases the Physical AI curriculum modules instead of generic Docusaurus features.

## Evaluation

The homepage features section has been successfully updated to highlight the curriculum modules as requested. The layout remains consistent while providing relevant information about each module to visitors.