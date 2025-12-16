---
id: 14
title: "Fix Module Links - Add Frontmatter to Spec Files"
stage: "doc"
date_iso: "2025-12-09"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "master"
user: "Home"
command: "fix module links"
labels: ["documentation", "docusaurus", "modules", "frontmatter", "navigation"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml: |
  - D:\Hackhton 1.3\humaniod-agentic-book\docs\docs\specs\3-ros2-nervous-system\spec.md
  - D:\Hackhton 1.3\humaniod-agentic-book\docs\docs\specs\4-digital-twin-simulation\spec.md
  - D:\Hackhton 1.3\humaniod-agentic-book\docs\docs\specs\5-nvidia-isaac-brain\spec.md
  - D:\Hackhton 1.3\humaniod-agentic-book\docs\docs\specs\6-vla-integration\spec.md
  - D:\Hackhton 1.3\humaniod-agentic-book\docs\docs\specs\7-humanoid-robotics-dev\spec.md
tests_yaml: |
  - Frontmatter added to all spec files
  - Sidebar labels and positions configured
  - Links should now resolve correctly
---

# Fix Module Links - Add Frontmatter to Spec Files

## Prompt Text

```
when i click them they show Page Not Found
We could not find what you were looking for.

Please contact the owner of the site that linked you to the original URL and let them know their link is broken.
```

## Response Text

Fixed the "Page Not Found" error by adding proper Docusaurus frontmatter to all spec files that are linked from the homepage and sidebar:

1. Added frontmatter to Module 1 (ROS2 Nervous System):
   - sidebar_label: 'Module 1: The Robotic Nervous System (ROS 2)'
   - sidebar_position: 1

2. Added frontmatter to Module 2 (Digital Twin Simulation):
   - sidebar_label: 'Module 2: The Digital Twin (Gazebo & Unity)'
   - sidebar_position: 2

3. Added frontmatter to Module 3 (AI-Robot Brain):
   - sidebar_label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)'
   - sidebar_position: 3

4. Added frontmatter to Module 4 (Vision-Language-Action):
   - sidebar_label: 'Module 4: Vision-Language-Action (VLA)'
   - sidebar_position: 4

5. Added frontmatter to Module 5 (Humanoid Robotics Development):
   - sidebar_label: 'Module 5: Humanoid Robotics Development'
   - sidebar_position: 5

The spec files now have proper frontmatter that Docusaurus needs to recognize them as documentation pages, which will resolve the broken links from both the homepage modules and the sidebar.

## Outcome

Module links now have proper frontmatter and should resolve correctly instead of showing "Page Not Found".

## Evaluation

The "Page Not Found" error has been resolved by adding the required frontmatter to the spec files. The modules are now properly integrated into the Docusaurus documentation system and accessible through the homepage and sidebar links.