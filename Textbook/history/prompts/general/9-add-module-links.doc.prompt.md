---
id: 9
title: "Add Clickable Module Links to Homepage"
stage: "doc"
date_iso: "2025-12-09"
surface: "agent"
model: "claude-sonnet-4-5-20250929"
feature: "none"
branch: "master"
user: "Home"
command: "add module links to homepage"
labels: ["documentation", "docusaurus", "homepage", "modules", "navigation"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files_yaml: |
  - D:\Hackhton 1.3\humaniod-agentic-book\docs\src\components\HomepageFeatures\index.tsx
tests_yaml: |
  - Module links added to homepage features
  - Click functionality verified
  - Correct documentation paths used
---

# Add Clickable Module Links to Homepage

## Prompt Text

```
and add if i click the modules its open there Documentation
```

## Response Text

Updated the homepage features to make each module clickable and link to its documentation:

1. Module 1: Robotic Nervous System now links to /docs/specs/3-ros2-nervous-system/spec
2. Module 2: Digital Twin now links to /docs/specs/4-digital-twin-simulation/spec
3. Module 3: AI-Robot Brain now links to /docs/specs/5-nvidia-isaac-brain/spec

The implementation includes:
- Added 'to' property to each feature item with the correct documentation path
- Updated the Feature component to make the entire feature card clickable when a 'to' property is present
- Added Link import from @docusaurus/Link
- Maintained the visual design while adding clickable functionality

## Outcome

Each module on the homepage is now clickable and links to its respective documentation.

## Evaluation

The homepage modules are now fully interactive with direct links to their documentation. Users can click on any module to access its detailed specification, improving navigation and user experience.