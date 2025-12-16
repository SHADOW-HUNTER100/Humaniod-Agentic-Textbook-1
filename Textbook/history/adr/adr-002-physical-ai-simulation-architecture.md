# ADR-002: Physical AI Simulation and Robotics Architecture

> **Scope**: Document decision clusters, not individual technology choices. Group related decisions that work together (e.g., "Frontend Stack" not separate ADRs for framework, styling, deployment).

- **Status:** Accepted
- **Date:** 2025-12-08
- **Feature:** AI-Native Software Development & Physical AI
- **Context:** Need to establish architecture for simulating Physical AI systems with realistic physics, integrating with robotics frameworks, and enabling sim-to-real transfer for humanoid robotics research.

<!-- Significance checklist (ALL must be true to justify this ADR)
     1) Impact: Long-term consequence for architecture/platform/security?
     2) Alternatives: Multiple viable options considered with tradeoffs?
     3) Scope: Cross-cutting concern (not an isolated detail)?
     If any are false, prefer capturing as a PHR note instead of an ADR. -->

## Decision

- Simulation Platform: Hybrid approach using both Isaac Sim 2023.1.0 and Gazebo for complementary capabilities
- Isaac Sim for: Photorealistic simulation, advanced AI perception, GPU-accelerated processing
- Gazebo for: Realistic physics simulation, URDF integration, ROS 2 native compatibility
- Robot Platform: Unitree Go2 as primary humanoid platform with support for G1 and OP3 alternatives
- Integration Layer: ROS 2 Humble as middleware connecting simulation, perception, and control systems
- VLA (Vision-Language-Action) Architecture: Integrated pipeline for human-robot interaction

## Consequences

### Positive

- Complementary simulation capabilities leveraging strengths of both platforms
- Realistic physics modeling combined with photorealistic rendering
- Standardized robot representation through URDF
- Extensive ROS 2 ecosystem for robotics development
- Pathway for sim-to-real transfer learning with proven hardware platform

### Negative

- Complex dual-simulation architecture requiring expertise in multiple platforms
- Potential synchronization challenges between Isaac Sim and Gazebo
- Hardware dependency on specific robot platform limiting experimentation
- Increased computational requirements for dual-simulation approach

## Alternatives Considered

Alternative Architecture A: Single simulation platform (Isaac Sim only)
- Why rejected: Missing advanced physics modeling capabilities of Gazebo

Alternative Architecture B: Different robot platform (ROBOTIS OP3 or custom platform)
- Why rejected: Unitree Go2 provides better balance of affordability, capabilities, and ROS 2 support

Alternative Architecture C: Pure cloud-based simulation
- Why rejected: Network latency concerns for real-time robotics applications

## References

- Feature Spec: specs/1-ai-native-physical-ai/spec.md
- Implementation Plan: specs/1-ai-native-physical-ai/plan.md
- Related ADRs: ADR-001
- Evaluator Evidence: specs/1-ai-native-physical-ai/research.md