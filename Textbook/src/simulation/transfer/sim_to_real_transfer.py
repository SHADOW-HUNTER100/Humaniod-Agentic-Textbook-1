"""
Sim-to-Real Transfer System for AI-Native Software Development & Physical AI project
Implements capabilities for transferring models and behaviors from simulation to real-world humanoid robots
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy


@dataclass
class TransferMetrics:
    """Data class for tracking sim-to-real transfer metrics"""
    sim_performance: float
    real_performance: float
    transfer_loss: float
    success_rate: float
    adaptation_steps: int
    domain_gap: float


class DomainRandomization:
    """
    Implements domain randomization techniques to improve sim-to-real transfer
    Randomizes simulation parameters to make models more robust to real-world variations
    """

    def __init__(self):
        """Initialize domain randomization parameters"""
        self.randomization_ranges = {
            "visual": {
                "lighting": (0.5, 1.5),  # Lighting intensity range
                "texture_noise": (0.0, 0.1),  # Texture variation
                "color_shift": (0.0, 0.05),  # Color variation
                "blur": (0.0, 2.0)  # Blur amount
            },
            "dynamics": {
                "friction": (0.1, 0.9),  # Friction coefficient range
                "mass": (0.8, 1.2),  # Mass scaling factor
                "damping": (0.5, 1.5),  # Damping coefficient range
                "gravity": (9.5, 10.5),  # Gravity range
                "actuator_noise": (0.0, 0.02)  # Actuator noise level
            },
            "sensor": {
                "noise": (0.0, 0.01),  # Sensor noise level
                "delay": (0.0, 0.02),  # Sensor delay in seconds
                "bias": (-0.001, 0.001)  # Sensor bias range
            }
        }

    def randomize_visual_properties(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomize visual properties for domain randomization

        Args:
            base_params: Base visual parameters

        Returns:
            Randomized visual parameters
        """
        randomized = copy.deepcopy(base_params)

        # Randomize lighting
        lighting_factor = np.random.uniform(*self.randomization_ranges["visual"]["lighting"])
        randomized["lighting_intensity"] = base_params.get("lighting_intensity", 1.0) * lighting_factor

        # Randomize texture noise
        texture_noise = np.random.uniform(*self.randomization_ranges["visual"]["texture_noise"])
        randomized["texture_noise"] = texture_noise

        # Randomize color shift
        color_shift = np.random.uniform(*self.randomization_ranges["visual"]["color_shift"])
        if "color" in randomized:
            # Add small random shift to each color channel
            base_color = np.array(randomized["color"])
            shifted_color = np.clip(base_color + color_shift * np.random.choice([-1, 1], size=base_color.shape), 0, 1)
            randomized["color"] = shifted_color.tolist()

        # Randomize blur
        blur_amount = np.random.uniform(*self.randomization_ranges["visual"]["blur"])
        randomized["blur"] = blur_amount

        return randomized

    def randomize_dynamics_properties(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomize dynamics properties for domain randomization

        Args:
            base_params: Base dynamics parameters

        Returns:
            Randomized dynamics parameters
        """
        randomized = copy.deepcopy(base_params)

        # Randomize friction
        friction_factor = np.random.uniform(*self.randomization_ranges["dynamics"]["friction"])
        randomized["friction"] = base_params.get("friction", 0.5) * friction_factor

        # Randomize mass
        mass_factor = np.random.uniform(*self.randomization_ranges["dynamics"]["mass"])
        randomized["mass"] = base_params.get("mass", 1.0) * mass_factor

        # Randomize damping
        damping_factor = np.random.uniform(*self.randomization_ranges["dynamics"]["damping"])
        randomized["damping"] = base_params.get("damping", 0.1) * damping_factor

        # Randomize gravity
        gravity_value = np.random.uniform(*self.randomization_ranges["dynamics"]["gravity"])
        randomized["gravity"] = gravity_value

        # Randomize actuator noise
        actuator_noise = np.random.uniform(*self.randomization_ranges["dynamics"]["actuator_noise"])
        randomized["actuator_noise"] = actuator_noise

        return randomized

    def randomize_sensor_properties(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Randomize sensor properties for domain randomization

        Args:
            base_params: Base sensor parameters

        Returns:
            Randomized sensor parameters
        """
        randomized = copy.deepcopy(base_params)

        # Randomize sensor noise
        noise_level = np.random.uniform(*self.randomization_ranges["sensor"]["noise"])
        randomized["noise"] = noise_level

        # Randomize sensor delay
        delay = np.random.uniform(*self.randomization_ranges["sensor"]["delay"])
        randomized["delay"] = delay

        # Randomize sensor bias
        bias = np.random.uniform(*self.randomization_ranges["sensor"]["bias"])
        randomized["bias"] = bias

        return randomized


class SystemIdentification:
    """
    System identification techniques to model real-world robot dynamics
    Helps bridge the gap between simulation and reality
    """

    def __init__(self):
        """Initialize system identification parameters"""
        self.model_params = {}
        self.identification_data = []

    def collect_identification_data(self, robot, trajectories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Collect data for system identification

        Args:
            robot: Robot object to collect data from
            trajectories: List of trajectory dictionaries with inputs and outputs

        Returns:
            Collected identification data
        """
        identification_data = []

        for traj in trajectories:
            # Simulate collecting data from robot execution
            input_signal = traj.get("input", [])
            output_signal = traj.get("output", [])
            time_stamps = traj.get("timestamps", [])

            # Process the data to extract system characteristics
            system_data = {
                "input": input_signal,
                "output": output_signal,
                "timestamps": time_stamps,
                "system_characteristics": self._extract_system_characteristics(input_signal, output_signal)
            }

            identification_data.append(system_data)

        self.identification_data.extend(identification_data)
        return identification_data

    def _extract_system_characteristics(self, input_signal: List[float], output_signal: List[float]) -> Dict[str, float]:
        """
        Extract system characteristics from input-output data

        Args:
            input_signal: System input signal
            output_signal: System output signal

        Returns:
            Dictionary of system characteristics
        """
        if len(input_signal) == 0 or len(output_signal) == 0:
            return {}

        # Calculate basic system characteristics
        gain = np.mean(output_signal) / (np.mean(input_signal) + 1e-8)  # Add small value to avoid division by zero

        # Calculate response time (time to reach 90% of final value)
        final_value = output_signal[-1] if output_signal else 0
        ninety_percent = 0.9 * final_value

        response_time_idx = next((i for i, val in enumerate(output_signal) if abs(val) >= abs(ninety_percent)), len(output_signal))
        response_time = response_time_idx / len(output_signal) if len(output_signal) > 0 else 0

        # Calculate stability (variance of response)
        stability = np.var(output_signal)

        return {
            "gain": float(gain),
            "response_time": float(response_time),
            "stability": float(stability),
            "delay": self._estimate_delay(input_signal, output_signal)
        }

    def _estimate_delay(self, input_signal: List[float], output_signal: List[float]) -> float:
        """
        Estimate system delay using cross-correlation

        Args:
            input_signal: System input signal
            output_signal: System output signal

        Returns:
            Estimated delay in samples
        """
        if len(input_signal) < 2 or len(output_signal) < 2:
            return 0.0

        # Normalize signals
        input_norm = (input_signal - np.mean(input_signal)) / (np.std(input_signal) + 1e-8)
        output_norm = (output_signal - np.mean(output_signal)) / (np.std(output_signal) + 1e-8)

        # Compute cross-correlation
        correlation = np.correlate(input_norm, output_norm, mode='full')

        # Find the lag with maximum correlation
        max_corr_idx = np.argmax(correlation)
        delay_samples = max_corr_idx - len(output_signal) + 1

        return float(delay_samples)

    def identify_model_parameters(self, system_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Identify model parameters from collected system data

        Args:
            system_data: List of system characteristic data

        Returns:
            Identified model parameters
        """
        if not system_data:
            return {}

        # Aggregate system characteristics
        gains = [data["system_characteristics"].get("gain", 0) for data in system_data]
        response_times = [data["system_characteristics"].get("response_time", 0) for data in system_data]
        stabilities = [data["system_characteristics"].get("stability", 0) for data in system_data]
        delays = [data["system_characteristics"].get("delay", 0) for data in system_data]

        # Compute average parameters
        avg_gain = np.mean(gains) if gains else 1.0
        avg_response_time = np.mean(response_times) if response_times else 0.1
        avg_stability = np.mean(stabilities) if stabilities else 0.01
        avg_delay = np.mean(delays) if delays else 0.0

        self.model_params = {
            "gain": float(avg_gain),
            "response_time": float(avg_response_time),
            "stability": float(avg_stability),
            "delay": float(avg_delay)
        }

        return self.model_params


class AdaptationMechanism:
    """
    Adaptive mechanisms to adjust behaviors during sim-to-real transfer
    Helps the system adapt to real-world conditions that differ from simulation
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize adaptation mechanism

        Args:
            learning_rate: Rate at which the system adapts
        """
        self.learning_rate = learning_rate
        self.adaptation_history = []
        self.performance_buffer = []  # Track recent performance for adaptation decisions

    def adapt_behavior(self, simulation_output: Any, real_world_feedback: Dict[str, float]) -> Any:
        """
        Adapt behavior based on real-world feedback

        Args:
            simulation_output: Output from simulation model
            real_world_feedback: Feedback from real-world execution

        Returns:
            Adapted output
        """
        # Extract relevant feedback metrics
        performance_error = real_world_feedback.get("performance_error", 0.0)
        tracking_error = real_world_feedback.get("tracking_error", 0.0)
        stability_metric = real_world_feedback.get("stability", 1.0)

        # Calculate adaptation factor based on errors
        adaptation_factor = self.learning_rate * (abs(performance_error) + abs(tracking_error)) * stability_metric

        # Store adaptation data for history
        adaptation_record = {
            "timestamp": len(self.adaptation_history),
            "performance_error": performance_error,
            "tracking_error": tracking_error,
            "stability": stability_metric,
            "adaptation_factor": adaptation_factor
        }
        self.adaptation_history.append(adaptation_record)

        # Apply adaptation to simulation output
        # This is a simplified example - real implementation would be more complex
        adapted_output = self._apply_adaptation(simulation_output, adaptation_factor)

        return adapted_output

    def _apply_adaptation(self, output: Any, adaptation_factor: float) -> Any:
        """
        Apply adaptation to the output

        Args:
            output: Original output from simulation
            adaptation_factor: Factor determining how much to adapt

        Returns:
            Adapted output
        """
        # For this example, we'll assume the output is numeric and apply a simple adaptation
        # In a real implementation, this would be specific to the type of output

        if isinstance(output, (int, float)):
            # Apply gain adjustment based on adaptation factor
            adapted_value = output * (1.0 + adaptation_factor)
            return adapted_value

        elif isinstance(output, (list, tuple, np.ndarray)):
            # Apply adaptation to array-like output
            output_array = np.array(output)
            adapted_array = output_array * (1.0 + adaptation_factor)
            return adapted_array.tolist() if isinstance(output, (list, tuple)) else adapted_array

        elif isinstance(output, dict):
            # Apply adaptation to dictionary values
            adapted_dict = {}
            for key, value in output.items():
                if isinstance(value, (int, float, list, tuple, np.ndarray)):
                    adapted_dict[key] = self._apply_adaptation(value, adaptation_factor)
                else:
                    adapted_dict[key] = value  # Keep non-numeric values unchanged
            return adapted_dict

        else:
            # For other types, return original (no adaptation applied)
            return output

    def update_performance_buffer(self, performance_metric: float, window_size: int = 10):
        """
        Update the performance buffer with a new metric

        Args:
            performance_metric: New performance metric value
            window_size: Size of the sliding window
        """
        self.performance_buffer.append(performance_metric)

        # Keep only the most recent values
        if len(self.performance_buffer) > window_size:
            self.performance_buffer = self.performance_buffer[-window_size:]

    def should_adapt(self, threshold: float = 0.1) -> bool:
        """
        Determine if adaptation should be triggered based on performance

        Args:
            threshold: Threshold for triggering adaptation

        Returns:
            True if adaptation should occur
        """
        if len(self.performance_buffer) < 2:
            return False

        # Check if recent performance has degraded
        recent_performance = self.performance_buffer[-2:]  # Last 2 measurements
        performance_change = abs(recent_performance[-1] - recent_performance[0])

        return performance_change > threshold


class SimToRealTransfer:
    """
    Main class for sim-to-real transfer in humanoid robotics
    Integrates domain randomization, system identification, and adaptation mechanisms
    """

    def __init__(self):
        """Initialize the sim-to-real transfer system"""
        self.domain_randomizer = DomainRandomization()
        self.system_identifier = SystemIdentification()
        self.adaptation_mechanism = AdaptationMechanism()
        self.transfer_metrics = []

    def initialize_simulation_model(self, robot_type: str, domain_randomization: bool = True) -> Dict[str, Any]:
        """
        Initialize simulation model with appropriate parameters for sim-to-real transfer

        Args:
            robot_type: Type of robot (e.g., "humanoid", "quadruped", "wheeled")
            domain_randomization: Whether to apply domain randomization

        Returns:
            Initialized simulation parameters
        """
        # Base parameters for different robot types
        base_params = {
            "humanoid": {
                "mass": 75.0,  # kg
                "height": 1.7,  # meters
                "friction": 0.5,
                "damping": 0.1,
                "gravity": 9.81,
                "actuator_limits": {"position": 1.5, "velocity": 5.0, "effort": 100.0}
            },
            "quadruped": {
                "mass": 25.0,  # kg
                "height": 0.6,  # meters
                "friction": 0.6,
                "damping": 0.2,
                "gravity": 9.81,
                "actuator_limits": {"position": 2.0, "velocity": 10.0, "effort": 50.0}
            },
            "wheeled": {
                "mass": 10.0,  # kg
                "friction": 0.7,
                "damping": 0.3,
                "gravity": 9.81,
                "actuator_limits": {"velocity": 2.0, "effort": 20.0}
            }
        }

        if robot_type not in base_params:
            robot_type = "humanoid"  # Default to humanoid

        params = base_params[robot_type]

        # Apply domain randomization if enabled
        if domain_randomization:
            params = self.domain_randomizer.randomize_dynamics_properties(params)

        return params

    def calibrate_for_real_robot(self, real_robot_data: Dict[str, Any]) -> bool:
        """
        Calibrate the simulation to match real robot characteristics

        Args:
            real_robot_data: Data collected from real robot operation

        Returns:
            True if calibration successful
        """
        try:
            # Use system identification to adjust simulation parameters
            model_params = self.system_identifier.identify_model_parameters([real_robot_data])

            # Update simulation parameters based on identified model
            # This would involve adjusting the physics engine parameters in a real implementation
            print(f"Calibrated simulation with model parameters: {model_params}")

            return True
        except Exception as e:
            print(f"Calibration failed: {e}")
            return False

    def transfer_policy(self, simulation_policy: Any, real_world_conditions: Dict[str, float]) -> Any:
        """
        Transfer a policy from simulation to real world with adaptation

        Args:
            simulation_policy: Policy learned in simulation
            real_world_conditions: Current real-world conditions

        Returns:
            Adapted policy for real-world execution
        """
        # Apply adaptation based on real-world conditions
        adapted_policy = self.adaptation_mechanism.adapt_behavior(simulation_policy, real_world_conditions)

        return adapted_policy

    def evaluate_transfer_performance(self, sim_performance: float, real_performance: float) -> TransferMetrics:
        """
        Evaluate the performance of sim-to-real transfer

        Args:
            sim_performance: Performance in simulation
            real_performance: Performance in real world

        Returns:
            Transfer metrics including loss and success rate
        """
        # Calculate transfer metrics
        transfer_loss = max(0, sim_performance - real_performance)  # Loss due to transfer
        success_rate = real_performance / sim_performance if sim_performance > 0 else 0.0
        domain_gap = abs(sim_performance - real_performance)  # Measure of domain gap

        # Calculate adaptation steps (placeholder - in real implementation would track adaptation iterations)
        adaptation_steps = len(self.adaptation_mechanism.adaptation_history)

        metrics = TransferMetrics(
            sim_performance=sim_performance,
            real_performance=real_performance,
            transfer_loss=transfer_loss,
            success_rate=success_rate,
            adaptation_steps=adaptation_steps,
            domain_gap=domain_gap
        )

        # Store metrics for analysis
        self.transfer_metrics.append(metrics)

        return metrics

    def run_adaptation_cycle(self, initial_policy: Any, episodes: int = 10) -> List[TransferMetrics]:
        """
        Run an adaptation cycle to improve sim-to-real transfer

        Args:
            initial_policy: Starting policy for adaptation
            episodes: Number of adaptation episodes

        Returns:
            List of transfer metrics from each episode
        """
        metrics_history = []

        current_policy = initial_policy

        for episode in range(episodes):
            # Simulate performance in simulation
            sim_perf = self._simulate_performance(current_policy, episode)

            # Simulate performance in "real world" (with domain gap)
            real_perf = self._real_world_performance(current_policy, episode, sim_perf)

            # Evaluate transfer
            metrics = self.evaluate_transfer_performance(sim_perf, real_perf)
            metrics_history.append(metrics)

            # Provide feedback for adaptation
            feedback = {
                "performance_error": metrics.transfer_loss,
                "tracking_error": 0.05 * episode,  # Simulate increasing tracking error
                "stability": metrics.success_rate
            }

            # Update performance buffer
            self.adaptation_mechanism.update_performance_buffer(metrics.success_rate)

            # Adapt if needed
            if self.adaptation_mechanism.should_adapt():
                current_policy = self.transfer_policy(current_policy, feedback)

            print(f"Episode {episode + 1}: Sim={sim_perf:.3f}, Real={real_perf:.3f}, Success={metrics.success_rate:.3f}")

        return metrics_history

    def _simulate_performance(self, policy: Any, episode: int) -> float:
        """
        Simulate performance in simulation environment (placeholder)

        Args:
            policy: Policy to evaluate
            episode: Episode number for variation

        Returns:
            Simulated performance metric
        """
        # Placeholder implementation - in reality would run simulation
        base_performance = 0.95
        variation = 0.05 * np.sin(episode * 0.5)  # Add some variation
        return max(0.0, min(1.0, base_performance + variation))

    def _real_world_performance(self, policy: Any, episode: int, sim_perf: float) -> float:
        """
        Simulate performance in real-world environment (with domain gap)

        Args:
            policy: Policy to evaluate
            episode: Episode number for variation
            sim_perf: Performance in simulation

        Returns:
            Real-world performance metric
        """
        # Simulate domain gap with some randomness
        domain_gap_factor = 0.8 + 0.1 * np.random.random()  # 80-90% of sim performance
        real_performance = sim_perf * domain_gap_factor

        # Add some noise to simulate real-world variability
        noise = 0.02 * np.random.random()
        real_performance = max(0.0, min(1.0, real_performance + noise))

        return real_performance


# Example usage and testing
def test_sim_to_real_transfer():
    """Test the sim-to-real transfer system"""
    transfer_system = SimToRealTransfer()

    print("Testing Sim-to-Real Transfer System...")
    print()

    # Initialize simulation model for humanoid robot
    print("Initializing simulation model for humanoid robot...")
    sim_params = transfer_system.initialize_simulation_model("humanoid", domain_randomization=True)
    print(f"Simulation parameters: {sim_params}")
    print()

    # Simulate collecting real robot data for calibration
    print("Simulating real robot data collection for calibration...")
    sample_trajectory = {
        "input": [0.1, 0.2, 0.3, 0.4, 0.5],
        "output": [0.05, 0.15, 0.25, 0.35, 0.45],
        "timestamps": [0.0, 0.1, 0.2, 0.3, 0.4]
    }

    identification_data = transfer_system.system_identifier.collect_identification_data(None, [sample_trajectory])
    print(f"Collected identification data: {len(identification_data)} samples")
    print()

    # Calibrate simulation
    print("Calibrating simulation to real robot characteristics...")
    calibration_success = transfer_system.calibrate_for_real_robot(identification_data[0])
    print(f"Calibration success: {calibration_success}")
    print()

    # Run adaptation cycle
    print("Running adaptation cycle...")
    initial_policy = {"action_mapping": [0.5, 0.6, 0.7], "control_gains": [1.0, 1.2]}
    metrics_history = transfer_system.run_adaptation_cycle(initial_policy, episodes=5)
    print()

    # Evaluate final transfer performance
    print("Transfer Performance Metrics:")
    for i, metrics in enumerate(metrics_history):
        print(f"Episode {i+1}:")
        print(f"  Sim Performance: {metrics.sim_performance:.3f}")
        print(f"  Real Performance: {metrics.real_performance:.3f}")
        print(f"  Transfer Loss: {metrics.transfer_loss:.3f}")
        print(f"  Success Rate: {metrics.success_rate:.3f}")
        print(f"  Domain Gap: {metrics.domain_gap:.3f}")
        print(f"  Adaptation Steps: {metrics.adaptation_steps}")
        print()

    # Calculate average metrics
    avg_sim_perf = np.mean([m.sim_performance for m in metrics_history])
    avg_real_perf = np.mean([m.real_performance for m in metrics_history])
    avg_success_rate = np.mean([m.success_rate for m in metrics_history])

    print(f"Average Performance:")
    print(f"  Sim: {avg_sim_perf:.3f}")
    print(f"  Real: {avg_real_perf:.3f}")
    print(f"  Success Rate: {avg_success_rate:.3f}")
    print(f"  Transfer Loss: {(avg_sim_perf - avg_real_perf):.3f}")


if __name__ == "__main__":
    test_sim_to_real_transfer()