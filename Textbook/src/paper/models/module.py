"""
Module Model for AI-Native Software Development & Physical AI project
Defines the structure and validation for Physical AI modules
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum


class ModuleType(Enum):
    """Enumeration of possible module types for Physical AI research"""
    PHYSICAL_AI_OVERVIEW = "physical_ai_overview"
    ROS2_ROBOSYSTEM_NERVOUS_SYSTEM = "ros2_robosystem_nervous_system"
    DIGITAL_TWIN_SIMULATION = "digital_twin_simulation"
    NVIDIA_ISAAC_AI_BRAIN = "nvidia_isaac_ai_brain"
    VLA_VISION_LANGUAGE_ACTION = "vla_vision_language_action"
    HUMANOID_ROBOTICS_DEVELOPMENT = "humanoid_robotics_development"
    WEEKLY_BREAKDOWN = "weekly_breakdown"
    ASSESSMENTS = "assessments"
    HARDWARE_REQUIREMENTS = "hardware_requirements"
    LAB_ARCHITECTURE_OPTIONS = "lab_architecture_options"


class Module:
    """
    Represents a specific module within the research paper covering Physical AI topics
    Validates that each module meets academic standards and maintains consistency with the overall paper
    """

    def __init__(self,
                 title: str,
                 content: str = "",
                 module_type: ModuleType = ModuleType.PHYSICAL_AI_OVERVIEW,
                 parent_paper_id: Optional[str] = None):
        """
        Initialize a Module with validation

        Args:
            title: The title of the module
            content: The text content of the module
            module_type: Type of Physical AI module (overview, ROS2, simulation, etc.)
            parent_paper_id: ID of the parent research paper this module belongs to
        """
        self.id = str(uuid.uuid4())
        self.title = title
        self.content = content
        self.module_type = module_type
        self.parent_paper_id = parent_paper_id
        self.created_date = datetime.now()
        self.last_modified = datetime.now()

        # Validate initial values
        self._validate_module_type()
        self._validate_content_length()

    def _validate_module_type(self):
        """Validate that the module type is one of the allowed types"""
        allowed_types = [item.value for item in ModuleType]
        if self.module_type.value not in allowed_types:
            raise ValueError(f"Module type '{self.module_type.value}' is not allowed. Allowed types: {allowed_types}")

    def _validate_content_length(self):
        """Validate that content is not empty and meets minimum requirements"""
        if not self.content or len(self.content.strip()) == 0:
            raise ValueError("Module content cannot be empty")

        # For Physical AI modules, we might have specific content requirements
        word_count = len(self.content.split())
        if word_count < 500:  # Minimum content for a substantial module
            raise ValueError(f"Module content too short ({word_count} words). Minimum 500 words required.")

    def update_content(self, content: str):
        """Update module content and validate requirements"""
        self.content = content
        self._validate_content_length()
        self.last_modified = datetime.now()

    def set_parent_paper(self, parent_paper_id: str):
        """Set the parent paper for this module"""
        self.parent_paper_id = parent_paper_id

    def validate_academic_standards(self) -> List[str]:
        """
        Validate that this module meets academic standards
        Returns list of issues found
        """
        issues = []

        # Check for proper academic tone and terminology
        content_lower = self.content.lower()

        # Check for basic academic elements
        if not any(word in content_lower for word in ['figure', 'table', 'equation', 'reference', 'cite']):
            issues.append("Module should include academic elements like figures, tables, equations, or references")

        # Check for proper terminology consistency
        if 'physical ai' not in content_lower and 'physical artificial intelligence' not in content_lower:
            issues.append("Module should include appropriate Physical AI terminology")

        # Check for proper structure
        if not any(header in self.content for header in ['#', '##', '###']):
            issues.append("Module should include proper section headers")

        return issues

    def validate_module_specific_requirements(self) -> List[str]:
        """
        Validate module-specific requirements based on module type
        Returns list of issues found
        """
        issues = []

        content_lower = self.content.lower()

        # Validate requirements based on module type
        if self.module_type == ModuleType.PHYSICAL_AI_OVERVIEW:
            required_terms = ['embodied intelligence', 'physical laws', 'real-world']
            missing_terms = [term for term in required_terms if term not in content_lower]
            if missing_terms:
                issues.append(f"Physical AI Overview module missing key concepts: {missing_terms}")

        elif self.module_type == ModuleType.ROS2_ROBOSYSTEM_NERVOUS_SYSTEM:
            required_terms = ['ros2', 'nodes', 'topics', 'services', 'actions']
            missing_terms = [term for term in required_terms if term not in content_lower]
            if missing_terms:
                issues.append(f"ROS2 module missing key ROS2 concepts: {missing_terms}")

        elif self.module_type == ModuleType.DIGITAL_TWIN_SIMULATION:
            required_terms = ['simulation', 'physics', 'gazebo', 'isaac']
            missing_terms = [term for term in required_terms if term not in content_lower]
            if missing_terms:
                issues.append(f"Digital Twin module missing key simulation concepts: {missing_terms}")

        elif self.module_type == ModuleType.NVIDIA_ISAAC_AI_BRAIN:
            required_terms = ['isaac', 'perception', 'navigation', 'training']
            missing_terms = [term for term in required_terms if term not in content_lower]
            if missing_terms:
                issues.append(f"NVIDIA Isaac module missing key Isaac concepts: {missing_terms}")

        elif self.module_type == ModuleType.VLA_VISION_LANGUAGE_ACTION:
            required_terms = ['vision', 'language', 'action', 'multimodal']
            missing_terms = [term for term in required_terms if term not in content_lower]
            if missing_terms:
                issues.append(f"VLA module missing key VLA concepts: {missing_terms}")

        elif self.module_type == ModuleType.HUMANOID_ROBOTICS_DEVELOPMENT:
            required_terms = ['kinematics', 'dynamics', 'balance', 'manipulation']
            missing_terms = [term for term in required_terms if term not in content_lower]
            if missing_terms:
                issues.append(f"Humanoid Robotics module missing key robotics concepts: {missing_terms}")

        return issues

    def get_citation_density(self) -> float:
        """
        Calculate citation density (citations per 500 words) as required by academic standards
        This is a simplified implementation - in practice would parse citations
        """
        word_count = len(self.content.split())
        # This is a placeholder - in a real implementation would count actual citations
        estimated_citations = content_lower.count('et al.') + content_lower.count('cite') + content_lower.count('reference')
        citations_per_500_words = (estimated_citations / word_count) * 500 if word_count > 0 else 0

        return citations_per_500_words

    def to_dict(self) -> Dict:
        """Convert the module to a dictionary representation"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "module_type": self.module_type.value,
            "parent_paper_id": self.parent_paper_id,
            "created_date": self.created_date.isoformat(),
            "last_modified": self.last_modified.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Module':
        """Create a Module instance from a dictionary"""
        return cls(
            title=data["title"],
            content=data.get("content", ""),
            module_type=ModuleType(data["module_type"]),
            parent_paper_id=data.get("parent_paper_id")
        )

    def __repr__(self) -> str:
        """String representation of the module"""
        return f"Module(id={self.id}, title='{self.title}', type={self.module_type.value})"


class ModuleCollection:
    """
    Collection class to manage multiple modules and validate their consistency
    """

    def __init__(self):
        """Initialize the module collection"""
        self.modules: List[Module] = []

    def add_module(self, module: Module):
        """Add a module to the collection"""
        self.modules.append(module)

    def validate_collection_consistency(self) -> List[str]:
        """
        Validate that all modules in the collection maintain consistency
        Returns list of consistency issues found
        """
        issues = []

        # Check for duplicate titles
        titles = [module.title for module in self.modules]
        if len(titles) != len(set(titles)):
            issues.append("Duplicate module titles found in collection")

        # Check for required module types (depending on research requirements)
        required_modules = {
            ModuleType.PHYSICAL_AI_OVERVIEW,
            ModuleType.ROS2_ROBOSYSTEM_NERVOUS_SYSTEM,
            ModuleType.DIGITAL_TWIN_SIMULATION
        }

        present_modules = {module.module_type for module in self.modules}
        missing_required = required_modules - present_modules

        if missing_required:
            issues.append(f"Missing required modules: {[m.value for m in missing_required]}")

        # Check for academic consistency across modules
        for i, module in enumerate(self.modules):
            issues.extend([
                f"Module {i+1} ({module.title}): {issue}"
                for issue in module.validate_academic_standards()
            ])

        return issues

    def get_module_by_type(self, module_type: ModuleType) -> Optional[Module]:
        """Get a module by its type"""
        for module in self.modules:
            if module.module_type == module_type:
                return module
        return None

    def to_dict(self) -> List[Dict]:
        """Convert the collection to a list of dictionaries"""
        return [module.to_dict() for module in self.modules]