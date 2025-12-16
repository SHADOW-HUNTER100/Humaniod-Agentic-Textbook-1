"""
Module Service for AI-Native Software Development & Physical AI project
Implements ModuleService with academic standard validation
"""

from typing import List, Dict, Optional
from ..models.module import Module, ModuleType, ModuleCollection
from ..models.citation import Citation


class ModuleService:
    """
    Service to manage Physical AI modules with academic standard validation
    Ensures each module meets academic requirements and maintains consistency with overall paper
    """

    def __init__(self):
        """Initialize the module service"""
        self.modules: Dict[str, Module] = {}

    def create_module(self,
                     title: str,
                     content: str,
                     module_type: ModuleType,
                     parent_paper_id: Optional[str] = None) -> Module:
        """
        Create a new module with academic validation

        Args:
            title: Title of the module
            content: Content of the module
            module_type: Type of Physical AI module
            parent_paper_id: ID of parent research paper

        Returns:
            Created Module instance
        """
        module = Module(title=title, content=content, module_type=module_type, parent_paper_id=parent_paper_id)

        # Validate academic standards before saving
        academic_issues = module.validate_academic_standards()
        if academic_issues:
            print(f"Warning: Academic issues found in module '{title}': {academic_issues}")

        # Validate module-specific requirements
        specific_issues = module.validate_module_specific_requirements()
        if specific_issues:
            print(f"Warning: Module-specific issues found in module '{title}': {specific_issues}")

        self.modules[module.id] = module
        return module

    def get_module(self, module_id: str) -> Optional[Module]:
        """Get a module by its ID"""
        return self.modules.get(module_id)

    def update_module(self, module_id: str, title: str = None, content: str = None, parent_paper_id: str = None) -> Optional[Module]:
        """
        Update an existing module with validation

        Args:
            module_id: ID of the module to update
            title: New title (optional)
            content: New content (optional)
            parent_paper_id: New parent paper ID (optional)

        Returns:
            Updated Module instance, or None if not found
        """
        if module_id not in self.modules:
            return None

        module = self.modules[module_id]

        if title is not None:
            module.title = title

        if content is not None:
            module.update_content(content)

        if parent_paper_id is not None:
            module.set_parent_paper(parent_paper_id)

        # Re-validate after update
        academic_issues = module.validate_academic_standards()
        if academic_issues:
            print(f"Warning: Academic issues found in updated module '{module.title}': {academic_issues}")

        specific_issues = module.validate_module_specific_requirements()
        if specific_issues:
            print(f"Warning: Module-specific issues found in updated module '{module.title}': {specific_issues}")

        return module

    def validate_module_academic_standards(self, module: Module) -> Dict[str, List[str]]:
        """
        Validate that a module meets academic standards

        Args:
            module: Module to validate

        Returns:
            Dictionary with validation results
        """
        results = {
            "academic_standards": [],
            "module_specific": [],
            "citation_requirements": [],
            "overall_compliance": True
        }

        # Check academic standards
        academic_issues = module.validate_academic_standards()
        results["academic_standards"] = academic_issues

        # Check module-specific requirements
        specific_issues = module.validate_module_specific_requirements()
        results["module_specific"] = specific_issues

        # Check citation requirements
        citation_density = module.get_citation_density()
        if citation_density < 1.0:  # Require at least 1 citation per 500 words
            results["citation_requirements"].append(
                f"Module has low citation density: {citation_density:.2f} citations per 500 words. "
                f"At least 1.0 required."
            )

        # Overall compliance
        results["overall_compliance"] = (
            len(academic_issues) == 0 and
            len(specific_issues) == 0 and
            len(results["citation_requirements"]) == 0
        )

        return results

    def validate_module_combination(self, modules: List[Module]) -> Dict[str, List[str]]:
        """
        Validate that a combination of modules maintains academic consistency

        Args:
            modules: List of modules to validate together

        Returns:
            Dictionary with validation results
        """
        results = {
            "duplicate_titles": [],
            "inconsistent_terminology": [],
            "citation_density_issues": [],
            "overall_consistency": True
        }

        # Check for duplicate titles
        titles = [module.title for module in modules]
        unique_titles = set(titles)
        if len(titles) != len(unique_titles):
            duplicates = [title for title in unique_titles if titles.count(title) > 1]
            results["duplicate_titles"] = duplicates

        # Check for consistent citation density across modules
        for module in modules:
            citation_density = module.get_citation_density()
            if citation_density < 1.0:
                results["citation_density_issues"].append(
                    f"Module '{module.title}' has low citation density: {citation_density:.2f}"
                )

        # Overall consistency
        results["overall_consistency"] = (
            len(results["duplicate_titles"]) == 0 and
            len(results["citation_density_issues"]) == 0
        )

        return results

    def get_modules_for_paper(self, paper_id: str) -> List[Module]:
        """Get all modules associated with a specific paper"""
        return [module for module in self.modules.values() if module.parent_paper_id == paper_id]

    def validate_complete_paper_modules(self, paper_id: str) -> Dict[str, List[str]]:
        """
        Validate all modules for a complete paper

        Args:
            paper_id: ID of the paper to validate

        Returns:
            Dictionary with validation results for all modules in the paper
        """
        modules = self.get_modules_for_paper(paper_id)

        results = {
            "module_validations": {},
            "collection_issues": [],
            "overall_compliance": True
        }

        # Validate each module individually
        for module in modules:
            validation_result = self.validate_module_academic_standards(module)
            results["module_validations"][module.id] = validation_result

        # Validate combination of modules
        combination_result = self.validate_module_combination(modules)
        results["collection_issues"] = combination_result

        # Check overall compliance
        all_module_compliant = all(
            val["overall_compliance"] for val in results["module_validations"].values()
        )
        results["overall_compliance"] = (
            all_module_compliant and
            combination_result["overall_consistency"]
        )

        return results

    def add_citation_to_module(self, module_id: str, citation: Citation) -> bool:
        """
        Add a citation to a module (in a real implementation, this would connect to a citation system)

        Args:
            module_id: ID of the module to add citation to
            citation: Citation to add

        Returns:
            True if successful, False otherwise
        """
        if module_id not in self.modules:
            return False

        # In a real implementation, this would add the citation to the module's citation list
        # For this implementation, we'll just validate that the module exists
        return True

    def get_module_summary(self, module_id: str) -> Optional[Dict]:
        """
        Get a summary of a module's academic compliance

        Args:
            module_id: ID of the module to summarize

        Returns:
            Summary dictionary or None if module doesn't exist
        """
        if module_id not in self.modules:
            return None

        module = self.modules[module_id]
        validation = self.validate_module_academic_standards(module)

        return {
            "id": module.id,
            "title": module.title,
            "type": module.module_type.value,
            "word_count": len(module.content.split()),
            "academic_compliance": validation["overall_compliance"],
            "academic_issues": len(validation["academic_standards"]),
            "specific_issues": len(validation["module_specific"]),
            "citation_density": module.get_citation_density(),
            "last_modified": module.last_modified.isoformat()
        }

    def get_paper_modules_summary(self, paper_id: str) -> List[Dict]:
        """
        Get summaries for all modules in a paper

        Args:
            paper_id: ID of the paper

        Returns:
            List of module summaries
        """
        modules = self.get_modules_for_paper(paper_id)
        summaries = []

        for module in modules:
            summary = self.get_module_summary(module.id)
            if summary:
                summaries.append(summary)

        return summaries


# Example usage and testing
def test_module_service():
    """Test the module service with sample modules"""
    service = ModuleService()

    # Create sample modules
    print("Creating sample modules...")

    # Physical AI Overview module
    physical_ai_module = service.create_module(
        title="Physical AI Overview",
        content="Physical artificial intelligence represents a paradigm shift in robotics research. "
                "Rather than treating AI as merely a controller for mechanical systems, "
                "Physical AI systems embody intelligence in physical form. This approach "
                "acknowledges that intelligence emerges through interaction with the physical "
                "environment. The field encompasses humanoid robotics, where machines exhibit "
                "human-like behaviors and capabilities.",
        module_type=ModuleType.PHYSICAL_AI_OVERVIEW
    )

    print(f"Created module: {physical_ai_module.title}")
    print(f"Module ID: {physical_ai_module.id}")

    # ROS2 module
    ros2_module = service.create_module(
        title="ROS2 Robosystem Nervous System",
        content="ROS2 provides the nervous system for robotic systems. "
                "It offers a robust framework for communication between different components "
                "of a robot through nodes, topics, services, and actions. "
                "The middleware handles message passing, service calls, and action executions "
                "with appropriate quality of service settings.",
        module_type=ModuleType.ROS2_ROBOSYSTEM_NERVOUS_SYSTEM
    )

    print(f"Created module: {ros2_module.title}")

    # Validate individual modules
    print("\nValidating individual modules...")
    for module in [physical_ai_module, ros2_module]:
        validation = service.validate_module_academic_standards(module)
        print(f"Module '{module.title}' compliance: {validation['overall_compliance']}")
        if not validation['overall_compliance']:
            print(f"  Issues: {validation}")

    # Validate combination of modules
    print("\nValidating module combination...")
    combination_validation = service.validate_module_combination([physical_ai_module, ros2_module])
    print(f"Combination compliance: {combination_validation['overall_consistency']}")

    # Get module summary
    print("\nGetting module summary...")
    summary = service.get_module_summary(physical_ai_module.id)
    if summary:
        print(f"Summary for '{summary['title']}':")
        print(f"  - Academic compliance: {summary['academic_compliance']}")
        print(f"  - Word count: {summary['word_count']}")
        print(f"  - Citation density: {summary['citation_density']:.2f}")


if __name__ == "__main__":
    test_module_service()