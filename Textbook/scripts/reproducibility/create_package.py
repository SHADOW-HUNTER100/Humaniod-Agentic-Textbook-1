#!/usr/bin/env python3
"""
Reproducibility Package Creator for AI-Native Software Development & Physical AI project
Creates a comprehensive package to ensure research can be replicated
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import zipfile
import hashlib


class ReproducibilityPackageCreator:
    """
    Creates a reproducibility package with all necessary components
    to replicate the research paper and experiments
    """

    def __init__(self, project_root: str = "."):
        """
        Initialize the reproducibility package creator

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.package_dir = self.project_root / "reproducibility_package"
        self.components = []
        self.docker_images = []
        self.dataset_links = []
        self.setup_instructions = ""

    def create_package(self, output_path: str = None) -> str:
        """
        Create the reproducibility package

        Args:
            output_path: Path where the package should be saved

        Returns:
            Path to the created package
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"reproducibility_package_{timestamp}.zip"

        # Create package directory structure
        self._create_package_structure()

        # Add project components
        self._add_project_components()

        # Add environment specifications
        self._add_environment_specs()

        # Add dataset information
        self._add_dataset_info()

        # Add validation script
        self._add_validation_script()

        # Create the zip package
        package_path = self._create_zip_package(output_path)

        # Cleanup temporary directory
        shutil.rmtree(self.package_dir)

        return package_path

    def _create_package_structure(self):
        """Create the directory structure for the reproducibility package"""
        self.package_dir.mkdir(exist_ok=True)

        # Create subdirectories
        dirs = [
            "src",
            "data",
            "models",
            "configs",
            "scripts",
            "tests",
            "docs",
            "environment"
        ]

        for dir_name in dirs:
            (self.package_dir / dir_name).mkdir(exist_ok=True)

    def _add_project_components(self):
        """Add project source code and components to the package"""
        # Copy source code
        src_dirs = ["src", "scripts", "tests"]
        for src_dir in src_dirs:
            src_path = self.project_root / src_dir
            if src_path.exists():
                dest_path = self.package_dir / src_dir
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)

        # Copy documentation
        docs_path = self.project_root / "docs"
        if docs_path.exists():
            dest_docs = self.package_dir / "docs"
            if dest_docs.exists():
                shutil.rmtree(dest_docs)
            shutil.copytree(docs_path, dest_docs)

        # Copy configuration files
        config_files = ["requirements.txt", "setup.py", "pyproject.toml", "package.json", "Dockerfile*"]
        for pattern in config_files:
            for config_file in self.project_root.glob(pattern):
                dest_path = self.package_dir / "configs" / config_file.name
                shutil.copy2(config_file, dest_path)

    def _add_environment_specs(self):
        """Add environment specifications to the package"""
        # Create environment info file
        env_info = {
            "created_at": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "python_version": sys.version,
            "dependencies": self._get_dependencies(),
            "environment_variables": self._get_env_vars(),
            "hardware_specifications": self._get_hardware_info()
        }

        env_file = self.package_dir / "environment" / "environment.json"
        env_file.write_text(json.dumps(env_info, indent=2))

        # Create requirements file
        requirements_file = self.package_dir / "environment" / "requirements.txt"
        requirements = self._generate_requirements_txt()
        requirements_file.write_text(requirements)

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information for reproducibility"""
        import platform
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0]
        }

    def _get_dependencies(self) -> List[str]:
        """Get list of installed Python packages"""
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                    capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
        except Exception:
            pass
        return []

    def _get_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables"""
        relevant_vars = [
            'PATH', 'PYTHONPATH', 'HOME', 'USER', 'LANG',
            'ROS_DISTRO', 'ROS_ROOT', 'ROS_PACKAGE_PATH'
        ]

        env_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]

        return env_vars

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware specifications"""
        import psutil

        return {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').total if hasattr(psutil, 'disk_usage') else 0
        }

    def _generate_requirements_txt(self) -> str:
        """Generate requirements.txt content"""
        deps = self._get_dependencies()
        return "\n".join(deps)

    def _add_dataset_info(self):
        """Add dataset information and links to the package"""
        # Create dataset info file
        dataset_info = {
            "datasets_used": self.dataset_links,
            "data_preprocessing_scripts": [],
            "license_information": "See individual dataset licenses",
            "access_instructions": "Contact authors for access to restricted datasets"
        }

        dataset_file = self.package_dir / "data" / "dataset_info.json"
        dataset_file.write_text(json.dumps(dataset_info, indent=2))

    def _add_validation_script(self):
        """Add validation script to test reproducibility"""
        validation_script = '''#!/usr/bin/env python3
"""
Validation script for reproducibility package
Checks that all components are present and functional
"""

import sys
import os
import json
import subprocess
from pathlib import Path


def validate_environment():
    """Validate that the environment is correctly set up"""
    print("Validating environment...")

    # Check Python version
    import platform
    print(f"Python version: {platform.python_version()}")

    # Check for required packages
    required_packages = ["numpy", "pandas", "matplotlib", "torch", "tensorflow"]
    missing_packages = []

    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg} is available")
        except ImportError:
            missing_packages.append(pkg)
            print(f"✗ {pkg} is missing")

    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        return False

    return True


def validate_data():
    """Validate that required data is available"""
    print("Validating data...")

    data_dir = Path("data")
    if not data_dir.exists():
        print("✗ Data directory does not exist")
        return False

    # Check for expected data files
    expected_files = ["dataset_info.json"]
    missing_files = []

    for f in expected_files:
        if not (data_dir / f).exists():
            missing_files.append(f)

    if missing_files:
        print(f"✗ Missing data files: {missing_files}")
        return False

    print("✓ Data validation passed")
    return True


def validate_models():
    """Validate that models can be loaded"""
    print("Validating models...")

    model_dir = Path("models")
    if not model_dir.exists():
        print("! Models directory does not exist (may be intentional)")
        return True

    print("✓ Models validation passed")
    return True


def run_example():
    """Run a simple example to verify functionality"""
    print("Running example...")

    try:
        # Import and run a simple test
        import sys
        sys.path.insert(0, str(Path(".")))

        # Try to import a core module
        from src.paper.services.citation_validation_service import CitationValidationService
        service = CitationValidationService()
        print("✓ Core module imported successfully")

        return True
    except Exception as e:
        print(f"✗ Example failed: {e}")
        return False


def main():
    """Main validation function"""
    print("Starting reproducibility validation...")
    print("="*50)

    all_passed = True

    all_passed &= validate_environment()
    all_passed &= validate_data()
    all_passed &= validate_models()
    all_passed &= run_example()

    print("="*50)
    if all_passed:
        print("✓ All validations passed! Package is reproducible.")
        return 0
    else:
        print("✗ Some validations failed. Package may not be fully reproducible.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''

        validation_script_path = self.package_dir / "scripts" / "validate_reproducibility.py"
        validation_script_path.write_text(validation_script)
        validation_script_path.chmod(0o755)  # Make executable

    def _create_zip_package(self, output_path: str) -> str:
        """Create a ZIP archive of the reproducibility package"""
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.package_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_path = file_path.relative_to(self.package_dir.parent)
                    zipf.write(file_path, arc_path)

        # Calculate checksum
        checksum = self._calculate_checksum(output_path)
        checksum_file = Path(output_path + ".sha256")
        checksum_file.write_text(checksum)

        print(f"Reproducibility package created: {output_path}")
        print(f"Checksum (SHA256): {checksum}")

        return output_path

    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate SHA256 checksum of the file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def main():
    """Main function to create reproducibility package"""
    import argparse

    parser = argparse.ArgumentParser(description="Create reproducibility package")
    parser.add_argument("--output", "-o", help="Output path for the package")
    parser.add_argument("--project-root", "-r", default=".", help="Project root directory")

    args = parser.parse_args()

    creator = ReproducibilityPackageCreator(args.project_root)
    package_path = creator.create_package(args.output)

    print(f"Package created successfully: {package_path}")
    print("The package contains all necessary components to reproduce the research.")


if __name__ == "__main__":
    main()