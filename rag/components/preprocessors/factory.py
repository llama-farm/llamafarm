"""Preprocessor Factory for dynamic preprocessor loading."""

import importlib.util
from pathlib import Path
from typing import Any, Optional

import yaml

from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.preprocessors.factory")


class PreprocessorFactory:
    """Factory for creating preprocessor instances."""

    # Cache for loaded preprocessor configurations
    _preprocessor_configs: dict[str, list[dict[str, Any]]] = {}
    # Cache for loaded preprocessor classes
    _preprocessor_classes: dict[str, type] = {}

    @classmethod
    def discover_preprocessors(cls) -> dict[str, list[dict[str, Any]]]:
        """Discover all available preprocessors from configuration files.

        Returns:
            Dictionary mapping preprocessor types to their configurations
        """
        if cls._preprocessor_configs:
            return cls._preprocessor_configs

        preprocessors_dir = Path(__file__).parent
        preprocessor_types: dict[str, list[dict[str, Any]]] = {}

        # Scan all subdirectories for config.yaml files
        for subdir in preprocessors_dir.iterdir():
            if not subdir.is_dir() or subdir.name.startswith("_"):
                continue

            config_file = subdir / "config.yaml"
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = yaml.safe_load(f)
                        if config and "preprocessors" in config:
                            preprocessor_type = subdir.name
                            preprocessor_types[preprocessor_type] = config[
                                "preprocessors"
                            ]
                            logger.info(
                                f"Discovered {len(config['preprocessors'])} {preprocessor_type} preprocessors"
                            )
                except Exception as e:
                    logger.error(f"Failed to load config from {config_file}: {e}")

        cls._preprocessor_configs = preprocessor_types
        return preprocessor_types

    @classmethod
    def list_preprocessors(cls, preprocessor_type: Optional[str] = None) -> list[str]:
        """List available preprocessors.

        Args:
            preprocessor_type: Optional filter by preprocessor type (ocr, markitdown, etc.)

        Returns:
            List of preprocessor names
        """
        preprocessors = cls.discover_preprocessors()
        preprocessor_names: list[str] = []

        for ptype, configs in preprocessors.items():
            if preprocessor_type and ptype != preprocessor_type:
                continue
            preprocessor_names.extend(config["name"] for config in configs)

        return preprocessor_names

    @classmethod
    def get_preprocessor_info(
        cls, preprocessor_name: str
    ) -> Optional[dict[str, Any]]:
        """Get information about a specific preprocessor.

        Args:
            preprocessor_name: Name of the preprocessor (e.g., "PaddleOCRPreprocessor")

        Returns:
            Preprocessor configuration dictionary or None
        """
        preprocessors = cls.discover_preprocessors()

        for preprocessor_type, configs in preprocessors.items():
            for config in configs:
                if config["name"] == preprocessor_name:
                    config["preprocessor_type"] = preprocessor_type
                    return config

        return None

    @classmethod
    def load_preprocessor_class(cls, preprocessor_name: str) -> Optional[type]:
        """Load a preprocessor class dynamically.

        Args:
            preprocessor_name: Name of the preprocessor (e.g., "PaddleOCRPreprocessor")

        Returns:
            Preprocessor class or None
        """
        # Check cache
        if preprocessor_name in cls._preprocessor_classes:
            return cls._preprocessor_classes[preprocessor_name]

        # Get preprocessor info
        info = cls.get_preprocessor_info(preprocessor_name)
        if not info:
            logger.error(f"Preprocessor {preprocessor_name} not found")
            return None

        # Check dependencies BEFORE trying to load
        deps = info.get("dependencies", {})
        required_deps = deps.get("required", [])
        missing_deps: list[str] = []

        # Package name to import name mapping for special cases
        PACKAGE_TO_IMPORT = {
            "python-docx": "docx",
            "beautifulsoup4": "bs4",
            "opencv-python": "cv2",
            "pillow": "PIL",
            "scikit-learn": "sklearn",
            "paddleocr": "paddleocr",
            "paddlepaddle": "paddle",
            "pymupdf": "fitz",
            "ocrmypdf": "ocrmypdf",
        }

        for dep in required_deps:
            # Strip version specifiers (>=, ==, etc.)
            import re
            clean_dep = re.split(r'[<>=!]', dep)[0]

            # Get the import name (may differ from package name)
            import_name = PACKAGE_TO_IMPORT.get(clean_dep, clean_dep.replace("-", "_"))

            try:
                # Try to import the dependency
                __import__(import_name)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            logger.warning(
                f"Preprocessor {preprocessor_name} missing dependencies: {missing_deps}"
            )
            # Return None to trigger fallback
            return None

        preprocessor_type = info["preprocessor_type"]

        # Try to find the implementation file
        preprocessors_dir = Path(__file__).parent
        impl_dir = preprocessors_dir / preprocessor_type

        # Convert class name to module name (e.g., PaddleOCRPreprocessor -> paddleocr_preprocessor)
        import re

        # Handle acronyms properly: PaddleOCRPreprocessor -> paddleocr_preprocessor
        module_name = re.sub(r'(?<!^)(?=[A-Z])', '_', preprocessor_name).lower()

        # Try common naming patterns
        # Remove "Preprocessor" suffix for base name
        class_base = preprocessor_name.replace("Preprocessor", "")

        potential_modules = [
            f"{module_name}.py",
            f"{preprocessor_name.lower()}.py",  # paddleocrpreprocessor.py
            f"{class_base.lower()}_preprocessor.py",  # paddleocr_preprocessor.py
            f"{preprocessor_type}_preprocessor.py",
            f"{preprocessor_type}.py",
            "preprocessor.py",
        ]

        for mod_file in potential_modules:
            mod_path = impl_dir / mod_file
            if mod_path.exists():
                try:
                    # Load the module
                    spec = importlib.util.spec_from_file_location(
                        f"components.preprocessors.{preprocessor_type}.{mod_path.stem}",
                        mod_path,
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Try to get the class
                        if hasattr(module, preprocessor_name):
                            preprocessor_class = getattr(module, preprocessor_name)
                            cls._preprocessor_classes[
                                preprocessor_name
                            ] = preprocessor_class
                            logger.info(
                                f"Loaded preprocessor {preprocessor_name} from {mod_path}"
                            )
                            return preprocessor_class
                except Exception as e:
                    logger.error(f"Failed to load preprocessor from {mod_path}: {e}")
                    continue

        logger.error(f"Could not find implementation for {preprocessor_name}")
        return None

    @classmethod
    def create(
        cls, preprocessor_name: str, config: Optional[dict[str, Any]] = None
    ) -> Optional[Any]:
        """Create a preprocessor instance.

        Args:
            preprocessor_name: Name of the preprocessor
            config: Configuration dictionary

        Returns:
            Preprocessor instance or None
        """
        preprocessor_class = cls.load_preprocessor_class(preprocessor_name)
        if not preprocessor_class:
            # Get info to provide helpful error message
            info = cls.get_preprocessor_info(preprocessor_name)
            if info:
                deps = info.get("dependencies", {})
                required_deps = deps.get("required", [])
                if required_deps:
                    # Quote each dependency to handle version specifiers like >=
                    quoted_deps = ' '.join(f"'{dep}'" for dep in required_deps)
                    raise ImportError(
                        f"Preprocessor '{preprocessor_name}' requires missing dependencies: {required_deps}\n"
                        f"Install with: uv pip install {quoted_deps}"
                    )

            raise ValueError(
                f"Preprocessor '{preprocessor_name}' not found. "
                f"Available preprocessors: {cls.list_preprocessors()}"
            )

        try:
            return preprocessor_class(config=config)
        except Exception as e:
            logger.error(f"Failed to create preprocessor {preprocessor_name}: {e}")
            raise
