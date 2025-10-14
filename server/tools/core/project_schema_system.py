"""
Project Schema Understanding and Manipulation System

This module provides a common interface for both tools and LLMs to:
1. Understand project configuration schemas and structure
2. Safely manipulate project configurations
3. Track changes and provide rollback capabilities
4. Generate human/LLM-friendly documentation

The system builds on the existing LlamaFarmConfig Pydantic model and JSON schema.
"""

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

# Import the existing configuration system
import sys
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
from config.datamodel import LlamaFarmConfig
from config import load_config, save_config
from services.project_service import ProjectService

logger = logging.getLogger(__name__)


@dataclass
class ConfigFieldInfo:
    """Rich information about a configuration field for LLM understanding"""
    name: str
    field_type: str
    description: str
    required: bool
    default_value: Any = None
    examples: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    nested_fields: Optional[List['ConfigFieldInfo']] = None
    enum_values: Optional[List[str]] = None
    llm_guidance: Optional[str] = None  # Specific guidance for LLMs


@dataclass
class ConfigChange:
    """Represents a single configuration change"""
    field_path: str  # e.g., "rag.strategies[0].components.parser.type"
    old_value: Any
    new_value: Any
    change_type: str  # "create", "update", "delete"
    timestamp: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None


@dataclass
class ConfigChangeSet:
    """Collection of related configuration changes"""
    changes: List[ConfigChange] = field(default_factory=list)
    description: str = ""
    user_intent: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def add_change(self, change: ConfigChange):
        """Add a change to this changeset"""
        self.changes.append(change)


class ProjectSchemaIntrospector:
    """Introspects the LlamaFarmConfig schema to provide structured information"""

    def __init__(self):
        self._schema_cache: Optional[Dict[str, ConfigFieldInfo]] = None
        self._load_json_schema()

    def _load_json_schema(self):
        """Load the JSON schema for additional metadata"""
        try:
            schema_path = repo_root / "config" / "schema.yaml"
            if schema_path.exists():
                import yaml
                with open(schema_path) as f:
                    self._json_schema = yaml.safe_load(f)
            else:
                self._json_schema = {}
        except Exception as e:
            logger.warning(f"Could not load JSON schema: {e}")
            self._json_schema = {}

    def get_all_fields(self) -> Dict[str, ConfigFieldInfo]:
        """Get comprehensive information about all configuration fields"""
        if self._schema_cache is None:
            self._schema_cache = self._build_field_info()
        return self._schema_cache

    def _build_field_info(self) -> Dict[str, ConfigFieldInfo]:
        """Build detailed field information from the Pydantic model"""
        fields = {}

        # Introspect the main LlamaFarmConfig model
        for field_name, field_info in LlamaFarmConfig.model_fields.items():
            fields[field_name] = self._analyze_field(
                field_name, field_info, f"LlamaFarmConfig.{field_name}"
            )

        return fields

    def _analyze_field(
        self, name: str, field_info: FieldInfo, path: str
    ) -> ConfigFieldInfo:
        """Analyze a single Pydantic field to extract comprehensive information"""

        # Extract basic information
        field_type = str(field_info.annotation) if field_info.annotation else "Any"
        description = field_info.description or f"Configuration field: {name}"
        required = field_info.is_required()
        default_value = (
            field_info.default if field_info.default is not Ellipsis else None
        )

        # Extract examples from field info
        examples = []
        if hasattr(field_info, 'examples') and field_info.examples:
            examples = field_info.examples

        # Extract constraints
        constraints: dict[str, Any] = {}
        if hasattr(field_info, 'constraints'):
            constraints = field_info.constraints or {}

        # Check for enum values
        enum_values = None
        if field_info.annotation and hasattr(field_info.annotation, '__members__'):
            enum_values = list(field_info.annotation.__members__.keys())

        # Generate LLM guidance
        llm_guidance = self._generate_llm_guidance(
            name, field_type, description, constraints, enum_values
        )

        return ConfigFieldInfo(
            name=name,
            field_type=field_type,
            description=description,
            required=required,
            default_value=default_value,
            examples=examples,
            constraints=constraints,
            enum_values=enum_values,
            llm_guidance=llm_guidance
        )

    def _generate_llm_guidance(self, name: str, field_type: str, description: str,
                             constraints: Dict[str, Any],
                             enum_values: Optional[List[str]]) -> str:
        """Generate specific guidance for LLMs on how to handle this field"""

        guidance_parts: list[str] = [f"Field '{name}': {description}"]

        if enum_values:
            guidance_parts.append(f"Must be one of: {', '.join(enum_values)}")

        if constraints:
            if 'ge' in constraints:
                guidance_parts.append(f"Minimum value: {constraints['ge']}")
            if 'le' in constraints:
                guidance_parts.append(f"Maximum value: {constraints['le']}")
            if 'min_length' in constraints:
                guidance_parts.append(f"Minimum length: {constraints['min_length']}")
            if 'max_length' in constraints:
                guidance_parts.append(f"Maximum length: {constraints['max_length']}")
            if 'pattern' in constraints:
                guidance_parts.append(f"Must match pattern: {constraints['pattern']}")

        # Add field-specific guidance
        guidance_map = {
            'name': (
                "Should be a valid project identifier without spaces or "
                "special characters"
            ),
            'namespace': (
                "Should be a valid namespace identifier, typically "
                "organization or team name"
            ),
            'version': "Must always be 'v1' for current config version",
            'prompts': (
                "List of prompt configurations. Each prompt needs a name and "
                "either sections or raw_text"
            ),
            'rag': (
                "RAG system configuration. Contains strategies, parsers, "
                "embedders, vector stores, and retrieval strategies"
            ),
            'datasets': (
                "List of dataset configurations. Each dataset needs a name "
                "and list of file hashes"
            ),
            'runtime': (
                "Runtime configuration specifying the AI provider "
                "(openai/ollama) and model details"
            )
        }

        if name in guidance_map:
            guidance_parts.append(guidance_map[name])

        return ". ".join(guidance_parts)

    def get_field_info(self, field_path: str) -> Optional[ConfigFieldInfo]:
        """Get information about a specific field by path (e.g., 'rag.strategies')"""
        fields = self.get_all_fields()

        # For nested paths, we'd need to traverse the structure
        # For now, handle top-level fields
        if '.' not in field_path:
            return fields.get(field_path)

        # TODO: Implement nested field traversal
        return None

    def get_llm_friendly_schema(self) -> Dict[str, Any]:
        """Generate a schema description optimized for LLM understanding"""
        fields = self.get_all_fields()

        schema = {
            "title": "LlamaFarm Project Configuration Schema",
            "description": "Complete schema for LlamaFarm project configuration files",
            "version": "v1",
            "sections": {},
            "field_guidance": {}
        }

        for field_name, field_info in fields.items():
            schema["sections"][field_name] = {
                "description": field_info.description,
                "type": field_info.field_type,
                "required": field_info.required,
                "examples": field_info.examples,
                "constraints": field_info.constraints
            }

            if field_info.enum_values:
                schema["sections"][field_name]["allowed_values"] = (
                    field_info.enum_values
                )

            schema["field_guidance"][field_name] = field_info.llm_guidance

        return schema


class ProjectConfigManipulator:
    """Safely manipulates project configurations with validation and change tracking"""

    def __init__(self, namespace: str, project_id: str):
        self.namespace = namespace
        self.project_id = project_id
        self.introspector = ProjectSchemaIntrospector()
        self._original_config: Optional[LlamaFarmConfig] = None
        self._current_config: Optional[LlamaFarmConfig] = None
        self._change_history: List[ConfigChangeSet] = []

    def load_config(self) -> LlamaFarmConfig:
        """Load the current project configuration"""
        try:
            self._original_config = ProjectService.load_config(
                self.namespace, self.project_id
            )
            self._current_config = deepcopy(self._original_config)
            if self._current_config is None:
                raise ValueError("Failed to load configuration")
            return self._current_config
        except Exception as e:
            logger.error(
                f"Failed to load config for {self.namespace}/{self.project_id}: {e}"
            )
            raise

    def get_current_config(self) -> Optional[LlamaFarmConfig]:
        """Get the current configuration (with any pending changes)"""
        return self._current_config

    def validate_change(
        self, field_path: str, new_value: Any
    ) -> Tuple[bool, Optional[str]]:
        """Validate a proposed configuration change"""
        if self._current_config is None:
            return False, "Configuration not loaded"

        try:
            # Create a test config with the proposed change
            test_config_dict = self._current_config.model_dump()

            # Apply the change to the test config
            self._set_nested_field(test_config_dict, field_path, new_value)

            # Try to validate the modified configuration
            LlamaFarmConfig(**test_config_dict)
            return True, None

        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    def apply_change(
        self, field_path: str, new_value: Any, description: str = ""
    ) -> ConfigChange:
        """Apply a configuration change with validation and tracking"""
        if self._current_config is None:
            raise ValueError("Configuration not loaded")

        # Validate the change first
        valid, error = self.validate_change(field_path, new_value)
        if not valid:
            raise ValueError(f"Invalid configuration change: {error}")

        # Get the old value
        old_value = self._get_nested_field(
            self._current_config.model_dump(), field_path
        )

        # Apply the change
        config_dict = self._current_config.model_dump()
        self._set_nested_field(config_dict, field_path, new_value)
        self._current_config = LlamaFarmConfig(**config_dict)

        # Track the change
        change = ConfigChange(
            field_path=field_path,
            old_value=old_value,
            new_value=new_value,
            change_type="update" if old_value is not None else "create",
            description=description
        )

        return change

    def apply_changeset(self, changeset: ConfigChangeSet) -> bool:
        """Apply a set of related configuration changes atomically"""
        if self._current_config is None:
            raise ValueError("Configuration not loaded")

        # Store the current state in case we need to rollback
        backup_config = deepcopy(self._current_config)

        try:
            # Apply all changes
            for change_spec in changeset.changes:
                self.apply_change(
                    change_spec.field_path,
                    change_spec.new_value,
                    change_spec.description or f"Change to {change_spec.field_path}"
                )

            # If we get here, all changes were successful
            self._change_history.append(changeset)
            return True

        except Exception as e:
            # Rollback to the backup state
            self._current_config = backup_config
            logger.error(f"Failed to apply changeset, rolled back: {e}")
            return False

    def save_config(self) -> LlamaFarmConfig:
        """Save the current configuration to the project"""
        if self._current_config is None:
            raise ValueError("No configuration to save")

        try:
            saved_config = ProjectService.save_config(
                self.namespace, self.project_id, self._current_config
            )
            # Update our original config to reflect the saved state
            self._original_config = deepcopy(saved_config)
            return saved_config
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise

    def get_changes(self) -> List[ConfigChange]:
        """Get all changes made since loading the configuration"""
        if not self._original_config or not self._current_config:
            return []

        # Compare original and current configurations to detect changes
        original_dict = self._original_config.model_dump()
        current_dict = self._current_config.model_dump()

        changes: list[ConfigChange] = []
        self._find_changes(original_dict, current_dict, "", changes)
        return changes

    def rollback_to_original(self):
        """Rollback all changes to the original loaded configuration"""
        if self._original_config:
            self._current_config = deepcopy(self._original_config)

    def _get_nested_field(self, obj: Dict[str, Any], field_path: str) -> Any:
        """Get a nested field value using dot notation"""
        keys = field_path.split('.')
        current = obj

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _set_nested_field(self, obj: Dict[str, Any], field_path: str, value: Any):
        """Set a nested field value using dot notation"""
        keys = field_path.split('.')
        current = obj

        # Navigate to the parent of the target field
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            if isinstance(current, dict):
                current = current[key]

        # Set the final field
        if isinstance(current, dict):
            current[keys[-1]] = value

    def _find_changes(
        self, original: Any, current: Any, path: str, changes: List[ConfigChange]
    ):
        """Recursively find changes between original and current configurations"""
        if isinstance(original, dict) and isinstance(current, dict):
            # Handle dictionary changes
            all_keys = set(original.keys()) | set(current.keys())

            for key in all_keys:
                new_path = f"{path}.{key}" if path else key

                if key not in original:
                    # New field added
                    changes.append(ConfigChange(
                        field_path=new_path,
                        old_value=None,
                        new_value=current[key],
                        change_type="create"
                    ))
                elif key not in current:
                    # Field removed
                    changes.append(ConfigChange(
                        field_path=new_path,
                        old_value=original[key],
                        new_value=None,
                        change_type="delete"
                    ))
                else:
                    # Field potentially modified
                    self._find_changes(original[key], current[key], new_path, changes)

        elif original != current:
            # Value changed
            changes.append(ConfigChange(
                field_path=path,
                old_value=original,
                new_value=current,
                change_type="update"
            ))


class LLMConfigurationAssistant:
    """High-level interface for LLMs to understand and modify project configurations"""

    def __init__(self, namespace: str, project_id: str):
        self.manipulator = ProjectConfigManipulator(namespace, project_id)
        self.introspector = ProjectSchemaIntrospector()

    def get_schema_documentation(self) -> str:
        """Get human-readable documentation of the configuration schema"""
        schema = self.introspector.get_llm_friendly_schema()

        doc_parts = [
            f"# {schema['title']}",
            f"{schema['description']}",
            f"Configuration Version: {schema['version']}",
            "",
            "## Configuration Sections:",
            ""
        ]

        for section_name, section_info in schema["sections"].items():
            doc_parts.extend([
                f"### {section_name}",
                f"**Description:** {section_info['description']}",
                f"**Type:** {section_info['type']}",
                f"**Required:** {'Yes' if section_info['required'] else 'No'}",
            ])

            if section_info.get('allowed_values'):
                doc_parts.append(
                    f"**Allowed Values:** {', '.join(section_info['allowed_values'])}"
                )

            if section_info.get('examples'):
                doc_parts.append(
                    f"**Examples:** {', '.join(map(str, section_info['examples']))}"
                )

            # Add LLM guidance
            if section_name in schema["field_guidance"]:
                doc_parts.append(
                    f"**Guidance:** {schema['field_guidance'][section_name]}"
                )

            doc_parts.append("")

        return "\n".join(doc_parts)

    def analyze_configuration(self) -> Dict[str, Any]:
        """Analyze the current configuration and provide insights for LLMs"""
        config = self.manipulator.load_config()

        analysis = {
            "project_info": {
                "namespace": config.namespace,
                "name": config.name,
                "version": config.version.value
            },
            "sections_configured": [],
            "sections_empty": [],
            "configuration_completeness": {},
            "potential_improvements": []
        }

        # Analyze each section
        config_dict = config.model_dump()

        for section_name, section_value in config_dict.items():
            if section_name in ['version', 'name', 'namespace']:
                continue

            if section_value:
                analysis["sections_configured"].append(section_name)

                # Analyze section completeness
                if section_name == "prompts" and len(section_value) == 0:
                    analysis["sections_empty"].append("prompts")
                    analysis["potential_improvements"].append(
                        "Consider adding custom prompts for better AI responses"
                    )

                if section_name == "datasets" and len(section_value) == 0:
                    analysis["sections_empty"].append("datasets")
                    analysis["potential_improvements"].append(
                        "Add datasets to enable RAG functionality"
                    )

            else:
                analysis["sections_empty"].append(section_name)

        return analysis

    def suggest_configuration_changes(self, user_intent: str) -> List[Dict[str, Any]]:
        """Suggest configuration changes based on user intent"""
        # This is where we could integrate with an LLM to parse user intent
        # and suggest specific configuration changes

        suggestions = []

        # Simple keyword-based suggestions for now
        intent_lower = user_intent.lower()

        if "openai" in intent_lower or "gpt" in intent_lower:
            suggestions.append({
                "field_path": "runtime.provider",
                "new_value": "openai",
                "description": "Switch to OpenAI provider",
                "rationale": "User mentioned OpenAI or GPT"
            })

        if "ollama" in intent_lower or "local" in intent_lower:
            suggestions.append({
                "field_path": "runtime.provider",
                "new_value": "ollama",
                "description": "Switch to local Ollama provider",
                "rationale": "User wants to use local models"
            })

        if "prompt" in intent_lower:
            suggestions.append({
                "field_path": "prompts",
                "new_value": [{
                    "name": "custom_prompt",
                    "raw_text": "You are a helpful assistant."
                }],
                "description": "Add a custom prompt",
                "rationale": "User mentioned prompts"
            })

        return suggestions

    def apply_user_changes(
        self, user_intent: str, changes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply user-requested changes with full tracking"""

        changeset = ConfigChangeSet(
            description=f"User requested changes: {user_intent}",
            user_intent=user_intent
        )

        for change_spec in changes:
            change = ConfigChange(
                field_path=change_spec["field_path"],
                old_value=None,  # Will be determined during application
                new_value=change_spec["new_value"],
                change_type="update",
                description=change_spec.get("description", "")
            )
            changeset.add_change(change)

        success = self.manipulator.apply_changeset(changeset)

        result = {
            "success": success,
            "changes_applied": len(changeset.changes),
            "changeset_id": id(changeset),  # Simple ID for tracking
        }

        if success:
            # Get current changes to show what actually happened
            current_changes = self.manipulator.get_changes()
            result["actual_changes"] = [
                {
                    "field": change.field_path,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "type": change.change_type
                }
                for change in current_changes
            ]

        return result
