"""
Project Configuration Tool

Provides LLMs with comprehensive project configuration management capabilities
built on the Project Schema System for safe, validated configuration changes.
"""

import logging
from typing import Any, Dict, List, Literal, Optional

from atomic_agents import BaseIOSchema, BaseTool

# Import the schema system
from tools.core.project_schema_system import (
    LLMConfigurationAssistant,
    ProjectConfigManipulator,
    ProjectSchemaIntrospector
)

logger = logging.getLogger(__name__)


class ProjectConfigToolInput(BaseIOSchema):
    """Input schema for project configuration tool."""
    action: Literal["get_schema", "analyze_config", "modify_config", "suggest_changes"]
    namespace: str
    project_id: str
    user_intent: Optional[str] = None  # Required for suggest_changes and modify_config
    changes: Optional[List[Dict[str, Any]]] = None  # Required for modify_config


class ProjectConfigToolOutput(BaseIOSchema):
    """Output schema for project configuration tool."""
    success: bool
    message: str
    action: str
    data: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None


class ProjectConfigTool(BaseTool[ProjectConfigToolInput, ProjectConfigToolOutput]):
    """Tool for comprehensive project configuration management."""

    def __init__(self):
        super().__init__()

    def run(self, input_data: ProjectConfigToolInput) -> ProjectConfigToolOutput:
        """Execute the project configuration tool based on the specified action."""
        try:
            logger.info(
                f"Executing project config tool - action: {input_data.action}, "
                f"namespace: {input_data.namespace}, project_id: {input_data.project_id}"
            )

            if input_data.action == "get_schema":
                return self._get_schema(input_data.namespace, input_data.project_id)

            elif input_data.action == "analyze_config":
                return self._analyze_config(input_data.namespace, input_data.project_id)

            elif input_data.action == "suggest_changes":
                if not input_data.user_intent:
                    return ProjectConfigToolOutput(
                        success=False,
                        message="user_intent is required for suggest_changes action",
                        action=input_data.action,
                        errors=["Missing required parameter: user_intent"]
                    )
                return self._suggest_changes(
                    input_data.namespace, input_data.project_id, input_data.user_intent
                )

            elif input_data.action == "modify_config":
                if not input_data.changes:
                    return ProjectConfigToolOutput(
                        success=False,
                        message="changes parameter is required for modify_config action",
                        action=input_data.action,
                        errors=["Missing required parameter: changes"]
                    )
                return self._modify_config(
                    input_data.namespace,
                    input_data.project_id,
                    input_data.user_intent or "Configuration changes",
                    input_data.changes
                )

            else:
                return ProjectConfigToolOutput(
                    success=False,
                    message=f"Unknown action: {input_data.action}",
                    action=input_data.action,
                    errors=[f"Unsupported action: {input_data.action}"]
                )

        except Exception as e:
            logger.exception(f"Unexpected error in project config tool: {e}")
            return ProjectConfigToolOutput(
                success=False,
                message=f"Tool execution failed: {str(e)}",
                action=input_data.action,
                errors=[str(e)]
            )

    def _get_schema(self, namespace: str, project_id: str) -> ProjectConfigToolOutput:
        """Get comprehensive schema documentation."""
        try:
            assistant = LLMConfigurationAssistant(namespace, project_id)
            schema_doc = assistant.get_schema_documentation()

            # Also get the structured schema data
            introspector = ProjectSchemaIntrospector()
            structured_schema = introspector.get_llm_friendly_schema()

            return ProjectConfigToolOutput(
                success=True,
                message=f"Retrieved configuration schema for {namespace}/{project_id}",
                action="get_schema",
                data={
                    "documentation": schema_doc,
                    "structured_schema": structured_schema,
                    "field_count": len(structured_schema.get("sections", {}))
                }
            )

        except Exception as e:
            logger.error(f"Failed to get schema for {namespace}/{project_id}: {e}")
            return ProjectConfigToolOutput(
                success=False,
                message=f"Failed to retrieve schema: {str(e)}",
                action="get_schema",
                errors=[str(e)]
            )

    def _analyze_config(self, namespace: str, project_id: str) -> ProjectConfigToolOutput:
        """Analyze the current project configuration."""
        try:
            assistant = LLMConfigurationAssistant(namespace, project_id)
            analysis = assistant.analyze_configuration()

            # Create a human-readable summary
            summary_parts = [
                f"Configuration Analysis for {namespace}/{project_id}:",
                f"- Project: {analysis['project_info']['name']}",
                f"- Namespace: {analysis['project_info']['namespace']}",
                f"- Version: {analysis['project_info']['version']}",
                f"- Configured sections: {', '.join(analysis['sections_configured'])}",
            ]

            if analysis['sections_empty']:
                summary_parts.append(f"- Empty sections: {', '.join(analysis['sections_empty'])}")

            if analysis['potential_improvements']:
                summary_parts.append("- Potential improvements:")
                for improvement in analysis['potential_improvements']:
                    summary_parts.append(f"  • {improvement}")

            summary = "\n".join(summary_parts)

            return ProjectConfigToolOutput(
                success=True,
                message=summary,
                action="analyze_config",
                data=analysis
            )

        except Exception as e:
            logger.error(f"Failed to analyze config for {namespace}/{project_id}: {e}")
            return ProjectConfigToolOutput(
                success=False,
                message=f"Failed to analyze configuration: {str(e)}",
                action="analyze_config",
                errors=[str(e)]
            )

    def _suggest_changes(self, namespace: str, project_id: str, user_intent: str) -> ProjectConfigToolOutput:
        """Suggest configuration changes based on user intent."""
        try:
            assistant = LLMConfigurationAssistant(namespace, project_id)
            suggestions = assistant.suggest_configuration_changes(user_intent)

            if not suggestions:
                return ProjectConfigToolOutput(
                    success=True,
                    message=f"No specific configuration changes suggested for: '{user_intent}'",
                    action="suggest_changes",
                    data={
                        "suggestions": [],
                        "user_intent": user_intent,
                        "suggestion_count": 0
                    }
                )

            # Create a summary of suggestions
            summary_parts = [
                f"Configuration change suggestions for: '{user_intent}'",
                f"Found {len(suggestions)} potential changes:",
            ]

            for i, suggestion in enumerate(suggestions, 1):
                summary_parts.append(
                    f"{i}. {suggestion['description']} "
                    f"(field: {suggestion['field_path']}) - {suggestion['rationale']}"
                )

            summary = "\n".join(summary_parts)

            return ProjectConfigToolOutput(
                success=True,
                message=summary,
                action="suggest_changes",
                data={
                    "suggestions": suggestions,
                    "user_intent": user_intent,
                    "suggestion_count": len(suggestions)
                }
            )

        except Exception as e:
            logger.error(f"Failed to suggest changes for {namespace}/{project_id}: {e}")
            return ProjectConfigToolOutput(
                success=False,
                message=f"Failed to suggest changes: {str(e)}",
                action="suggest_changes",
                errors=[str(e)]
            )

    def _modify_config(self, namespace: str, project_id: str, user_intent: str,
                      changes: List[Dict[str, Any]]) -> ProjectConfigToolOutput:
        """Apply configuration changes to a project."""
        try:
            assistant = LLMConfigurationAssistant(namespace, project_id)

            # Apply the changes
            result = assistant.apply_user_changes(user_intent, changes)

            if result["success"]:
                # Save the configuration
                assistant.manipulator.save_config()

                summary_parts = [
                    f"Successfully applied {result['changes_applied']} configuration changes:",
                    f"Intent: {user_intent}",
                ]

                if "actual_changes" in result:
                    summary_parts.append("Changes made:")
                    for change in result["actual_changes"]:
                        change_desc = f"  • {change['field']}: "
                        if change['type'] == 'create':
                            change_desc += f"added '{change['new_value']}'"
                        elif change['type'] == 'delete':
                            change_desc += f"removed '{change['old_value']}'"
                        else:
                            change_desc += f"'{change['old_value']}' → '{change['new_value']}'"
                        summary_parts.append(change_desc)

                summary = "\n".join(summary_parts)

                return ProjectConfigToolOutput(
                    success=True,
                    message=summary,
                    action="modify_config",
                    data={
                        "changes_applied": result["changes_applied"],
                        "changeset_id": result["changeset_id"],
                        "actual_changes": result.get("actual_changes", []),
                        "user_intent": user_intent
                    }
                )

            else:
                return ProjectConfigToolOutput(
                    success=False,
                    message="Failed to apply configuration changes - validation errors occurred",
                    action="modify_config",
                    data=result,
                    errors=["Configuration validation failed"]
                )

        except Exception as e:
            logger.error(f"Failed to modify config for {namespace}/{project_id}: {e}")
            return ProjectConfigToolOutput(
                success=False,
                message=f"Failed to modify configuration: {str(e)}",
                action="modify_config",
                errors=[str(e)]
            )


# Example usage
if __name__ == "__main__":
    # Example: Get schema documentation
    tool = ProjectConfigTool()

    result = tool.run(ProjectConfigToolInput(
        action="get_schema",
        namespace="test",
        project_id="sample"
    ))

    print(f"Schema result: {result.message}")

    # Example: Analyze configuration
    result = tool.run(ProjectConfigToolInput(
        action="analyze_config",
        namespace="test",
        project_id="sample"
    ))

    print(f"Analysis result: {result.message}")

    # Example: Suggest changes
    result = tool.run(ProjectConfigToolInput(
        action="suggest_changes",
        namespace="test",
        project_id="sample",
        user_intent="I want to use OpenAI instead of local models"
    ))

    print(f"Suggestions: {result.message}")
