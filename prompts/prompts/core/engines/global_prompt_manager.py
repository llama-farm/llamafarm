"""Global prompt manager for high-level prompts applied across templates."""

from typing import Any, Dict, List, Optional
import logging

from ..models.config import GlobalPromptConfig
from ..models.template import PromptTemplate


logger = logging.getLogger(__name__)


class GlobalPromptManager:
    """
    Manager for high-level global prompts that can be applied to any template.
    Global prompts provide system-wide behavior and context.
    """
    
    def __init__(self):
        """Initialize the global prompt manager."""
        self.global_prompts: Dict[str, GlobalPromptConfig] = {}
        self.application_stats: Dict[str, int] = {}
    
    def add_global_prompt(self, global_prompt: GlobalPromptConfig) -> None:
        """Add a global prompt to the manager."""
        self.global_prompts[global_prompt.global_id] = global_prompt
        self.application_stats[global_prompt.global_id] = 0
        logger.info(f"Added global prompt: {global_prompt.global_id}")
    
    def remove_global_prompt(self, global_id: str) -> bool:
        """Remove a global prompt."""
        if global_id in self.global_prompts:
            del self.global_prompts[global_id]
            if global_id in self.application_stats:
                del self.application_stats[global_id]
            logger.info(f"Removed global prompt: {global_id}")
            return True
        return False
    
    def get_global_prompt(self, global_id: str) -> Optional[GlobalPromptConfig]:
        """Get a global prompt by ID."""
        return self.global_prompts.get(global_id)
    
    def list_global_prompts(self) -> List[GlobalPromptConfig]:
        """List all global prompts."""
        return list(self.global_prompts.values())
    
    def get_applicable_prompts(
        self,
        template_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[GlobalPromptConfig]:
        """
        Get all global prompts that apply to a specific template.
        
        Args:
            template_id: The template ID to check
            context: Optional context for conditional application
            
        Returns:
            List of applicable global prompts, sorted by priority
        """
        applicable_prompts = []
        
        for global_prompt in self.global_prompts.values():
            if global_prompt.applies_to_template(template_id, context):
                applicable_prompts.append(global_prompt)
        
        # Sort by priority (lower number = higher priority)
        applicable_prompts.sort(key=lambda gp: gp.priority)
        
        return applicable_prompts
    
    def apply_global_prompts(
        self,
        template: PromptTemplate,
        global_prompts: List[GlobalPromptConfig]
    ) -> PromptTemplate:
        """
        Apply global prompts to a template, creating an enhanced version.
        
        Args:
            template: The base template
            global_prompts: List of global prompts to apply
            
        Returns:
            Enhanced template with global prompts applied
        """
        if not global_prompts:
            return template
        
        # Create a copy of the template
        enhanced_template = PromptTemplate(**template.dict())
        
        # Collect all prompt components
        system_prompts = []
        prefix_prompts = []
        suffix_prompts = []
        
        for global_prompt in global_prompts:
            # Update application stats
            self.application_stats[global_prompt.global_id] = (
                self.application_stats.get(global_prompt.global_id, 0) + 1
            )
            
            if global_prompt.system_prompt:
                system_prompts.append(global_prompt.system_prompt)
            
            if global_prompt.prefix_prompt:
                prefix_prompts.append(global_prompt.prefix_prompt)
            
            if global_prompt.suffix_prompt:
                suffix_prompts.append(global_prompt.suffix_prompt)
        
        # Apply system prompts (these typically go in metadata or special handling)
        if system_prompts:
            enhanced_template.metadata.description = (
                enhanced_template.metadata.description or ""
            ) + "\\n\\nSystem Context: " + " ".join(system_prompts)
        
        # Apply prefix and suffix prompts to template content
        original_template = enhanced_template.template
        
        # Build enhanced template content
        enhanced_content_parts = []
        
        # Add prefix prompts
        if prefix_prompts:
            enhanced_content_parts.extend(prefix_prompts)
        
        # Add original template content
        enhanced_content_parts.append(original_template)
        
        # Add suffix prompts
        if suffix_prompts:
            enhanced_content_parts.extend(suffix_prompts)
        
        # Combine all parts
        enhanced_template.template = "\\n\\n".join(enhanced_content_parts)
        
        # Update template metadata to indicate global prompts were applied
        enhanced_template.metadata.description = (
            enhanced_template.metadata.description or ""
        ) + f"\\n\\nGlobal prompts applied: {[gp.global_id for gp in global_prompts]}"
        
        logger.debug(
            f"Applied {len(global_prompts)} global prompts to template {template.template_id}"
        )
        
        return enhanced_template
    
    def create_system_prompt(
        self,
        global_id: str,
        name: str,
        system_prompt: str,
        applies_to: List[str] = None,
        conditions: Dict[str, Any] = None,
        priority: int = 100
    ) -> GlobalPromptConfig:
        """
        Create a system-level global prompt.
        
        Args:
            global_id: Unique identifier
            name: Human-readable name
            system_prompt: The system prompt content
            applies_to: Template patterns this applies to (default: all)
            conditions: Conditions for application
            priority: Application priority
            
        Returns:
            Created global prompt configuration
        """
        return GlobalPromptConfig(
            global_id=global_id,
            name=name,
            system_prompt=system_prompt,
            applies_to=applies_to or ["*"],
            conditions=conditions or {},
            priority=priority
        )
    
    def create_context_prompt(
        self,
        global_id: str,
        name: str,
        prefix_prompt: str = None,
        suffix_prompt: str = None,
        applies_to: List[str] = None,
        conditions: Dict[str, Any] = None,
        priority: int = 100
    ) -> GlobalPromptConfig:
        """
        Create a context-adding global prompt.
        
        Args:
            global_id: Unique identifier
            name: Human-readable name
            prefix_prompt: Prompt to add before template
            suffix_prompt: Prompt to add after template
            applies_to: Template patterns this applies to
            conditions: Conditions for application
            priority: Application priority
            
        Returns:
            Created global prompt configuration
        """
        return GlobalPromptConfig(
            global_id=global_id,
            name=name,
            prefix_prompt=prefix_prompt,
            suffix_prompt=suffix_prompt,
            applies_to=applies_to or ["*"],
            conditions=conditions or {},
            priority=priority
        )
    
    def create_domain_prompt(
        self,
        domain: str,
        system_context: str,
        prefix_instructions: str = None,
        priority: int = 50
    ) -> GlobalPromptConfig:
        """
        Create a domain-specific global prompt.
        
        Args:
            domain: The domain name
            system_context: System context for the domain
            prefix_instructions: Instructions to add before template
            priority: Application priority
            
        Returns:
            Created global prompt configuration
        """
        return GlobalPromptConfig(
            global_id=f"domain_{domain}",
            name=f"Domain Context: {domain.title()}",
            system_prompt=system_context,
            prefix_prompt=prefix_instructions,
            applies_to=["*"],
            conditions={"domain": domain},
            priority=priority
        )
    
    def create_role_prompt(
        self,
        role: str,
        role_context: str,
        behavior_instructions: str = None,
        priority: int = 60
    ) -> GlobalPromptConfig:
        """
        Create a role-based global prompt.
        
        Args:
            role: The user role
            role_context: Context about the role
            behavior_instructions: How to behave for this role
            priority: Application priority
            
        Returns:
            Created global prompt configuration
        """
        prefix_prompt = None
        if behavior_instructions:
            prefix_prompt = f"User Role: {role}\\n{behavior_instructions}\\n"
        
        return GlobalPromptConfig(
            global_id=f"role_{role}",
            name=f"Role Context: {role.title()}",
            system_prompt=role_context,
            prefix_prompt=prefix_prompt,
            applies_to=["*"],
            conditions={"user_role": role},
            priority=priority
        )
    
    def validate_global_prompt(self, global_prompt: GlobalPromptConfig) -> List[str]:
        """Validate a global prompt configuration."""
        errors = []
        
        # Basic validation
        if not global_prompt.global_id:
            errors.append("Global ID is required")
        
        if not global_prompt.name:
            errors.append("Name is required")
        
        # Content validation
        has_content = any([
            global_prompt.system_prompt,
            global_prompt.prefix_prompt,
            global_prompt.suffix_prompt
        ])
        
        if not has_content:
            errors.append("At least one prompt type (system, prefix, or suffix) is required")
        
        # Pattern validation
        if not global_prompt.applies_to:
            errors.append("At least one 'applies_to' pattern is required")
        
        return errors
    
    def test_global_prompt_application(
        self,
        template: PromptTemplate,
        global_prompt_ids: List[str],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Test applying global prompts to a template.
        
        Args:
            template: Template to test with
            global_prompt_ids: List of global prompt IDs to test
            context: Context for conditional application
            
        Returns:
            Test results
        """
        results = {
            "original_template": template.template,
            "applicable_prompts": [],
            "enhanced_template": None,
            "changes": {
                "system_prompts_added": 0,
                "prefix_prompts_added": 0,
                "suffix_prompts_added": 0
            },
            "errors": []
        }
        
        try:
            # Get specified global prompts
            global_prompts = []
            for global_id in global_prompt_ids:
                global_prompt = self.get_global_prompt(global_id)
                if global_prompt:
                    if global_prompt.applies_to_template(template.template_id, context):
                        global_prompts.append(global_prompt)
                        results["applicable_prompts"].append(global_id)
                else:
                    results["errors"].append(f"Global prompt not found: {global_id}")
            
            # Apply global prompts
            if global_prompts:
                enhanced_template = self.apply_global_prompts(template, global_prompts)
                results["enhanced_template"] = enhanced_template.template
                
                # Count changes
                for gp in global_prompts:
                    if gp.system_prompt:
                        results["changes"]["system_prompts_added"] += 1
                    if gp.prefix_prompt:
                        results["changes"]["prefix_prompts_added"] += 1
                    if gp.suffix_prompt:
                        results["changes"]["suffix_prompts_added"] += 1
            else:
                results["enhanced_template"] = template.template
                results["errors"].append("No applicable global prompts found")
        
        except Exception as e:
            results["errors"].append(f"Error applying global prompts: {str(e)}")
        
        return results
    
    def get_application_stats(self) -> Dict[str, Any]:
        """Get statistics about global prompt applications."""
        total_applications = sum(self.application_stats.values())
        
        stats = {
            "total_global_prompts": len(self.global_prompts),
            "total_applications": total_applications,
            "by_prompt": self.application_stats.copy()
        }
        
        # Add usage percentages
        if total_applications > 0:
            stats["usage_percentages"] = {
                global_id: (count / total_applications) * 100
                for global_id, count in self.application_stats.items()
            }
        else:
            stats["usage_percentages"] = {}
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset application statistics."""
        self.application_stats = {global_id: 0 for global_id in self.global_prompts.keys()}
    
    def clear(self) -> None:
        """Clear all global prompts and stats."""
        self.global_prompts.clear()
        self.application_stats.clear()