"""Template registry for managing prompt templates."""

from typing import Any, Dict, List, Optional, Set
import logging
from datetime import datetime

from ..models.template import PromptTemplate, TemplateType, TemplateComplexity


logger = logging.getLogger(__name__)


class TemplateRegistry:
    """
    Registry for managing prompt templates with search, filtering, and validation capabilities.
    """
    
    def __init__(self):
        """Initialize the template registry."""
        self.templates: Dict[str, PromptTemplate] = {}
        self.template_index: Dict[str, Set[str]] = {
            "by_type": {},
            "by_domain": {},
            "by_complexity": {},
            "by_tags": {},
            "by_use_case": {}
        }
        
        # Registry stats
        self.registration_count = 0
        self.search_count = 0
    
    def register_template(self, template: PromptTemplate) -> None:
        """Register a new template."""
        # Validate template
        validation_errors = self._validate_template(template)
        if validation_errors:
            raise ValueError(f"Template validation failed: {', '.join(validation_errors)}")
        
        # Check for existing template with same ID
        if template.template_id in self.templates:
            logger.warning(f"Overwriting existing template: {template.template_id}")
        
        # Register template
        self.templates[template.template_id] = template
        
        # Update indices
        self._update_indices(template)
        
        self.registration_count += 1
        logger.info(f"Registered template: {template.template_id}")
    
    def unregister_template(self, template_id: str) -> bool:
        """Unregister a template."""
        if template_id not in self.templates:
            return False
        
        template = self.templates[template_id]
        
        # Remove from indices
        self._remove_from_indices(template)
        
        # Remove template
        del self.templates[template_id]
        
        logger.info(f"Unregistered template: {template_id}")
        return True
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def has_template(self, template_id: str) -> bool:
        """Check if template exists."""
        return template_id in self.templates
    
    def list_templates(self, filter_by: Optional[Dict[str, Any]] = None) -> List[PromptTemplate]:
        """List templates with optional filtering."""
        templates = list(self.templates.values())
        
        if not filter_by:
            return templates
        
        filtered_templates = []
        for template in templates:
            if self._matches_filter(template, filter_by):
                filtered_templates.append(template)
        
        return filtered_templates
    
    def search_templates(self, query: str) -> List[PromptTemplate]:
        """Search templates by query string."""
        self.search_count += 1
        
        if not query.strip():
            return list(self.templates.values())
        
        query_lower = query.lower()
        matching_templates = []
        
        for template in self.templates.values():
            # Search in template ID, name, description
            if (query_lower in template.template_id.lower() or
                query_lower in template.name.lower() or
                (template.metadata.description and query_lower in template.metadata.description.lower())):
                matching_templates.append(template)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in template.metadata.tags):
                matching_templates.append(template)
                continue
            
            # Search in use case
            if query_lower in template.metadata.use_case.lower():
                matching_templates.append(template)
                continue
            
            # Search in domain
            if query_lower in template.metadata.domain.lower():
                matching_templates.append(template)
                continue
            
            # Search in template content (partial match)
            if query_lower in template.template.lower():
                matching_templates.append(template)
                continue
        
        logger.debug(f"Search '{query}' returned {len(matching_templates)} results")
        return matching_templates
    
    def find_templates_by_type(self, template_type: TemplateType) -> List[PromptTemplate]:
        """Find templates by type."""
        return [t for t in self.templates.values() if t.type == template_type]
    
    def find_templates_by_domain(self, domain: str) -> List[PromptTemplate]:
        """Find templates by domain."""
        return [t for t in self.templates.values() if t.metadata.domain == domain]
    
    def find_templates_by_complexity(self, complexity: TemplateComplexity) -> List[PromptTemplate]:
        """Find templates by complexity."""
        return [t for t in self.templates.values() if t.metadata.complexity == complexity]
    
    def find_templates_by_tag(self, tag: str) -> List[PromptTemplate]:
        """Find templates by tag."""
        return [t for t in self.templates.values() if tag in t.metadata.tags]
    
    def find_compatible_templates(self, context: Dict[str, Any]) -> List[PromptTemplate]:
        """Find templates compatible with given context."""
        compatible = []
        for template in self.templates.values():
            if self._is_template_compatible(template, context):
                compatible.append(template)
        return compatible
    
    def validate_all_templates(self) -> Dict[str, List[str]]:
        """Validate all registered templates."""
        validation_results = {}
        
        for template_id, template in self.templates.items():
            errors = self._validate_template(template)
            if errors:
                validation_results[template_id] = errors
        
        return validation_results
    
    def get_template_stats(self) -> Dict[str, Any]:
        """Get template registry statistics."""
        stats = {
            "total_templates": len(self.templates),
            "registration_count": self.registration_count,
            "search_count": self.search_count,
            "by_type": {},
            "by_domain": {},
            "by_complexity": {},
            "by_use_case": {}
        }
        
        # Count by type
        for template in self.templates.values():
            template_type = template.type.value
            stats["by_type"][template_type] = stats["by_type"].get(template_type, 0) + 1
            
            domain = template.metadata.domain
            stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
            
            complexity = template.metadata.complexity.value
            stats["by_complexity"][complexity] = stats["by_complexity"].get(complexity, 0) + 1
            
            use_case = template.metadata.use_case
            stats["by_use_case"][use_case] = stats["by_use_case"].get(use_case, 0) + 1
        
        return stats
    
    def export_templates(self, template_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Export templates as dictionary."""
        if template_ids:
            templates_to_export = {
                tid: self.templates[tid] for tid in template_ids if tid in self.templates
            }
        else:
            templates_to_export = self.templates
        
        return {
            tid: template.to_dict() 
            for tid, template in templates_to_export.items()
        }
    
    def import_templates(self, templates_data: Dict[str, Dict[str, Any]], overwrite: bool = False) -> Dict[str, Any]:
        """Import templates from dictionary."""
        results = {
            "imported": 0,
            "skipped": 0,
            "errors": 0,
            "error_details": {}
        }
        
        for template_id, template_data in templates_data.items():
            try:
                template = PromptTemplate.from_dict(template_data)
                
                if template_id in self.templates and not overwrite:
                    results["skipped"] += 1
                    continue
                
                self.register_template(template)
                results["imported"] += 1
                
            except Exception as e:
                results["errors"] += 1
                results["error_details"][template_id] = str(e)
        
        return results
    
    def _validate_template(self, template: PromptTemplate) -> List[str]:
        """Validate a template."""
        errors = []
        
        # Basic validation
        if not template.template_id:
            errors.append("Template ID is required")
        
        if not template.name:
            errors.append("Template name is required")
        
        if not template.template:
            errors.append("Template content is required")
        
        # Template content validation - use template engine for proper filter support
        try:
            from .template_engine import TemplateEngine
            engine = TemplateEngine()
            syntax_errors = engine.validate_template_syntax(template.template)
            errors.extend(syntax_errors)
        except Exception as e:
            errors.append(f"Template validation error: {str(e)}")
        
        # Variable validation
        if template.input_variables:
            # Check if required variables are used in template
            template_content = template.template.lower()
            for var in template.input_variables:
                # Check for variable usage: {{ var }}, {% for x in var %}, {% if var %}, etc.
                var_patterns = [
                    f"{{{{{var.lower()}}}}}",       # {{var}}
                    f"{{{{ {var.lower()} }}}}",     # {{ var }}
                    f"{{{{{var.lower()}|",          # {{var|
                    f"{{{{ {var.lower()}|",         # {{ var|
                    f"{{{{{var.lower()}.",          # {{var.
                    f"{{{{ {var.lower()}.",         # {{ var.
                    f"{{{{{var.lower()} ",          # {{var 
                    f"{{{{ {var.lower()} ",         # {{ var 
                    f"in {var.lower()}",            # {% for x in var %}
                    f"in {var.lower()} ",           # {% for x in var %}
                    f"{{% if {var.lower()}",        # {% if var %}
                    f"{{% if {var.lower()} ",       # {% if var %}
                ]
                if not any(pattern in template_content for pattern in var_patterns):
                    errors.append(f"Required variable '{var}' not found in template")
        
        return errors
    
    def _matches_filter(self, template: PromptTemplate, filter_criteria: Dict[str, Any]) -> bool:
        """Check if template matches filter criteria."""
        for key, value in filter_criteria.items():
            if key == "type":
                if template.type != value:
                    return False
            elif key == "domain":
                if template.metadata.domain != value:
                    return False
            elif key == "complexity":
                if template.metadata.complexity != value:
                    return False
            elif key == "use_case":
                if template.metadata.use_case != value:
                    return False
            elif key == "tags":
                if isinstance(value, list):
                    if not any(tag in template.metadata.tags for tag in value):
                        return False
                elif value not in template.metadata.tags:
                    return False
            elif key == "has_variables":
                if bool(template.input_variables) != bool(value):
                    return False
            elif key == "created_after":
                if template.metadata.created_at and template.metadata.created_at < value:
                    return False
            elif key == "created_before":
                if template.metadata.created_at and template.metadata.created_at > value:
                    return False
        
        return True
    
    def _is_template_compatible(self, template: PromptTemplate, context: Dict[str, Any]) -> bool:
        """Check if template is compatible with context."""
        return template.is_compatible_with(context)
    
    def _update_indices(self, template: PromptTemplate) -> None:
        """Update search indices for a template."""
        template_id = template.template_id
        
        # Update type index
        template_type = template.type.value
        if template_type not in self.template_index["by_type"]:
            self.template_index["by_type"][template_type] = set()
        self.template_index["by_type"][template_type].add(template_id)
        
        # Update domain index
        domain = template.metadata.domain
        if domain not in self.template_index["by_domain"]:
            self.template_index["by_domain"][domain] = set()
        self.template_index["by_domain"][domain].add(template_id)
        
        # Update complexity index
        complexity = template.metadata.complexity.value
        if complexity not in self.template_index["by_complexity"]:
            self.template_index["by_complexity"][complexity] = set()
        self.template_index["by_complexity"][complexity].add(template_id)
        
        # Update tags index
        for tag in template.metadata.tags:
            if tag not in self.template_index["by_tags"]:
                self.template_index["by_tags"][tag] = set()
            self.template_index["by_tags"][tag].add(template_id)
        
        # Update use case index
        use_case = template.metadata.use_case
        if use_case not in self.template_index["by_use_case"]:
            self.template_index["by_use_case"][use_case] = set()
        self.template_index["by_use_case"][use_case].add(template_id)
    
    def _remove_from_indices(self, template: PromptTemplate) -> None:
        """Remove template from search indices."""
        template_id = template.template_id
        
        # Remove from all indices
        for index_type, index_dict in self.template_index.items():
            for key, template_set in index_dict.items():
                template_set.discard(template_id)
                
                # Clean up empty sets
                if not template_set:
                    del index_dict[key]
    
    def clear(self) -> None:
        """Clear all templates and indices."""
        self.templates.clear()
        for index_dict in self.template_index.values():
            index_dict.clear()
        self.registration_count = 0
        self.search_count = 0