"""Template loader utility for loading templates from individual files."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..models.template import PromptTemplate


logger = logging.getLogger(__name__)


class TemplateLoader:
    """
    Utility class for loading prompt templates from individual files
    organized in a directory structure.
    """
    
    def __init__(self, templates_dir: str = "templates"):
        """Initialize the template loader."""
        self.templates_dir = Path(templates_dir)
        self.loaded_templates: Dict[str, PromptTemplate] = {}
        self.load_errors: List[str] = []
    
    def load_all_templates(self) -> Dict[str, PromptTemplate]:
        """
        Load all templates from the templates directory structure.
        
        Returns:
            Dictionary mapping template IDs to PromptTemplate objects
        """
        self.loaded_templates.clear()
        self.load_errors.clear()
        
        if not self.templates_dir.exists():
            error_msg = f"Templates directory not found: {self.templates_dir}"
            logger.error(error_msg)
            self.load_errors.append(error_msg)
            return {}
        
        # Load templates from each category directory
        category_dirs = [
            "basic",
            "chat", 
            "few_shot",
            "advanced",
            "domain_specific",
            "agentic"
        ]
        
        for category in category_dirs:
            category_path = self.templates_dir / category
            if category_path.exists() and category_path.is_dir():
                self._load_category_templates(category_path, category)
        
        logger.info(f"Loaded {len(self.loaded_templates)} templates with {len(self.load_errors)} errors")
        return self.loaded_templates.copy()
    
    def _load_category_templates(self, category_path: Path, category: str) -> None:
        """Load all templates from a specific category directory."""
        template_files = list(category_path.glob("*.json")) + list(category_path.glob("*.yaml")) + list(category_path.glob("*.yml"))
        
        for template_file in template_files:
            if template_file.name.startswith('.') or template_file.name == 'README.md':
                continue
                
            try:
                template = self._load_template_file(template_file)
                if template:
                    # Validate template type matches directory
                    if template.type.value != category:
                        logger.warning(
                            f"Template {template.template_id} type '{template.type.value}' "
                            f"doesn't match directory '{category}'"
                        )
                    
                    self.loaded_templates[template.template_id] = template
                    logger.debug(f"Loaded template: {template.template_id}")
                    
            except Exception as e:
                error_msg = f"Failed to load template from {template_file}: {str(e)}"
                logger.error(error_msg)
                self.load_errors.append(error_msg)
    
    def _load_template_file(self, file_path: Path) -> Optional[PromptTemplate]:
        """Load a single template file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Validate required fields
            if not data.get('template_id'):
                raise ValueError("Template missing required field: template_id")
            
            if not data.get('template'):
                raise ValueError("Template missing required field: template")
            
            # Create template object
            template = PromptTemplate(**data)
            return template
            
        except Exception as e:
            logger.error(f"Error loading template from {file_path}: {str(e)}")
            raise
    
    def load_template_by_id(self, template_id: str) -> Optional[PromptTemplate]:
        """Load a specific template by ID."""
        # First check if already loaded
        if template_id in self.loaded_templates:
            return self.loaded_templates[template_id]
        
        # Search for template file
        template_files = list(self.templates_dir.rglob("*.json")) + list(self.templates_dir.rglob("*.yaml"))
        
        for template_file in template_files:
            try:
                template = self._load_template_file(template_file)
                if template and template.template_id == template_id:
                    self.loaded_templates[template_id] = template
                    return template
            except Exception:
                continue
        
        return None
    
    def get_templates_by_category(self, category: str) -> Dict[str, PromptTemplate]:
        """Get all templates for a specific category."""
        return {
            tid: template for tid, template in self.loaded_templates.items()
            if template.type.value == category
        }
    
    def get_templates_by_domain(self, domain: str) -> Dict[str, PromptTemplate]:
        """Get all templates for a specific domain."""
        return {
            tid: template for tid, template in self.loaded_templates.items()
            if template.metadata.domain == domain
        }
    
    def validate_all_templates(self) -> Dict[str, List[str]]:
        """Validate all loaded templates."""
        validation_results = {}
        
        for template_id, template in self.loaded_templates.items():
            errors = self._validate_template(template)
            if errors:
                validation_results[template_id] = errors
        
        return validation_results
    
    def _validate_template(self, template: PromptTemplate) -> List[str]:
        """Validate a single template."""
        errors = []
        
        try:
            # Basic validation
            if not template.template_id:
                errors.append("Missing template_id")
            
            if not template.name:
                errors.append("Missing name")
            
            if not template.template:
                errors.append("Missing template content")
            
            # Jinja2 syntax validation
            from jinja2 import Environment, TemplateSyntaxError
            try:
                env = Environment()
                env.from_string(template.template)
            except TemplateSyntaxError as e:
                errors.append(f"Jinja2 syntax error: {str(e)}")
            
            # Variable validation
            if template.input_variables:
                template_content = template.template.lower()
                for var in template.input_variables:
                    if f"{{{{{var}}}}}" not in template_content and f"{{{{{var}|" not in template_content:
                        errors.append(f"Required variable '{var}' not found in template")
            
            # Metadata validation
            if not template.metadata.use_case:
                errors.append("Missing use_case in metadata")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def get_load_summary(self) -> Dict[str, any]:
        """Get a summary of the loading process."""
        summary = {
            "total_templates": len(self.loaded_templates),
            "load_errors": len(self.load_errors),
            "error_details": self.load_errors.copy(),
            "templates_by_category": {},
            "templates_by_domain": {}
        }
        
        # Group by category
        for template in self.loaded_templates.values():
            category = template.type.value
            if category not in summary["templates_by_category"]:
                summary["templates_by_category"][category] = 0
            summary["templates_by_category"][category] += 1
            
            # Group by domain
            domain = template.metadata.domain
            if domain not in summary["templates_by_domain"]:
                summary["templates_by_domain"][domain] = 0
            summary["templates_by_domain"][domain] += 1
        
        return summary
    
    def export_templates_to_config(self, output_file: str = None) -> Dict[str, any]:
        """Export loaded templates to a configuration format."""
        config_templates = {}
        
        for template_id, template in self.loaded_templates.items():
            config_templates[template_id] = template.dict()
        
        config_data = {
            "templates": config_templates,
            "metadata": {
                "generated_by": "TemplateLoader",
                "total_templates": len(config_templates),
                "categories": list(set(t.type.value for t in self.loaded_templates.values())),
                "domains": list(set(t.metadata.domain for t in self.loaded_templates.values()))
            }
        }
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                else:
                    json.dump(config_data, f, indent=2, default=str)
        
        return config_data
    
    def create_template_file(
        self, 
        template: PromptTemplate, 
        category: str = None,
        overwrite: bool = False
    ) -> str:
        """Create a new template file in the appropriate directory."""
        # Determine category
        if not category:
            category = template.type.value
        
        # Create category directory if needed
        category_path = self.templates_dir / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{template.template_id}.yaml"
        file_path = category_path / filename
        
        # Check if file exists
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"Template file already exists: {file_path}")
        
        # Write template file
        template_data = template.dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(template_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created template file: {file_path}")
        return str(file_path)