"""Prompt configuration agent."""

from typing import Dict, Any, List
from .base_agent import BaseAgent


class PromptAgent(BaseAgent):
    """Agent specialized in prompt configurations."""
    
    def __init__(self):
        super().__init__("prompts")
    
    async def create_config(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create a prompt configuration using the LlamaFarm prompts system."""
        output_path = self.settings.generated_configs_dir / self.component_name / f"generated_{requirements.get('use_case', 'custom')}.yaml"
        
        # Use the LlamaFarm client to create the config
        result = await self.client.create_prompt_config(requirements, str(output_path))
        
        if result["success"]:
            return {
                "config": result["config"],
                "file_path": result["config_path"],
                "validation_output": result.get("validation_output", ""),
                "success": True
            }
        else:
            raise ValueError(f"Failed to create prompt config: {result.get('error')}")
    
    async def _configure_global_prompts(self, domain: str) -> Dict[str, Any]:
        """Configure global prompts based on domain."""
        base_prompts = {
            "system_context": {
                "prompt": "You are a helpful AI assistant.",
                "enabled": True,
                "apply_to": ["all"]
            },
            "safety_guidelines": {
                "prompt": "Always provide safe, helpful, and accurate responses.",
                "enabled": True,
                "apply_to": ["all"]
            }
        }
        
        if domain == "medical":
            base_prompts.update({
                "medical_disclaimer": {
                    "prompt": "This information is for educational purposes only and should not replace professional medical advice.",
                    "enabled": True,
                    "apply_to": ["medical_qa", "medical_analysis"]
                },
                "privacy_notice": {
                    "prompt": "Maintain patient confidentiality and follow HIPAA guidelines.",
                    "enabled": True,
                    "apply_to": ["all"]
                }
            })
        elif domain == "legal":
            base_prompts.update({
                "legal_disclaimer": {
                    "prompt": "This information is for general guidance only and does not constitute legal advice.",
                    "enabled": True,
                    "apply_to": ["legal_analysis", "contract_review"]
                },
                "confidentiality": {
                    "prompt": "Maintain attorney-client privilege and confidentiality.",
                    "enabled": True,
                    "apply_to": ["all"]
                }
            })
        elif domain == "code":
            base_prompts.update({
                "code_quality": {
                    "prompt": "Provide clean, well-documented, and secure code following best practices.",
                    "enabled": True,
                    "apply_to": ["code_generation", "code_review"]
                },
                "security_focus": {
                    "prompt": "Always consider security implications and avoid vulnerable patterns.",
                    "enabled": True,
                    "apply_to": ["code_generation"]
                }
            })
        
        return base_prompts
    
    async def _configure_templates(self, template_names: List[str], domain: str) -> Dict[str, Any]:
        """Configure templates based on names and domain."""
        templates = {}
        
        for template_name in template_names:
            if template_name == "qa_basic":
                templates["qa_basic"] = {
                    "template_id": "qa_basic",
                    "name": "Basic Question Answering",
                    "type": "basic",
                    "template": "Based on the following context:\n{{ context | format_documents }}\n\nQuestion: {{ query }}\nAnswer:",
                    "variables": ["context", "query"],
                    "metadata": {
                        "category": "basic",
                        "domain": domain,
                        "use_cases": ["simple_qa", "fact_retrieval"]
                    }
                }
            elif template_name == "qa_detailed":
                templates["qa_detailed"] = {
                    "template_id": "qa_detailed", 
                    "name": "Detailed Question Answering",
                    "type": "basic",
                    "template": "Context:\n{{ context | format_documents }}\n\nQuestion: {{ query }}\n\nProvide a detailed answer with reasoning:\n1. Key information from context\n2. Analysis and reasoning\n3. Final answer\n\nAnswer:",
                    "variables": ["context", "query"],
                    "metadata": {
                        "category": "basic",
                        "domain": domain,
                        "use_cases": ["detailed_analysis", "research"]
                    }
                }
            elif template_name == "chain_of_thought":
                templates["chain_of_thought"] = {
                    "template_id": "chain_of_thought",
                    "name": "Chain of Thought Reasoning",
                    "type": "advanced", 
                    "template": "Context: {{ context | format_documents }}\n\nQuestion: {{ query }}\n\nLet's think through this step by step:\n1. What information do we have?\n2. What is being asked?\n3. How can we reason through this?\n4. What is the conclusion?\n\nStep-by-step reasoning:",
                    "variables": ["context", "query"],
                    "metadata": {
                        "category": "advanced",
                        "domain": domain,
                        "use_cases": ["complex_reasoning", "problem_solving"]
                    }
                }
            elif template_name == "summarization":
                templates["summarization"] = {
                    "template_id": "summarization",
                    "name": "Document Summarization",
                    "type": "basic",
                    "template": "Please summarize the following document:\n\n{{ content }}\n\nProvide a concise summary highlighting the key points:\n\nSummary:",
                    "variables": ["content"],
                    "metadata": {
                        "category": "basic",
                        "domain": domain,
                        "use_cases": ["document_summary", "content_condensation"]
                    }
                }
        
        # Add domain-specific templates
        if domain == "medical":
            templates.update(await self._get_medical_templates())
        elif domain == "legal":
            templates.update(await self._get_legal_templates())
        elif domain == "code":
            templates.update(await self._get_code_templates())
        
        return templates
    
    async def _get_medical_templates(self) -> Dict[str, Any]:
        """Get medical domain-specific templates."""
        return {
            "medical_qa": {
                "template_id": "medical_qa",
                "name": "Medical Question Answering",
                "type": "domain_specific",
                "template": "Medical Context: {{ context | format_documents }}\n\nMedical Question: {{ query }}\n\nBased on the provided medical information, provide a professional response. Include:\n1. Relevant medical information\n2. Key considerations\n3. Recommendations for further consultation if needed\n\nResponse:",
                "variables": ["context", "query"],
                "metadata": {
                    "category": "domain_specific",
                    "domain": "medical",
                    "requires_disclaimer": True
                }
            }
        }
    
    async def _get_legal_templates(self) -> Dict[str, Any]:
        """Get legal domain-specific templates."""
        return {
            "legal_analysis": {
                "template_id": "legal_analysis",
                "name": "Legal Document Analysis",
                "type": "domain_specific",
                "template": "Legal Documents: {{ context | format_documents }}\n\nAnalysis Request: {{ query }}\n\nLegal Analysis:\n1. Relevant Legal Provisions\n2. Key Issues Identified\n3. Analysis and Interpretation\n4. Conclusions and Recommendations\n\nAnalysis:",
                "variables": ["context", "query"],
                "metadata": {
                    "category": "domain_specific",
                    "domain": "legal",
                    "requires_disclaimer": True
                }
            }
        }
    
    async def _get_code_templates(self) -> Dict[str, Any]:
        """Get code domain-specific templates."""
        return {
            "code_analysis": {
                "template_id": "code_analysis",
                "name": "Code Analysis and Review",
                "type": "domain_specific",
                "template": "Code to analyze:\n{{ code }}\n\nAnalysis request: {{ query }}\n\nCode Analysis:\n1. Code Structure and Design\n2. Potential Issues or Bugs\n3. Security Considerations\n4. Performance Implications\n5. Recommendations for Improvement\n\nAnalysis:",
                "variables": ["code", "query"],
                "metadata": {
                    "category": "domain_specific",
                    "domain": "code",
                    "focus_areas": ["security", "performance", "maintainability"]
                }
            }
        }
    
    async def _configure_strategies(self, use_case: str) -> Dict[str, Any]:
        """Configure template selection strategies."""
        strategies = {}
        
        if use_case == "basic":
            strategies["simple_strategy"] = {
                "type": "rule_based",
                "description": "Simple template selection based on query type",
                "rules": [
                    {
                        "condition": "query_type == 'question'",
                        "template": "qa_basic"
                    },
                    {
                        "condition": "query_type == 'summary'",
                        "template": "summarization"
                    }
                ],
                "default_template": "qa_basic"
            }
        elif use_case == "advanced":
            strategies["context_aware_strategy"] = {
                "type": "context_aware",
                "description": "Context-aware template selection",
                "factors": [
                    {"name": "query_complexity", "weight": 0.4},
                    {"name": "domain_match", "weight": 0.3},
                    {"name": "historical_performance", "weight": 0.3}
                ],
                "fallback_template": "qa_detailed"
            }
        elif use_case == "experimental":
            strategies["ml_strategy"] = {
                "type": "ml_driven",
                "description": "Machine learning-based template selection",
                "model_config": {
                    "model_type": "classification",
                    "features": ["query_embedding", "context_length", "domain_indicators"],
                    "training_data_path": "data/template_selection_training.json"
                },
                "confidence_threshold": 0.7,
                "fallback_strategy": "context_aware_strategy"
            }
        
        return strategies
    
    async def _select_default_strategy(self, use_case: str) -> str:
        """Select the default strategy based on use case."""
        if use_case == "basic":
            return "simple_strategy"
        elif use_case == "advanced":
            return "context_aware_strategy"
        elif use_case == "experimental":
            return "ml_strategy"
        else:
            return "simple_strategy"
    
    async def edit_config(self, config: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """Edit an existing prompt configuration."""
        result = self._merge_configs(config, changes)
        
        # Validate the merged configuration
        validation = await self.validate_config(result)
        if not validation["valid"]:
            raise ValueError(f"Invalid configuration after edit: {validation['errors']}")
        
        return result
    
    async def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a prompt configuration."""
        errors = []
        warnings = []
        
        # Check required fields
        if "templates" not in config:
            errors.append("Missing 'templates' section")
        else:
            templates = config["templates"]
            if not templates:
                errors.append("No templates configured")
            
            # Validate each template
            for template_id, template_config in templates.items():
                if "template" not in template_config:
                    errors.append(f"Template '{template_id}' missing 'template' field")
                
                # Check for required variables
                template_text = template_config.get("template", "")
                variables = template_config.get("variables", [])
                
                # Simple variable detection (looking for {{ variable }})
                import re
                found_vars = re.findall(r'\{\{\s*(\w+)', template_text)
                for var in found_vars:
                    if var not in variables:
                        warnings.append(f"Template '{template_id}' uses variable '{var}' not listed in variables")
        
        # Check strategies
        if "strategies" in config:
            strategies = config["strategies"]
            default_strategy = config.get("default_strategy")
            
            if default_strategy and default_strategy not in strategies:
                errors.append(f"Default strategy '{default_strategy}' not found in strategies")
            
            # Validate strategy configurations
            for strategy_name, strategy_config in strategies.items():
                if "type" not in strategy_config:
                    errors.append(f"Strategy '{strategy_name}' missing 'type' field")
        
        # Check global prompts
        if "global_prompts" in config:
            global_prompts = config["global_prompts"]
            for prompt_name, prompt_config in global_prompts.items():
                if "prompt" not in prompt_config:
                    errors.append(f"Global prompt '{prompt_name}' missing 'prompt' field")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": self._generate_suggestions(config)
        }
    
    def _generate_suggestions(self, config: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving the prompt configuration."""
        suggestions = []
        
        templates = config.get("templates", {})
        
        # Suggest adding domain-specific templates
        if len(templates) == 1 and "qa_basic" in templates:
            suggestions.append("Consider adding more specialized templates for better responses")
        
        # Suggest adding global prompts
        if not config.get("global_prompts"):
            suggestions.append("Add global prompts for consistent system behavior")
        
        # Suggest advanced strategies
        strategies = config.get("strategies", {})
        if all(s.get("type") == "rule_based" for s in strategies.values()):
            suggestions.append("Consider context-aware strategies for better template selection")
        
        # Domain-specific suggestions
        domain_templates = [t for t in templates.values() if t.get("type") == "domain_specific"]
        if not domain_templates:
            suggestions.append("Add domain-specific templates for specialized use cases")
        
        return suggestions