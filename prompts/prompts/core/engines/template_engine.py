"""Template rendering engine with Jinja2 support."""

from typing import Any, Dict, List, Optional
from jinja2 import Environment, BaseLoader, Template, TemplateError
import re

from ..models.template import PromptTemplate


class PromptTemplateLoader(BaseLoader):
    """Custom Jinja2 loader for prompt templates."""
    
    def __init__(self, templates_dict: Dict[str, str]):
        self.templates = templates_dict
    
    def get_source(self, environment: Environment, template: str):
        if template not in self.templates:
            raise TemplateError(f"Template '{template}' not found")
        
        source = self.templates[template]
        return source, None, lambda: True


class TemplateEngine:
    """
    Template rendering engine that handles Jinja2 template rendering
    with additional features for prompt management.
    """
    
    def __init__(self):
        """Initialize the template engine."""
        # Jinja2 environment
        self.env = Environment(
            loader=PromptTemplateLoader({}),
            autoescape=False,  # Don't escape HTML for prompts
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        # Add custom filters
        self._add_custom_filters()
        
        # Template cache
        self._template_cache: Dict[str, Template] = {}
    
    def _add_custom_filters(self) -> None:
        """Add custom Jinja2 filters for prompt processing."""
        
        def truncate_words(text: str, max_words: int = 50, suffix: str = "...") -> str:
            """Truncate text to maximum number of words."""
            if not text:
                return text
            
            words = text.split()
            if len(words) <= max_words:
                return text
            
            return " ".join(words[:max_words]) + suffix
        
        def format_documents(docs: List[Dict[str, Any]], max_length: int = 1000) -> str:
            """Format a list of documents for prompt inclusion."""
            if not docs:
                return "No documents available."
            
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                content = doc.get("content", str(doc))
                title = doc.get("title", f"Document {i}")
                
                # Truncate content if too long
                if len(content) > max_length:
                    content = content[:max_length] + "..."
                
                formatted_docs.append(f"[Document {i}: {title}]\\n{content}")
            
            return "\\n\\n".join(formatted_docs)
        
        def format_list(items: List[Any], bullet: str = "•", max_items: int = 10) -> str:
            """Format a list of items as a bulleted list."""
            if not items:
                return "No items available."
            
            # Limit number of items
            limited_items = items[:max_items]
            if len(items) > max_items:
                limited_items.append(f"... and {len(items) - max_items} more")
            
            return "\\n".join([f"{bullet} {item}" for item in limited_items])
        
        def extract_keywords(text: str, max_keywords: int = 5) -> str:
            """Extract keywords from text (simple implementation)."""
            if not text:
                return ""
            
            # Simple keyword extraction (you might want to use NLP libraries)
            words = re.findall(r'\\b\\w{4,}\\b', text.lower())
            
            # Count word frequency
            word_count = {}
            for word in words:
                if word not in ['that', 'this', 'with', 'from', 'they', 'have', 'been']:
                    word_count[word] = word_count.get(word, 0) + 1
            
            # Get top keywords
            top_keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, count in top_keywords[:max_keywords]]
            
            return ", ".join(keywords)
        
        def sentiment_indicator(text: str) -> str:
            """Simple sentiment indicator (positive/neutral/negative)."""
            if not text:
                return "neutral"
            
            # Simple sentiment analysis based on keywords
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return "positive"
            elif negative_count > positive_count:
                return "negative"
            else:
                return "neutral"
        
        def format_time(timestamp: str) -> str:
            """Format timestamp for display."""
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                return timestamp
        
        def clean_template(text: str) -> str:
            """Clean up template formatting by normalizing whitespace and line breaks."""
            if not text:
                return text
            
            # Replace multiple consecutive newlines with double newlines
            import re
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            
            # Clean up extra spaces around newlines
            text = re.sub(r' +\n', '\n', text)
            text = re.sub(r'\n +', '\n', text)
            
            # Ensure consistent spacing around section headers
            text = re.sub(r'\n\*\*([^*]+)\*\*\n', r'\n\n**\1**\n', text)
            
            return text.strip()
        
        def format_sections(text: str, sections: list) -> str:
            """Format text with consistent section spacing."""
            if not text or not sections:
                return text
            
            formatted = text
            for section in sections:
                # Add proper spacing around section headers
                formatted = formatted.replace(f"**{section}:**", f"\n**{section}:**\n")
            
            # Clean up any double spacing
            formatted = re.sub(r'\n\n\n+', '\n\n', formatted)
            return formatted.strip()
        
        # Register filters
        self.env.filters['truncate_words'] = truncate_words
        self.env.filters['format_documents'] = format_documents
        self.env.filters['format_list'] = format_list
        self.env.filters['extract_keywords'] = extract_keywords
        self.env.filters['sentiment_indicator'] = sentiment_indicator
        self.env.filters['format_time'] = format_time
        self.env.filters['clean_template'] = clean_template
        self.env.filters['format_sections'] = format_sections
    
    def render_template(
        self,
        template: PromptTemplate,
        variables: Dict[str, Any],
        use_cache: bool = True
    ) -> str:
        """
        Render a prompt template with provided variables.
        
        Args:
            template: The prompt template to render
            variables: Variables to substitute in the template
            use_cache: Whether to use template cache
            
        Returns:
            Rendered prompt string
        """
        # Validate inputs
        template.validate_inputs(variables)
        
        # Get or create Jinja2 template
        cache_key = f"{template.template_id}_{hash(template.template)}"
        
        if use_cache and cache_key in self._template_cache:
            jinja_template = self._template_cache[cache_key]
        else:
            try:
                jinja_template = self.env.from_string(template.template)
                if use_cache:
                    self._template_cache[cache_key] = jinja_template
            except TemplateError as e:
                raise ValueError(f"Template syntax error in '{template.template_id}': {str(e)}")
        
        # Add template metadata to variables
        render_vars = variables.copy()
        render_vars.update({
            '_template_id': template.template_id,
            '_template_name': template.name,
            '_template_type': template.type.value,
        })
        
        # Render template
        try:
            rendered = jinja_template.render(**render_vars)
            
            # Post-process the rendered template
            rendered = self._post_process_rendered_template(rendered, template)
            
            return rendered
            
        except TemplateError as e:
            raise ValueError(f"Template rendering error in '{template.template_id}': {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error rendering '{template.template_id}': {str(e)}")
    
    def _post_process_rendered_template(self, rendered: str, template: PromptTemplate) -> str:
        """Post-process the rendered template."""
        # Remove excessive whitespace
        rendered = re.sub(r'\\n\\s*\\n\\s*\\n', '\\n\\n', rendered)
        rendered = rendered.strip()
        
        # Apply any post-processing steps defined in template
        for step in template.postprocessing_steps:
            if step == "remove_empty_lines":
                rendered = "\\n".join([line for line in rendered.split("\\n") if line.strip()])
            elif step == "normalize_whitespace":
                rendered = re.sub(r'\\s+', ' ', rendered)
            elif step == "trim_lines":
                lines = [line.strip() for line in rendered.split("\\n")]
                rendered = "\\n".join(lines)
        
        return rendered
    
    def validate_template_syntax(self, template_content: str) -> List[str]:
        """
        Validate Jinja2 template syntax.
        
        Args:
            template_content: Template content to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Try to parse the template
            self.env.from_string(template_content)
        except TemplateError as e:
            errors.append(f"Syntax error: {str(e)}")
        except Exception as e:
            errors.append(f"Unexpected error: {str(e)}")
        
        return errors
    
    def get_template_variables(self, template_content: str) -> List[str]:
        """
        Extract variables used in a template.
        
        Args:
            template_content: Template content to analyze
            
        Returns:
            List of variable names found in template
        """
        try:
            # Parse template and get AST
            ast = self.env.parse(template_content)
            
            # Extract variable names
            variables = set()
            for node in ast.find_all():
                if hasattr(node, 'name') and isinstance(node.name, str):
                    # Skip Jinja2 built-ins and our custom variables
                    if not node.name.startswith('_') and node.name not in ['loop', 'super']:
                        variables.add(node.name)
            
            return sorted(list(variables))
            
        except Exception:
            # Fallback: simple regex extraction
            import re
            pattern = r'{{\\s*([a-zA-Z_][a-zA-Z0-9_]*).*?}}'
            matches = re.findall(pattern, template_content)
            return sorted(list(set(matches)))
    
    def test_template(
        self,
        template_content: str,
        test_variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a template with provided variables.
        
        Args:
            template_content: Template content to test
            test_variables: Variables to use for testing
            
        Returns:
            Test results dictionary
        """
        result = {
            "success": False,
            "rendered_output": None,
            "errors": [],
            "warnings": [],
            "variables_used": [],
            "variables_missing": []
        }
        
        try:
            # Validate syntax
            syntax_errors = self.validate_template_syntax(template_content)
            if syntax_errors:
                result["errors"].extend(syntax_errors)
                return result
            
            # Get template variables
            template_vars = self.get_template_variables(template_content)
            result["variables_used"] = template_vars
            
            # Check for missing variables
            missing_vars = set(template_vars) - set(test_variables.keys())
            result["variables_missing"] = sorted(list(missing_vars))
            
            if missing_vars:
                result["warnings"].append(f"Missing variables: {', '.join(missing_vars)}")
            
            # Render template
            jinja_template = self.env.from_string(template_content)
            rendered = jinja_template.render(**test_variables)
            
            result["rendered_output"] = rendered
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._template_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get template cache statistics."""
        return {
            "cached_templates": len(self._template_cache),
            "cache_keys": list(self._template_cache.keys())
        }