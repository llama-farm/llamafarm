"""Context models for prompt execution."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PromptContext(BaseModel):
    """Context information for prompt selection and execution."""
    
    # Query information
    query: str = Field(..., description="The user's query/request")
    query_type: Optional[str] = Field(None, description="Type of query (qa, summary, etc.)")
    query_complexity: Optional[str] = Field(None, description="Query complexity level")
    query_intent: Optional[str] = Field(None, description="Detected query intent")
    
    # User context
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_role: Optional[str] = Field(None, description="User role/permission level")
    user_expertise: Optional[str] = Field(None, description="User expertise level")
    user_preferences: Dict[str, Any] = Field(
        default_factory=dict,
        description="User preferences and settings"
    )
    
    # Document context (from RAG system)
    documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved documents"
    )
    document_count: int = Field(default=0, description="Number of retrieved documents")
    document_types: List[str] = Field(
        default_factory=list,
        description="Types of retrieved documents"
    )
    document_domains: List[str] = Field(
        default_factory=list,
        description="Domains of retrieved documents"
    )
    avg_relevance_score: Optional[float] = Field(
        None,
        description="Average relevance score of retrieved documents"
    )
    
    # RAG system context
    retrieval_strategy: Optional[str] = Field(
        None,
        description="RAG retrieval strategy used"
    )
    retrieval_quality: Optional[str] = Field(
        None,
        description="Assessment of retrieval quality"
    )
    
    # System context
    environment: str = Field(default="production", description="Environment (dev/staging/prod)")
    timestamp: datetime = Field(default_factory=datetime.now, description="Request timestamp")
    processing_mode: Optional[str] = Field(None, description="Processing mode (batch/realtime)")
    
    # Domain and task context
    domain: str = Field(default="general", description="Domain/subject area")
    task_type: Optional[str] = Field(None, description="Type of task being performed")
    complexity_level: Optional[str] = Field(None, description="Overall complexity level")
    
    # Performance context
    max_response_time: Optional[int] = Field(
        None,
        description="Maximum allowed response time in milliseconds"
    )
    quality_threshold: Optional[float] = Field(
        None,
        description="Minimum quality threshold"
    )
    
    # A/B testing context
    experiment_id: Optional[str] = Field(None, description="A/B test experiment ID")
    variant_group: Optional[str] = Field(None, description="A/B test variant group")
    
    # Custom context
    custom_attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom context attributes"
    )
    
    def add_document(self, document: Dict[str, Any]) -> None:
        """Add a document to the context."""
        self.documents.append(document)
        self.document_count = len(self.documents)
        
        # Update document metadata
        if "type" in document:
            doc_type = document["type"]
            if doc_type not in self.document_types:
                self.document_types.append(doc_type)
        
        if "domain" in document:
            doc_domain = document["domain"]
            if doc_domain not in self.document_domains:
                self.document_domains.append(doc_domain)
    
    def set_user_info(self, user_id: str, role: str = None, expertise: str = None) -> None:
        """Set user information."""
        self.user_id = user_id
        if role:
            self.user_role = role
        if expertise:
            self.user_expertise = expertise
    
    def add_custom_attribute(self, key: str, value: Any) -> None:
        """Add a custom attribute to the context."""
        self.custom_attributes[key] = value
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute from context (supports nested keys)."""
        # Check direct attributes first
        if hasattr(self, key):
            return getattr(self, key)
        
        # Check custom attributes
        if key in self.custom_attributes:
            return self.custom_attributes[key]
        
        # Check nested attributes (dot notation)
        if "." in key:
            parts = key.split(".")
            value = self
            
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            return value
        
        return default
    
    def to_selection_context(self) -> Dict[str, Any]:
        """Convert to context dictionary for template selection."""
        return {
            "query_type": self.query_type,
            "query_complexity": self.query_complexity,
            "query_intent": self.query_intent,
            "user_role": self.user_role,
            "user_expertise": self.user_expertise,
            "document_count": self.document_count,
            "document_types": self.document_types,
            "document_domains": self.document_domains,
            "avg_relevance_score": self.avg_relevance_score,
            "retrieval_strategy": self.retrieval_strategy,
            "retrieval_quality": self.retrieval_quality,
            "domain": self.domain,
            "task_type": self.task_type,
            "complexity_level": self.complexity_level,
            "environment": self.environment,
            "processing_mode": self.processing_mode,
            **self.custom_attributes
        }


class ExecutionContext(BaseModel):
    """Context for prompt execution and result tracking."""
    
    # Execution metadata
    execution_id: str = Field(..., description="Unique execution identifier")
    prompt_context: PromptContext = Field(..., description="Original prompt context")
    
    # Template selection
    selected_template_id: Optional[str] = Field(None, description="Selected template ID")
    selected_strategy_id: Optional[str] = Field(None, description="Selected strategy ID")
    selection_reason: Optional[str] = Field(None, description="Reason for selection")
    fallback_used: bool = Field(default=False, description="Whether fallback was used")
    
    # Global prompts applied
    applied_global_prompts: List[str] = Field(
        default_factory=list,
        description="IDs of global prompts applied"
    )
    
    # Execution timing
    start_time: datetime = Field(default_factory=datetime.now, description="Execution start time")
    template_selection_time: Optional[datetime] = Field(None, description="Template selection completion time")
    rendering_time: Optional[datetime] = Field(None, description="Template rendering completion time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    
    # Input/Output
    input_variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Variables used for template rendering"
    )
    rendered_prompt: Optional[str] = Field(None, description="Final rendered prompt")
    raw_response: Optional[str] = Field(None, description="Raw model response")
    processed_response: Optional[str] = Field(None, description="Processed response")
    
    # Performance metrics
    total_duration_ms: Optional[int] = Field(None, description="Total execution time in ms")
    template_selection_duration_ms: Optional[int] = Field(None, description="Selection time in ms")
    rendering_duration_ms: Optional[int] = Field(None, description="Rendering time in ms")
    
    # Quality metrics
    response_quality_score: Optional[float] = Field(None, description="Response quality score")
    user_satisfaction_score: Optional[float] = Field(None, description="User satisfaction score")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="Execution errors")
    warnings: List[str] = Field(default_factory=list, description="Execution warnings")
    
    # LangGraph workflow context
    langgraph_execution_id: Optional[str] = Field(None, description="LangGraph execution ID")
    workflow_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="LangGraph workflow state"
    )
    
    def mark_template_selected(self, template_id: str, strategy_id: str, reason: str = None) -> None:
        """Mark template selection completion."""
        self.template_selection_time = datetime.now()
        self.selected_template_id = template_id
        self.selected_strategy_id = strategy_id
        self.selection_reason = reason
        
        # Calculate selection duration
        if self.start_time:
            duration = (self.template_selection_time - self.start_time).total_seconds() * 1000
            self.template_selection_duration_ms = int(duration)
    
    def mark_rendering_complete(self, rendered_prompt: str) -> None:
        """Mark template rendering completion."""
        self.rendering_time = datetime.now()
        self.rendered_prompt = rendered_prompt
        
        # Calculate rendering duration
        if self.template_selection_time:
            duration = (self.rendering_time - self.template_selection_time).total_seconds() * 1000
            self.rendering_duration_ms = int(duration)
    
    def mark_execution_complete(self, response: str = None) -> None:
        """Mark execution completion."""
        self.end_time = datetime.now()
        if response:
            self.raw_response = response
        
        # Calculate total duration
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds() * 1000
            self.total_duration_ms = int(duration)
    
    def add_error(self, error: str) -> None:
        """Add an error to the execution context."""
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the execution context."""
        self.warnings.append(warning)
    
    def has_errors(self) -> bool:
        """Check if execution has errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if execution has warnings."""
        return len(self.warnings) > 0
    
    def to_metrics_dict(self) -> Dict[str, Any]:
        """Convert to metrics dictionary for monitoring."""
        return {
            "execution_id": self.execution_id,
            "template_id": self.selected_template_id,
            "strategy_id": self.selected_strategy_id,
            "total_duration_ms": self.total_duration_ms,
            "selection_duration_ms": self.template_selection_duration_ms,
            "rendering_duration_ms": self.rendering_duration_ms,
            "fallback_used": self.fallback_used,
            "errors_count": len(self.errors),
            "warnings_count": len(self.warnings),
            "response_quality_score": self.response_quality_score,
            "user_satisfaction_score": self.user_satisfaction_score,
            "timestamp": self.start_time.isoformat(),
        }