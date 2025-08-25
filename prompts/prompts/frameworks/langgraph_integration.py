"""LangGraph integration for advanced prompt orchestration workflows."""

import json
import uuid
from typing import Any, Dict, List, Optional, Type
from datetime import datetime

try:
    from langgraph import StateGraph, END
    from langgraph.graph import Graph
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
    from langchain_core.runnables import Runnable
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    Graph = None
    BaseMessage = None
    HumanMessage = None
    AIMessage = None
    Runnable = None

from ..models.context import PromptContext, ExecutionContext
from ..models.template import PromptTemplate


class PromptState(dict):
    """State object for LangGraph workflows with prompt-specific data."""
    
    def __init__(self, initial_data: Dict[str, Any] = None):
        super().__init__(initial_data or {})
        
        # Standard prompt fields
        self.setdefault("query", "")
        self.setdefault("context", {})
        self.setdefault("variables", {})
        self.setdefault("templates_used", [])
        self.setdefault("execution_path", [])
        self.setdefault("results", [])
        self.setdefault("errors", [])
        self.setdefault("metadata", {})
    
    def add_execution_step(self, step_name: str, result: Any = None, error: str = None):
        """Add an execution step to the workflow history."""
        step = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "error": error
        }
        self.get("execution_path", []).append(step)
    
    def add_template_usage(self, template_id: str, rendered_prompt: str = None):
        """Track template usage in the workflow."""
        usage = {
            "template_id": template_id,
            "timestamp": datetime.now().isoformat(),
            "rendered_prompt": rendered_prompt
        }
        self.get("templates_used", []).append(usage)


class LangGraphWorkflowManager:
    """
    Manager for LangGraph-based prompt orchestration workflows.
    
    Enables complex prompt workflows like:
    - Multi-step reasoning chains
    - Template selection pipelines
    - Conditional prompt routing
    - Self-correction loops
    - Parallel prompt execution
    """
    
    def __init__(self, prompt_system=None):
        """Initialize the workflow manager."""
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is not available. Install with: pip install langgraph"
            )
        
        self.prompt_system = prompt_system
        self.workflows: Dict[str, Graph] = {}
        self.workflow_configs: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in workflows
        self._register_builtin_workflows()
    
    def _register_builtin_workflows(self):
        """Register built-in workflow templates."""
        
        # Multi-step reasoning workflow
        self.register_workflow(
            "multi_step_reasoning",
            self._create_multi_step_reasoning_workflow(),
            {
                "name": "Multi-Step Reasoning",
                "description": "Break down complex queries into multiple reasoning steps",
                "parameters": ["query", "context", "max_steps"]
            }
        )
        
        # Template selection pipeline
        self.register_workflow(
            "template_selection_pipeline",
            self._create_template_selection_pipeline(),
            {
                "name": "Template Selection Pipeline",
                "description": "Advanced template selection with validation and fallbacks",
                "parameters": ["query", "context", "selection_criteria"]
            }
        )
        
        # Self-correction workflow
        self.register_workflow(
            "self_correction",
            self._create_self_correction_workflow(),
            {
                "name": "Self-Correction Loop",
                "description": "Generate, validate, and correct responses iteratively",
                "parameters": ["query", "context", "quality_threshold", "max_iterations"]
            }
        )
        
        # Parallel comparison workflow
        self.register_workflow(
            "parallel_comparison",
            self._create_parallel_comparison_workflow(),
            {
                "name": "Parallel Template Comparison",
                "description": "Execute multiple templates in parallel and compare results",
                "parameters": ["query", "context", "template_ids"]
            }
        )
    
    def register_workflow(self, workflow_id: str, workflow: Graph, config: Dict[str, Any]):
        """Register a new workflow."""
        self.workflows[workflow_id] = workflow
        self.workflow_configs[workflow_id] = config
    
    def execute_workflow(
        self,
        workflow_id: str,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a registered workflow."""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow '{workflow_id}' not found")
        
        workflow = self.workflows[workflow_id]
        
        # Create initial state
        state = PromptState(initial_state)
        state["workflow_id"] = workflow_id
        state["workflow_config"] = config or {}
        state["execution_id"] = str(uuid.uuid4())
        
        try:
            # Execute workflow
            result = workflow.invoke(state)
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "execution_id": state["execution_id"],
                "result": result,
                "execution_path": result.get("execution_path", []),
                "templates_used": result.get("templates_used", []),
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "workflow_id": workflow_id,
                "execution_id": state.get("execution_id"),
                "error": str(e),
                "execution_path": state.get("execution_path", [])
            }
    
    def _create_multi_step_reasoning_workflow(self) -> Graph:
        """Create a multi-step reasoning workflow."""
        
        def analyze_query(state: PromptState) -> PromptState:
            """Analyze the query and plan reasoning steps."""
            state.add_execution_step("analyze_query")
            
            query = state["query"]
            context = state.get("context", {})
            
            # Simple heuristic for determining reasoning steps
            # In a real implementation, this could use ML models
            steps = []
            
            if "compare" in query.lower() or "difference" in query.lower():
                steps = ["identify_items", "extract_features", "compare_features", "synthesize"]
            elif "analyze" in query.lower() or "explain" in query.lower():
                steps = ["understand_context", "identify_key_points", "analyze_relationships", "conclude"]
            elif "summarize" in query.lower():
                steps = ["extract_main_points", "organize_information", "create_summary"]
            else:
                steps = ["understand_query", "gather_information", "formulate_response"]
            
            state["reasoning_steps"] = steps
            state["current_step"] = 0
            state["step_results"] = []
            
            return state
        
        def execute_reasoning_step(state: PromptState) -> PromptState:
            """Execute the current reasoning step."""
            current_step = state["current_step"]
            steps = state["reasoning_steps"]
            
            if current_step >= len(steps):
                return state
            
            step_name = steps[current_step]
            state.add_execution_step(f"reasoning_step_{current_step}", step_name)
            
            # Create step-specific prompt
            step_query = self._create_step_query(state["query"], step_name, state)
            
            # Execute prompt if system is available
            if self.prompt_system:
                context = PromptContext(
                    query=step_query,
                    custom_attributes={
                        "reasoning_step": step_name,
                        "step_number": current_step,
                        "total_steps": len(steps)
                    }
                )
                
                exec_context = self.prompt_system.execute_prompt(step_query, context)
                
                step_result = {
                    "step": step_name,
                    "query": step_query,
                    "result": exec_context.rendered_prompt,
                    "template_used": exec_context.selected_template_id
                }
                
                state["step_results"].append(step_result)
                state.add_template_usage(exec_context.selected_template_id, exec_context.rendered_prompt)
            
            state["current_step"] += 1
            return state
        
        def check_completion(state: PromptState) -> str:
            """Check if all reasoning steps are complete."""
            current_step = state["current_step"]
            total_steps = len(state["reasoning_steps"])
            
            if current_step >= total_steps:
                return "synthesize_result"
            else:
                return "execute_reasoning_step"
        
        def synthesize_result(state: PromptState) -> PromptState:
            """Synthesize the final result from all reasoning steps."""
            state.add_execution_step("synthesize_result")
            
            # Combine all step results
            step_results = state["step_results"]
            
            synthesis_query = f"""
            Original Query: {state['query']}
            
            Reasoning Steps Completed:
            """
            
            for i, step_result in enumerate(step_results, 1):
                synthesis_query += f"\n{i}. {step_result['step']}: {step_result['result']}\n"
            
            synthesis_query += "\nPlease provide a comprehensive final answer based on this step-by-step reasoning:"
            
            # Execute synthesis if system is available
            if self.prompt_system:
                context = PromptContext(
                    query=synthesis_query,
                    custom_attributes={"workflow_step": "synthesis", "reasoning_complete": True}
                )
                
                exec_context = self.prompt_system.execute_prompt(synthesis_query, context)
                
                state["final_result"] = exec_context.rendered_prompt
                state.add_template_usage(exec_context.selected_template_id, exec_context.rendered_prompt)
            
            return state
        
        # Build the workflow graph
        workflow = StateGraph(PromptState)
        
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("execute_reasoning_step", execute_reasoning_step)
        workflow.add_node("synthesize_result", synthesize_result)
        
        workflow.set_entry_point("analyze_query")
        
        workflow.add_edge("analyze_query", "execute_reasoning_step")
        workflow.add_conditional_edges(
            "execute_reasoning_step",
            check_completion,
            {
                "execute_reasoning_step": "execute_reasoning_step",
                "synthesize_result": "synthesize_result"
            }
        )
        workflow.add_edge("synthesize_result", END)
        
        return workflow.compile()
    
    def _create_template_selection_pipeline(self) -> Graph:
        """Create an advanced template selection pipeline."""
        
        def analyze_selection_context(state: PromptState) -> PromptState:
            """Analyze context for template selection."""
            state.add_execution_step("analyze_selection_context")
            
            query = state["query"]
            context = state.get("context", {})
            
            # Analyze query characteristics
            analysis = {
                "query_length": len(query),
                "complexity_indicators": [],
                "domain_indicators": [],
                "intent_indicators": []
            }
            
            # Simple keyword analysis
            if any(word in query.lower() for word in ["analyze", "complex", "detailed"]):
                analysis["complexity_indicators"].append("high_complexity")
            
            if any(word in query.lower() for word in ["medical", "health", "clinical"]):
                analysis["domain_indicators"].append("medical")
            
            if any(word in query.lower() for word in ["compare", "contrast", "versus"]):
                analysis["intent_indicators"].append("comparison")
            
            state["selection_analysis"] = analysis
            return state
        
        def select_candidate_templates(state: PromptState) -> PromptState:
            """Select candidate templates based on analysis."""
            state.add_execution_step("select_candidate_templates")
            
            analysis = state["selection_analysis"]
            candidates = []
            
            # Simple template selection logic
            if "medical" in analysis["domain_indicators"]:
                candidates.append("medical_qa")
            
            if "comparison" in analysis["intent_indicators"]:
                candidates.append("comparative_analysis")
            
            if "high_complexity" in analysis["complexity_indicators"]:
                candidates.append("chain_of_thought")
            
            # Always include basic templates as fallbacks
            candidates.extend(["qa_detailed", "qa_basic"])
            
            # Remove duplicates while preserving order
            unique_candidates = []
            for candidate in candidates:
                if candidate not in unique_candidates:
                    unique_candidates.append(candidate)
            
            state["candidate_templates"] = unique_candidates
            return state
        
        def validate_templates(state: PromptState) -> PromptState:
            """Validate that candidate templates exist and work."""
            state.add_execution_step("validate_templates")
            
            candidates = state["candidate_templates"]
            validated = []
            
            if self.prompt_system:
                for template_id in candidates:
                    template = self.prompt_system.get_template(template_id)
                    if template:
                        # Basic validation
                        errors = self.prompt_system.validate_template(template)
                        if not errors:
                            validated.append(template_id)
            else:
                # If no prompt system, assume all are valid
                validated = candidates
            
            state["validated_templates"] = validated
            return state
        
        def select_final_template(state: PromptState) -> PromptState:
            """Select the final template to use."""
            state.add_execution_step("select_final_template")
            
            validated = state["validated_templates"]
            
            if validated:
                selected = validated[0]  # Use highest priority validated template
            else:
                selected = "qa_basic"  # Ultimate fallback
            
            state["selected_template"] = selected
            return state
        
        # Build the workflow graph
        workflow = StateGraph(PromptState)
        
        workflow.add_node("analyze_selection_context", analyze_selection_context)
        workflow.add_node("select_candidate_templates", select_candidate_templates)
        workflow.add_node("validate_templates", validate_templates)
        workflow.add_node("select_final_template", select_final_template)
        
        workflow.set_entry_point("analyze_selection_context")
        workflow.add_edge("analyze_selection_context", "select_candidate_templates")
        workflow.add_edge("select_candidate_templates", "validate_templates")
        workflow.add_edge("validate_templates", "select_final_template")
        workflow.add_edge("select_final_template", END)
        
        return workflow.compile()
    
    def _create_self_correction_workflow(self) -> Graph:
        """Create a self-correction workflow."""
        
        def generate_initial_response(state: PromptState) -> PromptState:
            """Generate initial response."""
            state.add_execution_step("generate_initial_response")
            
            if self.prompt_system:
                query = state["query"]
                context = PromptContext(query=query)
                
                exec_context = self.prompt_system.execute_prompt(query, context)
                
                state["current_response"] = exec_context.rendered_prompt
                state["iterations"] = 0
                state.add_template_usage(exec_context.selected_template_id, exec_context.rendered_prompt)
            
            return state
        
        def evaluate_response(state: PromptState) -> PromptState:
            """Evaluate the current response quality."""
            state.add_execution_step("evaluate_response")
            
            response = state["current_response"]
            quality_threshold = state.get("workflow_config", {}).get("quality_threshold", 0.7)
            
            # Simple quality evaluation (in real implementation, use ML models)
            quality_score = min(1.0, len(response) / 1000)  # Very basic heuristic
            
            state["quality_score"] = quality_score
            state["quality_threshold"] = quality_threshold
            
            return state
        
        def check_quality(state: PromptState) -> str:
            """Check if quality is sufficient or max iterations reached."""
            quality_score = state["quality_score"]
            quality_threshold = state["quality_threshold"]
            iterations = state["iterations"]
            max_iterations = state.get("workflow_config", {}).get("max_iterations", 3)
            
            if quality_score >= quality_threshold or iterations >= max_iterations:
                return "finalize_response"
            else:
                return "improve_response"
        
        def improve_response(state: PromptState) -> PromptState:
            """Improve the current response."""
            state.add_execution_step("improve_response")
            
            current_response = state["current_response"]
            original_query = state["query"]
            
            improvement_query = f"""
            Original Query: {original_query}
            
            Current Response: {current_response}
            
            Please improve this response by making it more accurate, comprehensive, and helpful:
            """
            
            if self.prompt_system:
                context = PromptContext(
                    query=improvement_query,
                    custom_attributes={"workflow_step": "improvement", "iteration": state["iterations"]}
                )
                
                exec_context = self.prompt_system.execute_prompt(improvement_query, context)
                
                state["current_response"] = exec_context.rendered_prompt
                state.add_template_usage(exec_context.selected_template_id, exec_context.rendered_prompt)
            
            state["iterations"] += 1
            return state
        
        def finalize_response(state: PromptState) -> PromptState:
            """Finalize the response."""
            state.add_execution_step("finalize_response")
            
            state["final_response"] = state["current_response"]
            state["final_quality_score"] = state["quality_score"]
            
            return state
        
        # Build the workflow graph
        workflow = StateGraph(PromptState)
        
        workflow.add_node("generate_initial_response", generate_initial_response)
        workflow.add_node("evaluate_response", evaluate_response)
        workflow.add_node("improve_response", improve_response)
        workflow.add_node("finalize_response", finalize_response)
        
        workflow.set_entry_point("generate_initial_response")
        workflow.add_edge("generate_initial_response", "evaluate_response")
        workflow.add_conditional_edges(
            "evaluate_response",
            check_quality,
            {
                "improve_response": "improve_response",
                "finalize_response": "finalize_response"
            }
        )
        workflow.add_edge("improve_response", "evaluate_response")
        workflow.add_edge("finalize_response", END)
        
        return workflow.compile()
    
    def _create_parallel_comparison_workflow(self) -> Graph:
        """Create a parallel template comparison workflow."""
        
        def prepare_parallel_execution(state: PromptState) -> PromptState:
            """Prepare for parallel template execution."""
            state.add_execution_step("prepare_parallel_execution")
            
            template_ids = state.get("workflow_config", {}).get("template_ids", ["qa_basic", "qa_detailed"])
            state["template_ids"] = template_ids
            state["parallel_results"] = {}
            
            return state
        
        def execute_templates_parallel(state: PromptState) -> PromptState:
            """Execute multiple templates in parallel (simulated)."""
            state.add_execution_step("execute_templates_parallel")
            
            query = state["query"]
            template_ids = state["template_ids"]
            
            if self.prompt_system:
                for template_id in template_ids:
                    context = PromptContext(
                        query=query,
                        custom_attributes={"parallel_execution": True, "template_id": template_id}
                    )
                    
                    exec_context = self.prompt_system.execute_prompt(
                        query, 
                        context, 
                        template_override=template_id
                    )
                    
                    state["parallel_results"][template_id] = {
                        "response": exec_context.rendered_prompt,
                        "execution_time": exec_context.total_duration_ms,
                        "template_used": exec_context.selected_template_id,
                        "errors": exec_context.errors
                    }
                    
                    state.add_template_usage(template_id, exec_context.rendered_prompt)
            
            return state
        
        def compare_results(state: PromptState) -> PromptState:
            """Compare the parallel execution results."""
            state.add_execution_step("compare_results")
            
            results = state["parallel_results"]
            
            comparison = {
                "total_templates": len(results),
                "successful_executions": sum(1 for r in results.values() if not r["errors"]),
                "avg_execution_time": sum(r["execution_time"] or 0 for r in results.values()) / len(results),
                "response_lengths": {tid: len(r["response"]) for tid, r in results.items()},
                "best_template": None
            }
            
            # Simple best template selection (longest response wins)
            if results:
                best_template = max(results.keys(), key=lambda t: len(results[t]["response"]))
                comparison["best_template"] = best_template
                state["recommended_response"] = results[best_template]["response"]
            
            state["comparison_results"] = comparison
            return state
        
        # Build the workflow graph
        workflow = StateGraph(PromptState)
        
        workflow.add_node("prepare_parallel_execution", prepare_parallel_execution)
        workflow.add_node("execute_templates_parallel", execute_templates_parallel)
        workflow.add_node("compare_results", compare_results)
        
        workflow.set_entry_point("prepare_parallel_execution")
        workflow.add_edge("prepare_parallel_execution", "execute_templates_parallel")
        workflow.add_edge("execute_templates_parallel", "compare_results")
        workflow.add_edge("compare_results", END)
        
        return workflow.compile()
    
    def _create_step_query(self, original_query: str, step_name: str, state: PromptState) -> str:
        """Create a step-specific query for reasoning workflows."""
        
        step_templates = {
            "understand_context": f"First, let's understand the context for this query: {original_query}",
            "identify_key_points": f"What are the key points to consider for: {original_query}",
            "analyze_relationships": f"How do the key concepts relate to each other in: {original_query}",
            "conclude": f"Based on the analysis, what is the conclusion for: {original_query}",
            "identify_items": f"What items or concepts need to be compared in: {original_query}",
            "extract_features": f"What are the key features or characteristics to compare for: {original_query}",
            "compare_features": f"How do these features compare for: {original_query}",
            "synthesize": f"Synthesize the comparison results for: {original_query}",
            "extract_main_points": f"What are the main points from the context for: {original_query}",
            "organize_information": f"How should we organize this information for: {original_query}",
            "create_summary": f"Create a summary based on the organized information for: {original_query}",
            "understand_query": f"Let's break down what is being asked in: {original_query}",
            "gather_information": f"What information do we need to answer: {original_query}",
            "formulate_response": f"Based on the gathered information, formulate a response to: {original_query}"
        }
        
        return step_templates.get(step_name, f"Process step '{step_name}' for query: {original_query}")
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflows."""
        workflows = []
        
        for workflow_id, config in self.workflow_configs.items():
            workflows.append({
                "workflow_id": workflow_id,
                "name": config.get("name", workflow_id),
                "description": config.get("description", ""),
                "parameters": config.get("parameters", [])
            })
        
        return workflows
    
    def get_workflow_config(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific workflow."""
        return self.workflow_configs.get(workflow_id)
    
    def validate_workflow_inputs(self, workflow_id: str, inputs: Dict[str, Any]) -> List[str]:
        """Validate inputs for a workflow."""
        errors = []
        
        config = self.workflow_configs.get(workflow_id)
        if not config:
            errors.append(f"Workflow '{workflow_id}' not found")
            return errors
        
        required_params = config.get("parameters", [])
        
        for param in required_params:
            if param not in inputs:
                errors.append(f"Missing required parameter: {param}")
        
        return errors