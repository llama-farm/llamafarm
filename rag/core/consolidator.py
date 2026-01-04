"""Consolidator - Memory synthesis agent for the Embedded Trinity Memory System.

The Consolidator is the "hippocampus" of the memory system. It:
1. Reads raw data from WorkingMemory
2. Synthesizes facts using LLM or rule-based extraction
3. Creates graph nodes from extracted facts
4. Creates vector embeddings from summaries (future)
5. Prunes raw data after consolidation

This enables long-term memory formation from short-term buffers.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class Consolidator:
    """Memory synthesis agent that consolidates working memory into long-term storage.

    Features:
    - Read pending records from WorkingMemory
    - Extract facts using LLM or rule-based patterns
    - Create graph nodes/edges from facts
    - Prune processed raw data
    - Configurable thresholds and retention policies

    Configuration:
        buffer_threshold: Minimum records before consolidation runs
        consolidation_interval: Seconds between consolidation cycles
        retention_days: Days to retain raw data before pruning
    """

    def __init__(
        self,
        memory_store: Any,
        config: dict[str, Any] | None = None,
        llm_client: Any = None,
    ):
        """Initialize the Consolidator.

        Args:
            memory_store: MemoryStore instance to consolidate
            config: Configuration dictionary
            llm_client: Optional LLM client for synthesis
        """
        self.memory_store = memory_store
        self.llm_client = llm_client

        config = config or {}
        self.buffer_threshold = config.get("buffer_threshold", 10)
        self.consolidation_interval = config.get("consolidation_interval", 300)  # 5 min
        self.retention_days = config.get("retention_days", 7)

        self._last_consolidation = None

        logger.info(
            f"Consolidator initialized: threshold={self.buffer_threshold}, "
            f"interval={self.consolidation_interval}s, retention={self.retention_days}d"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Read Operations
    # ─────────────────────────────────────────────────────────────────────

    def get_pending_records(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get pending records from working memory.

        Args:
            limit: Maximum records to retrieve

        Returns:
            List of record dictionaries
        """
        return self.memory_store.working_memory.get_recent(limit=limit)

    # ─────────────────────────────────────────────────────────────────────
    # Synthesis Operations
    # ─────────────────────────────────────────────────────────────────────

    def synthesize(
        self,
        records: list[dict[str, Any]],
        use_llm: bool = True,
    ) -> dict[str, Any]:
        """Synthesize facts from records.

        Args:
            records: List of records to synthesize
            use_llm: Whether to use LLM (falls back to rule-based if unavailable)

        Returns:
            Dictionary with 'facts' and 'summary'
        """
        if not records:
            return {"facts": [], "summary": ""}

        if use_llm and self.llm_client:
            try:
                return self._synthesize_with_llm(records)
            except Exception as e:
                logger.warning(f"LLM synthesis failed, falling back to rules: {e}")
                return self._synthesize_rule_based(records)
        else:
            return self._synthesize_rule_based(records)

    def _synthesize_with_llm(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Synthesize using LLM."""
        prompt = self._build_synthesis_prompt(records)
        response = self._call_llm(prompt)
        return response

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call the LLM for synthesis.

        This method should be overridden or mocked in tests.
        """
        if not self.llm_client:
            raise ValueError("No LLM client configured")

        # Placeholder for actual LLM call
        # In production, this would call the LlamaFarm API
        response = self.llm_client.generate(prompt)
        return self._parse_llm_response(response)

    def _build_synthesis_prompt(self, records: list[dict[str, Any]]) -> str:
        """Build the synthesis prompt for the LLM."""
        context = "\n".join(
            f"[{r.get('data_type', 'unknown')}] {r.get('content', '')}"
            for r in records[:50]  # Limit context size
        )

        return f"""Analyze the following records and extract key facts and relationships.

Records:
{context}

Extract:
1. Named entities (people, places, organizations)
2. Relationships between entities
3. Key events or actions
4. Temporal information

Return as JSON:
{{
  "facts": [
    {{"subject": "...", "predicate": "...", "object": "..."}}
  ],
  "summary": "Brief summary of the records"
}}
"""

    def _parse_llm_response(self, response: Any) -> dict[str, Any]:
        """Parse LLM response into structured format."""
        if isinstance(response, dict):
            return response

        # Try to parse as JSON
        try:
            if isinstance(response, str):
                # Extract JSON from response
                match = re.search(r"\{.*\}", response, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass

        return {"facts": [], "summary": str(response)[:200]}

    def _synthesize_rule_based(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Synthesize using rule-based extraction."""
        facts = []
        entities = set()

        for record in records:
            content = record.get("content", "")
            data_type = record.get("data_type", "unknown")

            # Extract named entities (simple pattern matching)
            # Pattern: "Name: message" or mentions of proper nouns
            name_match = re.match(r"^([A-Z][a-z]+(?:\.\s*[A-Z][a-z]+)?)\s*:", content)
            if name_match:
                speaker = name_match.group(1).strip()
                entities.add(speaker)
                facts.append(
                    {
                        "subject": speaker,
                        "predicate": "said",
                        "object": content[name_match.end() :].strip()[:100],
                    }
                )

            # Extract locations (simple patterns)
            loc_patterns = [
                r"at\s+([A-Z][a-zA-Z\s]+(?:Delta|Alpha|Bravo|Charlie)?)",
                r"to\s+sector\s+(\d+)",
                r"grid reference\s+([\d\.\-,\s]+)",
            ]
            for pattern in loc_patterns:
                loc_match = re.search(pattern, content)
                if loc_match:
                    location = loc_match.group(1).strip()
                    entities.add(location)
                    facts.append(
                        {
                            "subject": "event",
                            "predicate": "location",
                            "object": location,
                        }
                    )

            # Extract telemetry data
            if data_type == "telemetry":
                try:
                    data = json.loads(content) if isinstance(content, str) else content
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key in ("soldier", "sensor", "location"):
                                entities.add(str(value))
                            facts.append(
                                {
                                    "subject": data.get(
                                        "soldier", data.get("sensor", "sensor")
                                    ),
                                    "predicate": key,
                                    "object": str(value),
                                }
                            )
                except (json.JSONDecodeError, TypeError):
                    pass

        # Generate summary
        summary = self._generate_summary(records, entities)

        return {
            "facts": facts,
            "summary": summary,
            "entities": list(entities),
        }

    def _generate_summary(
        self,
        records: list[dict[str, Any]],
        entities: set,
    ) -> str:
        """Generate a summary of the records."""
        data_types = {}
        for r in records:
            dt = r.get("data_type", "unknown")
            data_types[dt] = data_types.get(dt, 0) + 1

        type_summary = ", ".join(f"{v} {k}" for k, v in data_types.items())
        entity_list = ", ".join(list(entities)[:5]) if entities else "none identified"

        return (
            f"Processed {len(records)} records ({type_summary}). "
            f"Key entities: {entity_list}."
        )

    # ─────────────────────────────────────────────────────────────────────
    # Fact Extraction
    # ─────────────────────────────────────────────────────────────────────

    def extract_facts(self, llm_response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract facts from LLM response.

        Args:
            llm_response: Response from LLM

        Returns:
            List of fact dictionaries
        """
        if not isinstance(llm_response, dict):
            return []

        facts = llm_response.get("facts", [])

        if not isinstance(facts, list):
            return []

        # Validate fact structure
        valid_facts = []
        for fact in facts:
            if isinstance(fact, dict):
                valid_facts.append(fact)

        return valid_facts

    # ─────────────────────────────────────────────────────────────────────
    # Graph Creation
    # ─────────────────────────────────────────────────────────────────────

    def create_graph_nodes(self, facts: list[dict[str, Any]]) -> int:
        """Create graph nodes and edges from extracted facts.

        Args:
            facts: List of fact dictionaries with subject/predicate/object

        Returns:
            Number of nodes/edges created
        """
        if not facts:
            return 0

        created = 0
        seen_nodes = set()

        for fact in facts:
            subject = fact.get("subject")
            predicate = fact.get("predicate")
            obj = fact.get("object")

            if not subject or not predicate:
                continue

            # Create subject node if not seen
            subject_id = f"entity:{subject.lower().replace(' ', '_')}"
            if subject_id not in seen_nodes:
                try:
                    self.memory_store.add(
                        data={"id": subject_id, "name": subject},
                        data_type="node",
                        metadata={"node_type": "entity", "source": "consolidator"},
                    )
                    seen_nodes.add(subject_id)
                    created += 1
                except Exception as e:
                    logger.warning(f"Failed to create node for {subject}: {e}")

            # Create object node and edge if object exists
            if obj and isinstance(obj, str) and len(obj) < 100:
                object_id = f"entity:{obj.lower().replace(' ', '_')[:50]}"
                if object_id not in seen_nodes:
                    try:
                        self.memory_store.add(
                            data={"id": object_id, "name": obj[:50]},
                            data_type="node",
                            metadata={"node_type": "entity", "source": "consolidator"},
                        )
                        seen_nodes.add(object_id)
                        created += 1
                    except Exception as e:
                        logger.warning(f"Failed to create node for {obj}: {e}")

                # Create edge
                try:
                    self.memory_store.add(
                        data={
                            "source": subject_id,
                            "edge_type": predicate,
                            "target": object_id,
                        },
                        data_type="edge",
                    )
                    created += 1
                except Exception as e:
                    logger.warning(f"Failed to create edge: {e}")

        logger.info(f"Created {created} graph nodes/edges from facts")
        return created

    # ─────────────────────────────────────────────────────────────────────
    # Pruning Operations
    # ─────────────────────────────────────────────────────────────────────

    def prune(self) -> int:
        """Prune expired records from working memory.

        Returns:
            Number of records pruned
        """
        return self.memory_store.prune_working_memory()

    # ─────────────────────────────────────────────────────────────────────
    # Consolidation Cycle
    # ─────────────────────────────────────────────────────────────────────

    def run_cycle(self, use_llm: bool = True) -> dict[str, Any]:
        """Run a full consolidation cycle.

        Args:
            use_llm: Whether to use LLM for synthesis

        Returns:
            Dictionary with cycle results
        """
        result = {
            "records_processed": 0,
            "facts_extracted": 0,
            "nodes_created": 0,
            "pruned": 0,
            "skipped": False,
        }

        # Get pending records
        pending = self.get_pending_records(limit=100)

        # Check threshold
        if len(pending) < self.buffer_threshold:
            result["skipped"] = True
            logger.info(
                f"Consolidation skipped: {len(pending)} records < {self.buffer_threshold} threshold"
            )
            return result

        # Synthesize
        synthesis = self.synthesize(pending, use_llm=use_llm)
        facts = synthesis.get("facts", [])

        result["records_processed"] = len(pending)
        result["facts_extracted"] = len(facts)

        # Create graph nodes
        if facts:
            result["nodes_created"] = self.create_graph_nodes(facts)

        # Prune expired
        result["pruned"] = self.prune()

        # Update timestamp
        self._last_consolidation = datetime.now()

        logger.info(
            f"Consolidation cycle complete: {result['records_processed']} records, "
            f"{result['facts_extracted']} facts, {result['nodes_created']} nodes"
        )

        return result

    def should_run(self) -> bool:
        """Check if consolidation should run based on interval.

        Returns:
            True if consolidation should run
        """
        if self._last_consolidation is None:
            return True

        elapsed = (datetime.now() - self._last_consolidation).total_seconds()
        return elapsed >= self.consolidation_interval
