"""Service for resolving chat parameter defaults from model config."""

from dataclasses import dataclass

from config.datamodel import ChatDefaults

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger()


@dataclass
class ResolvedChatParams:
    """Resolved chat parameters after merging model defaults with request."""

    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    rag_enabled: bool | None
    database: str | None
    rag_retrieval_strategy: str | None
    rag_top_k: int | None
    rag_score_threshold: float | None
    rag_queries: list[str] | None
    think: bool | None
    thinking_budget: int | None
    n_ctx: int | None


class ChatDefaultsService:
    """Service for resolving chat parameter defaults."""

    @classmethod
    def resolve(
        cls,
        model_defaults: ChatDefaults | None,
        request_params: dict,
    ) -> ResolvedChatParams:
        """Merge model defaults with request parameters.

        Priority: request_params > model_defaults > None

        Args:
            model_defaults: ChatDefaults from model config (optional)
            request_params: Dict of parameters from the request

        Returns:
            ResolvedChatParams with merged values
        """
        defaults = model_defaults.model_dump() if model_defaults else {}

        def get(key: str):
            # Request value wins if explicitly provided (not None)
            request_val = request_params.get(key)
            if request_val is not None:
                return request_val
            return defaults.get(key)

        return ResolvedChatParams(
            temperature=get("temperature"),
            top_p=get("top_p"),
            max_tokens=get("max_tokens"),
            rag_enabled=get("rag_enabled"),
            database=get("database"),
            rag_retrieval_strategy=get("rag_retrieval_strategy"),
            rag_top_k=get("rag_top_k"),
            rag_score_threshold=get("rag_score_threshold"),
            rag_queries=get("rag_queries"),
            think=get("think"),
            thinking_budget=get("thinking_budget"),
            n_ctx=get("n_ctx"),
        )

    @classmethod
    def to_request_dict(cls, params: ResolvedChatParams) -> dict:
        """Convert ResolvedChatParams to a dict for passing to downstream functions.

        Only includes non-None values.
        """
        result = {}
        for field in [
            "temperature",
            "top_p",
            "max_tokens",
            "rag_enabled",
            "database",
            "rag_retrieval_strategy",
            "rag_top_k",
            "rag_score_threshold",
            "rag_queries",
            "think",
            "thinking_budget",
            "n_ctx",
        ]:
            val = getattr(params, field)
            if val is not None:
                result[field] = val
        return result
