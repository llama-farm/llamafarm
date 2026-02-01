"""
Route learning and quality tracking for semantic routing.

Tracks:
- Success/failure rates per route
- Latency statistics
- Quality scores based on feedback

This enables the gradient table to make better routing decisions
over time by learning from actual usage patterns.
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# Configuration
FEEDBACK_WINDOW_SEC = 3600  # 1 hour window for recent feedback
MAX_FEEDBACK_ENTRIES = 10000
QUALITY_DECAY = 0.95  # Exponential decay for old feedback


@dataclass
class RouteFeedback:
    """A single feedback entry for a route."""
    timestamp: float
    route_id: str  # capability_id or composite key
    success: bool
    latency_ms: float
    error_type: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class RouteQuality:
    """Aggregated quality metrics for a route."""
    route_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    quality_score: float = 1.0  # 0.0 - 1.0, higher is better
    
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def to_dict(self) -> dict:
        return {
            "route_id": self.route_id,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate(),
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "last_success": self.last_success,
            "last_failure": self.last_failure,
            "quality_score": self.quality_score
        }


class RouteLearner:
    """
    Learns from route feedback to improve routing decisions.
    
    Tracks per-route quality metrics and provides adjusted scores
    for the gradient table.
    
    Usage:
        learner = RouteLearner()
        
        # Record feedback after each routing decision
        learner.record_success(route_id, latency_ms=50)
        learner.record_failure(route_id, error_type="timeout")
        
        # Get quality-adjusted score
        adjusted = learner.adjust_score(route_id, base_score=0.9)
        
        # Persist state
        learner.save("/path/to/learner.json")
    """
    
    def __init__(
        self,
        feedback_window: float = FEEDBACK_WINDOW_SEC,
        max_feedback: int = MAX_FEEDBACK_ENTRIES,
        storage_path: Optional[Path] = None
    ):
        self.feedback_window = feedback_window
        self.max_feedback = max_feedback
        self.storage_path = storage_path
        
        self._feedback: List[RouteFeedback] = []
        self._quality: Dict[str, RouteQuality] = {}
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        if storage_path and storage_path.exists():
            self._load()
    
    def record_success(
        self,
        route_id: str,
        latency_ms: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """Record a successful routing decision."""
        feedback = RouteFeedback(
            timestamp=time.time(),
            route_id=route_id,
            success=True,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )
        self._add_feedback(feedback)
    
    def record_failure(
        self,
        route_id: str,
        latency_ms: float = 0.0,
        error_type: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Record a failed routing decision."""
        feedback = RouteFeedback(
            timestamp=time.time(),
            route_id=route_id,
            success=False,
            latency_ms=latency_ms,
            error_type=error_type,
            metadata=metadata or {}
        )
        self._add_feedback(feedback)
    
    def _add_feedback(self, feedback: RouteFeedback) -> None:
        """Add feedback and update quality metrics."""
        with self._lock:
            self._feedback.append(feedback)
            self._trim_feedback()
            
            # Update quality metrics
            quality = self._quality.get(feedback.route_id)
            if quality is None:
                quality = RouteQuality(route_id=feedback.route_id)
                self._quality[feedback.route_id] = quality
            
            quality.total_requests += 1
            
            if feedback.success:
                quality.successful_requests += 1
                quality.last_success = feedback.timestamp
                self._latencies[feedback.route_id].append(feedback.latency_ms)
            else:
                quality.failed_requests += 1
                quality.last_failure = feedback.timestamp
            
            # Update latency stats
            latencies = self._latencies[feedback.route_id]
            if latencies:
                quality.avg_latency_ms = np.mean(latencies[-100:])
                quality.p95_latency_ms = np.percentile(latencies[-100:], 95)
            
            # Update quality score
            quality.quality_score = self._compute_quality_score(quality)
    
    def _compute_quality_score(self, quality: RouteQuality) -> float:
        """
        Compute a quality score from 0.0 to 1.0.
        
        Considers:
        - Success rate (60% weight)
        - Latency (20% weight)
        - Recency of failures (20% weight)
        """
        # Success rate component
        success_component = quality.success_rate() * 0.6
        
        # Latency component (lower is better, normalized)
        # Assume 1000ms as "bad" threshold
        latency_normalized = max(0, 1 - (quality.avg_latency_ms / 1000))
        latency_component = latency_normalized * 0.2
        
        # Recency of failure component
        now = time.time()
        if quality.last_failure == 0:
            recency_component = 0.2  # No failures = full score
        else:
            time_since_failure = now - quality.last_failure
            # Failures decay over 1 hour
            recency_component = min(0.2, (time_since_failure / 3600) * 0.2)
        
        return min(1.0, success_component + latency_component + recency_component)
    
    def _trim_feedback(self) -> None:
        """Remove old feedback entries."""
        if len(self._feedback) > self.max_feedback:
            self._feedback = self._feedback[-self.max_feedback:]
        
        # Also trim old latencies
        for route_id in list(self._latencies.keys()):
            latencies = self._latencies[route_id]
            if len(latencies) > 100:
                self._latencies[route_id] = latencies[-100:]
    
    def adjust_score(
        self,
        route_id: str,
        base_score: float,
        weight: float = 0.3
    ) -> float:
        """
        Adjust a base similarity score based on learned quality.
        
        Args:
            route_id: The route to adjust
            base_score: Original similarity score
            weight: How much to weight quality (0.0-1.0)
            
        Returns:
            Adjusted score
        """
        with self._lock:
            quality = self._quality.get(route_id)
            if quality is None:
                return base_score  # No data, use base score
            
            # Blend base score with quality score
            return (1 - weight) * base_score + weight * quality.quality_score * base_score
    
    def get_quality(self, route_id: str) -> Optional[RouteQuality]:
        """Get quality metrics for a route."""
        with self._lock:
            return self._quality.get(route_id)
    
    def get_all_quality(self) -> Dict[str, RouteQuality]:
        """Get quality metrics for all routes."""
        with self._lock:
            return dict(self._quality)
    
    def get_best_routes(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get the highest quality routes."""
        with self._lock:
            sorted_routes = sorted(
                self._quality.items(),
                key=lambda x: x[1].quality_score,
                reverse=True
            )
            return [
                (route_id, q.quality_score)
                for route_id, q in sorted_routes[:top_k]
            ]
    
    def get_degraded_routes(self, threshold: float = 0.7) -> List[str]:
        """Get routes with quality below threshold."""
        with self._lock:
            return [
                route_id
                for route_id, q in self._quality.items()
                if q.quality_score < threshold
            ]
    
    def reset_route(self, route_id: str) -> None:
        """Reset quality metrics for a route."""
        with self._lock:
            if route_id in self._quality:
                del self._quality[route_id]
            if route_id in self._latencies:
                del self._latencies[route_id]
    
    def save(self, path: Optional[Path] = None) -> None:
        """Persist learning state to disk."""
        path = path or self.storage_path
        if not path:
            return
        
        with self._lock:
            data = {
                "quality": {
                    k: v.to_dict() for k, v in self._quality.items()
                },
                "latencies": {
                    k: v[-100:] for k, v in self._latencies.items()
                }
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        """Load learning state from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            data = json.loads(self.storage_path.read_text())
            
            # Reconstruct quality entries
            for route_id, q_data in data.get("quality", {}).items():
                self._quality[route_id] = RouteQuality(
                    route_id=route_id,
                    total_requests=q_data.get("total_requests", 0),
                    successful_requests=q_data.get("successful_requests", 0),
                    failed_requests=q_data.get("failed_requests", 0),
                    avg_latency_ms=q_data.get("avg_latency_ms", 0),
                    p95_latency_ms=q_data.get("p95_latency_ms", 0),
                    last_success=q_data.get("last_success", 0),
                    last_failure=q_data.get("last_failure", 0),
                    quality_score=q_data.get("quality_score", 1.0)
                )
            
            # Reconstruct latencies
            for route_id, latencies in data.get("latencies", {}).items():
                self._latencies[route_id] = latencies
                
            logger.info(f"Loaded {len(self._quality)} route quality entries")
            
        except Exception as e:
            logger.error(f"Failed to load route learner state: {e}")
    
    def stats(self) -> dict:
        """Get overall learning statistics."""
        with self._lock:
            if not self._quality:
                return {
                    "routes_tracked": 0,
                    "total_feedback": 0,
                    "avg_quality_score": 1.0
                }
            
            qualities = list(self._quality.values())
            return {
                "routes_tracked": len(qualities),
                "total_feedback": len(self._feedback),
                "avg_quality_score": np.mean([q.quality_score for q in qualities]),
                "degraded_routes": len(self.get_degraded_routes()),
                "total_requests": sum(q.total_requests for q in qualities),
                "total_failures": sum(q.failed_requests for q in qualities)
            }


# Global learner instance
_learner: Optional[RouteLearner] = None


def get_route_learner(storage_path: Optional[Path] = None) -> RouteLearner:
    """Get or create the global route learner."""
    global _learner
    if _learner is None:
        _learner = RouteLearner(storage_path=storage_path)
    return _learner
