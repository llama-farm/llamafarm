"""Concept Drift Detection using River.

Detects when the statistical properties of streaming data change over time.
This is important for monitoring ML models in production to know when to retrain.

Supported detectors:
- ADWIN: Adaptive Windowing algorithm (for numeric data)
- PageHinkley: Page-Hinkley test (for numeric data)
- KSWIN: Kolmogorov-Smirnov Windowing (for distribution changes)
- DDM: Drift Detection Method (for error rate monitoring)

Usage:
    detector = DriftDetector(algorithm="adwin")
    for value in stream:
        result = detector.update(value)
        if result["drift_detected"]:
            print(f"Drift at index {result['index']}")
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_ALGORITHMS = ["adwin", "page_hinkley", "kswin", "ddm"]


class DriftDetector:
    """Concept drift detector using River library.

    Detects when the statistical properties of a data stream change,
    indicating that a model may need retraining.

    Attributes:
        algorithm: Detection algorithm (adwin, page_hinkley, kswin, ddm)
        detector: The underlying River drift detector
    """

    def __init__(
        self,
        algorithm: str = "adwin",
        **kwargs: Any,
    ):
        """Initialize drift detector.

        Args:
            algorithm: Detection algorithm to use
            **kwargs: Algorithm-specific parameters
                For ADWIN: delta (default 0.002)
                For PageHinkley: min_instances, delta, threshold, alpha
                For KSWIN: alpha, window_size, stat_size, seed
                For DDM: (no additional parameters)
        """
        if algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from {SUPPORTED_ALGORITHMS}")

        self.algorithm = algorithm.lower()
        self.params = kwargs
        self.detector = self._create_detector()
        self._index = 0
        self._drift_points: list[dict[str, Any]] = []

    def _create_detector(self) -> Any:
        """Create the appropriate River drift detector."""
        if self.algorithm == "adwin":
            from river.drift import ADWIN
            delta = self.params.get("delta", 0.002)
            return ADWIN(delta=delta)

        elif self.algorithm == "page_hinkley":
            from river.drift import PageHinkley
            return PageHinkley(
                min_instances=self.params.get("min_instances", 30),
                delta=self.params.get("delta", 0.005),
                threshold=self.params.get("threshold", 50.0),
                alpha=self.params.get("alpha", 1 - 0.0001),
            )

        elif self.algorithm == "kswin":
            from river.drift import KSWIN
            return KSWIN(
                alpha=self.params.get("alpha", 0.005),
                window_size=self.params.get("window_size", 100),
                stat_size=self.params.get("stat_size", 30),
                seed=self.params.get("seed", None),
            )

        elif self.algorithm == "ddm":
            from river.drift.binary import DDM
            return DDM()

        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def update(self, value: float) -> dict[str, Any]:
        """Update detector with a new value and check for drift.

        Args:
            value: New data point (or error indicator for DDM)

        Returns:
            Dictionary with:
                - drift_detected: Whether drift was detected
                - index: Current index in the stream
                - value: The value that was processed
                - warning (for DDM): Whether in warning zone
        """
        self._index += 1
        self.detector.update(value)

        drift_detected = self.detector.drift_detected
        result = {
            "drift_detected": drift_detected,
            "index": self._index,
            "value": value,
        }

        # DDM has additional warning zone
        if self.algorithm == "ddm":
            result["warning"] = getattr(self.detector, "warning_detected", False)

        if drift_detected:
            drift_info = {
                "index": self._index,
                "algorithm": self.algorithm,
            }
            self._drift_points.append(drift_info)
            result["drift_info"] = drift_info

        return result

    def update_batch(self, values: list[float]) -> dict[str, Any]:
        """Update detector with multiple values.

        Args:
            values: List of data points

        Returns:
            Dictionary with:
                - total_processed: Number of values processed
                - drift_detected: Whether any drift was detected
                - drift_points: List of indices where drift occurred
                - final_index: Index after processing all values
        """
        drift_points = []
        for value in values:
            result = self.update(value)
            if result["drift_detected"]:
                drift_points.append(result["index"])

        return {
            "total_processed": len(values),
            "drift_detected": len(drift_points) > 0,
            "drift_points": drift_points,
            "final_index": self._index,
            "algorithm": self.algorithm,
        }

    def reset(self) -> None:
        """Reset the detector to initial state."""
        self.detector = self._create_detector()
        self._index = 0
        self._drift_points = []

    def get_state(self) -> dict[str, Any]:
        """Get current state of the detector.

        Returns:
            Dictionary with detector state information
        """
        return {
            "algorithm": self.algorithm,
            "index": self._index,
            "total_drift_points": len(self._drift_points),
            "drift_points": self._drift_points,
            "params": self.params,
        }


def detect_drift_in_stream(
    values: list[float],
    algorithm: str = "adwin",
    **kwargs: Any,
) -> dict[str, Any]:
    """Convenience function to detect drift in a complete stream.

    Args:
        values: Complete data stream
        algorithm: Detection algorithm
        **kwargs: Algorithm-specific parameters

    Returns:
        Dictionary with drift detection results
    """
    detector = DriftDetector(algorithm=algorithm, **kwargs)
    return detector.update_batch(values)
