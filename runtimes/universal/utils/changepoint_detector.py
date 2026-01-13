"""Change Point Detection using Ruptures.

Detects structural changes (change points) in time-series data using
various algorithms from the ruptures library.

Supported algorithms:
- Pelt: Optimal algorithm for offline detection with linear complexity
- Binseg: Binary segmentation (fast but approximate)
- Window: Sliding window algorithm (good for trend changes)
- BottomUp: Bottom-up segmentation (starts with many segments)

Usage:
    detector = ChangePointDetector(algorithm="pelt", model="rbf")
    result = detector.detect([1, 1, 1, 5, 5, 5, 2, 2, 2])
    # Returns: {"change_points": [3, 6], "n_segments": 3}
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


SUPPORTED_ALGORITHMS = ["pelt", "binseg", "window", "bottomup"]
SUPPORTED_MODELS = ["l1", "l2", "rbf", "normal", "ar"]


class ChangePointDetector:
    """Change point detector using ruptures library.

    Detects points in a time-series where the statistical properties
    (mean, variance, trend) change significantly.

    Attributes:
        algorithm: Detection algorithm (pelt, binseg, window, bottomup)
        model: Cost model for detection (l1, l2, rbf, normal, ar)
        min_size: Minimum segment size
    """

    def __init__(
        self,
        algorithm: str = "pelt",
        model: str = "rbf",
        min_size: int = 2,
    ):
        """Initialize change point detector.

        Args:
            algorithm: Detection algorithm to use
            model: Cost model for segment comparison
            min_size: Minimum size of segments between change points
        """
        if algorithm.lower() not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from {SUPPORTED_ALGORITHMS}")
        if model.lower() not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}. Choose from {SUPPORTED_MODELS}")

        self.algorithm = algorithm.lower()
        self.model = model.lower()
        self.min_size = min_size

    def detect(
        self,
        values: list[float],
        n_changepoints: int | None = None,
        penalty: float | None = None,
    ) -> dict[str, Any]:
        """Detect change points in a time-series.

        Either n_changepoints or penalty must be provided:
        - n_changepoints: Exact number of change points to find
        - penalty: Penalty value for Pelt/Binseg algorithms (higher = fewer points)

        Args:
            values: Time-series values
            n_changepoints: Number of change points to detect (if known)
            penalty: Penalty value for regularization (if n_changepoints unknown)

        Returns:
            Dictionary with:
                - change_points: Indices where changes occur (0-indexed)
                - n_segments: Number of segments after splitting
                - segment_boundaries: Start/end indices for each segment
        """
        import ruptures as rpt

        if len(values) < self.min_size * 2:
            raise ValueError(f"Signal too short. Need at least {self.min_size * 2} points.")

        if n_changepoints is None and penalty is None:
            # Default penalty based on signal length
            penalty = np.log(len(values)) * np.std(values) ** 2

        # Convert to numpy array
        signal = np.array(values, dtype=np.float64)

        # Handle 1D signal for ruptures (needs 2D)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        # Select algorithm
        algo_class = {
            "pelt": rpt.Pelt,
            "binseg": rpt.Binseg,
            "window": rpt.Window,
            "bottomup": rpt.BottomUp,
        }[self.algorithm]

        # Create and fit algorithm
        if self.algorithm == "window":
            algo = algo_class(width=max(self.min_size, 5), model=self.model)
        else:
            algo = algo_class(model=self.model, min_size=self.min_size)

        algo.fit(signal)

        # Predict change points
        if n_changepoints is not None:
            result = algo.predict(n_bkps=n_changepoints)
        else:
            result = algo.predict(pen=penalty)

        # Remove the last element (which is always signal length)
        change_points = [cp for cp in result if cp < len(values)]

        # Build segment boundaries
        boundaries = [0] + change_points + [len(values)]
        segments = [
            {"start": boundaries[i], "end": boundaries[i + 1]}
            for i in range(len(boundaries) - 1)
        ]

        return {
            "change_points": change_points,
            "n_segments": len(segments),
            "segment_boundaries": segments,
            "signal_length": len(values),
            "algorithm": self.algorithm,
            "model": self.model,
        }

    def detect_batch(
        self,
        series_list: list[list[float]],
        n_changepoints: int | None = None,
        penalty: float | None = None,
    ) -> list[dict[str, Any]]:
        """Detect change points in multiple time-series.

        Args:
            series_list: List of time-series
            n_changepoints: Number of change points per series
            penalty: Penalty value for regularization

        Returns:
            List of detection results
        """
        results = []
        for values in series_list:
            result = self.detect(values, n_changepoints=n_changepoints, penalty=penalty)
            results.append(result)
        return results
