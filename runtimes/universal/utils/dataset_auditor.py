"""Dataset Quality Audit using Cleanlab.

Provides utilities for finding label errors, duplicates, and quality issues
in classification datasets.

Usage:
    auditor = DatasetAuditor()
    result = auditor.audit(texts, labels, pred_probs)
    # Returns: {"label_issues": [...], "duplicates": [...], "summary": {...}}
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DatasetAuditor:
    """Dataset quality auditor using Cleanlab.

    Identifies:
    - Label errors (mislabeled examples)
    - Near-duplicate examples
    - Low-quality labels with confidence scores

    Attributes:
        label_names: Optional mapping of label indices to names
    """

    def __init__(
        self,
        label_names: list[str] | None = None,
    ):
        """Initialize dataset auditor.

        Args:
            label_names: Optional list of label names for human-readable output
        """
        self.label_names = label_names

    def audit(
        self,
        labels: list[int],
        pred_probs: np.ndarray,
        features: np.ndarray | None = None,
        check_duplicates: bool = True,
        duplicate_threshold: float = 0.95,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Audit dataset for quality issues.

        Args:
            labels: Ground truth labels (integer indices)
            pred_probs: Predicted probabilities from classifier (n_samples, n_classes)
            features: Feature vectors for duplicate detection (optional)
            check_duplicates: Whether to check for near-duplicates
            duplicate_threshold: Cosine similarity threshold for duplicates
            n_jobs: Number of parallel jobs for duplicate detection

        Returns:
            Dictionary with label_issues, duplicates, and summary statistics
        """
        from cleanlab.filter import find_label_issues

        labels_array = np.array(labels)

        # Handle empty dataset
        if len(labels) == 0:
            return {
                "label_issues": [],
                "duplicates": [],
                "summary": {
                    "total_samples": 0,
                    "label_issue_count": 0,
                    "label_issue_rate": 0,
                    "duplicate_pair_count": 0,
                    "unique_labels": 0,
                },
            }

        # Find label issues
        label_issue_mask = find_label_issues(
            labels=labels_array,
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
        )

        # Get predicted labels for comparison
        predicted_labels = np.argmax(pred_probs, axis=1)

        # Build label issues list
        label_issues = []
        for idx in label_issue_mask:
            given_label = int(labels_array[idx])
            suggested_label = int(predicted_labels[idx])
            confidence = float(pred_probs[idx, given_label])
            suggested_confidence = float(pred_probs[idx, suggested_label])

            issue = {
                "index": int(idx),
                "given_label": given_label,
                "suggested_label": suggested_label,
                "given_confidence": confidence,
                "suggested_confidence": suggested_confidence,
            }

            # Add label names if available
            if self.label_names:
                issue["given_label_name"] = self.label_names[given_label]
                issue["suggested_label_name"] = self.label_names[suggested_label]

            label_issues.append(issue)

        # Find duplicates if requested
        duplicates = []
        if check_duplicates and features is not None:
            duplicates = self._find_duplicates(
                features=features,
                threshold=duplicate_threshold,
            )

        # Compute summary statistics
        summary = {
            "total_samples": len(labels),
            "label_issue_count": len(label_issues),
            "label_issue_rate": len(label_issues) / len(labels) if labels else 0,
            "duplicate_pair_count": len(duplicates),
            "unique_labels": len(set(labels)),
        }

        return {
            "label_issues": label_issues,
            "duplicates": duplicates,
            "summary": summary,
        }

    def _find_duplicates(
        self,
        features: np.ndarray,
        threshold: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Find near-duplicate examples using cosine similarity.

        Args:
            features: Feature vectors (n_samples, n_features)
            threshold: Cosine similarity threshold

        Returns:
            List of duplicate pairs with similarity scores
        """
        # Normalize features for cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms > 0, norms, 1)
        normalized = features / norms

        # Compute pairwise similarities efficiently
        # For large datasets, this could be batched
        n_samples = len(features)
        duplicates = []

        # Only compute upper triangle (avoid duplicates and self-comparison)
        for i in range(n_samples):
            # Vectorized similarity for all j > i
            if i + 1 < n_samples:
                similarities = np.dot(normalized[i], normalized[i + 1 :].T)
                high_sim_indices = np.where(similarities >= threshold)[0] + i + 1

                for j in high_sim_indices:
                    duplicates.append({
                        "index_a": int(i),
                        "index_b": int(j),
                        "similarity": float(similarities[j - i - 1]),
                    })

        # Sort by similarity (highest first)
        duplicates.sort(key=lambda x: x["similarity"], reverse=True)

        return duplicates

    def get_label_quality_scores(
        self,
        labels: list[int],
        pred_probs: np.ndarray,
    ) -> np.ndarray:
        """Get per-sample label quality scores.

        Args:
            labels: Ground truth labels
            pred_probs: Predicted probabilities

        Returns:
            Array of quality scores (0-1, higher = better quality)
        """
        from cleanlab.rank import get_label_quality_scores

        labels_array = np.array(labels)
        return get_label_quality_scores(labels_array, pred_probs)

    def suggest_corrections(
        self,
        labels: list[int],
        pred_probs: np.ndarray,
        min_confidence_diff: float = 0.2,
    ) -> list[dict[str, Any]]:
        """Suggest label corrections with high confidence.

        Only suggests corrections where the predicted label has
        significantly higher confidence than the given label.

        Args:
            labels: Ground truth labels
            pred_probs: Predicted probabilities
            min_confidence_diff: Minimum confidence difference to suggest

        Returns:
            List of suggested corrections
        """
        labels_array = np.array(labels)
        predicted_labels = np.argmax(pred_probs, axis=1)

        corrections = []
        for i in range(len(labels)):
            given = labels_array[i]
            predicted = predicted_labels[i]

            if given != predicted:
                given_conf = pred_probs[i, given]
                predicted_conf = pred_probs[i, predicted]
                conf_diff = predicted_conf - given_conf

                if conf_diff >= min_confidence_diff:
                    correction = {
                        "index": i,
                        "given_label": int(given),
                        "suggested_label": int(predicted),
                        "confidence_improvement": float(conf_diff),
                    }

                    if self.label_names:
                        correction["given_label_name"] = self.label_names[given]
                        correction["suggested_label_name"] = self.label_names[predicted]

                    corrections.append(correction)

        # Sort by confidence improvement (highest first)
        corrections.sort(key=lambda x: x["confidence_improvement"], reverse=True)

        return corrections


def create_auditor_for_classifier(
    classifier_model: Any,
    texts: list[str],
    labels: list[int],
    label_names: list[str] | None = None,
    batch_size: int = 32,
) -> tuple[DatasetAuditor, np.ndarray]:
    """Create auditor with predictions from a classifier.

    This is a convenience function that:
    1. Runs the classifier on all texts to get prediction probabilities
    2. Creates an auditor configured with the label names

    Args:
        classifier_model: Classifier with predict_proba or predict_batch method
        texts: List of text samples
        labels: List of label indices
        label_names: Optional label name mapping
        batch_size: Batch size for inference

    Returns:
        Tuple of (DatasetAuditor, pred_probs array)
    """
    # Get predictions from classifier
    if hasattr(classifier_model, "predict_proba"):
        # sklearn-style
        pred_probs = classifier_model.predict_proba(texts)
    elif hasattr(classifier_model, "predict_batch"):
        # LlamaFarm-style async classifier
        import asyncio

        async def get_probs():
            results = await classifier_model.predict_batch(texts, batch_size=batch_size)
            # Extract probabilities from results
            n_classes = len(label_names) if label_names else max(labels) + 1
            probs = np.zeros((len(texts), n_classes))
            for i, result in enumerate(results):
                if isinstance(result, dict) and "class_scores" in result:
                    for cls_name, score in result["class_scores"].items():
                        if label_names and cls_name in label_names:
                            probs[i, label_names.index(cls_name)] = score
            return probs

        pred_probs = asyncio.run(get_probs())
    else:
        raise ValueError("Classifier must have predict_proba or predict_batch method")

    auditor = DatasetAuditor(label_names=label_names)
    return auditor, pred_probs
