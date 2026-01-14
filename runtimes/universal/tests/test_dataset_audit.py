"""Tests for Dataset Quality Audit with Cleanlab.

Phase 16 of Universal Runtime ML Enhancements.
"""

import numpy as np

from utils.dataset_auditor import DatasetAuditor


def generate_clean_dataset(
    n_samples: int = 100,
    n_classes: int = 3,
    seed: int = 42,
) -> tuple[list[int], np.ndarray]:
    """Generate a clean dataset where predictions match labels."""
    np.random.seed(seed)
    labels = np.random.randint(0, n_classes, n_samples).tolist()

    # Create predictions that mostly agree with labels
    pred_probs = np.zeros((n_samples, n_classes))
    for i, label in enumerate(labels):
        # High probability for correct label
        pred_probs[i, label] = 0.8 + np.random.uniform(0, 0.15)
        # Distribute remaining probability
        remaining = 1 - pred_probs[i, label]
        for j in range(n_classes):
            if j != label:
                pred_probs[i, j] = remaining / (n_classes - 1)

    return labels, pred_probs


def generate_noisy_dataset(
    n_samples: int = 100,
    n_classes: int = 3,
    noise_rate: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], np.ndarray, list[int]]:
    """Generate a dataset with label noise.

    Returns labels, pred_probs, and indices of noisy samples.
    """
    np.random.seed(seed)

    # True labels from predictions
    true_labels = np.random.randint(0, n_classes, n_samples)

    # Create strong predictions for true labels
    pred_probs = np.zeros((n_samples, n_classes))
    for i, label in enumerate(true_labels):
        pred_probs[i, label] = 0.85 + np.random.uniform(0, 0.1)
        remaining = 1 - pred_probs[i, label]
        for j in range(n_classes):
            if j != label:
                pred_probs[i, j] = remaining / (n_classes - 1)

    # Add label noise
    labels = true_labels.tolist()
    n_noisy = int(n_samples * noise_rate)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False).tolist()

    for idx in noisy_indices:
        # Flip to a different label
        wrong_labels = [lbl for lbl in range(n_classes) if lbl != true_labels[idx]]
        labels[idx] = np.random.choice(wrong_labels)

    return labels, pred_probs, noisy_indices


def generate_features_with_duplicates(
    n_samples: int = 100,
    n_features: int = 10,
    n_duplicate_pairs: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Generate features with some near-duplicates."""
    np.random.seed(seed)

    features = np.random.randn(n_samples, n_features)

    # Create near-duplicates
    duplicate_pairs = []
    for i in range(n_duplicate_pairs):
        idx_a = i * 2
        idx_b = i * 2 + 1
        if idx_b < n_samples:
            # Make idx_b a near-copy of idx_a
            features[idx_b] = features[idx_a] + np.random.randn(n_features) * 0.01
            duplicate_pairs.append((idx_a, idx_b))

    return features, duplicate_pairs


class TestDatasetAuditor:
    """Test DatasetAuditor class."""

    def test_auditor_initializes(self):
        """Test that auditor initializes correctly."""
        auditor = DatasetAuditor()
        assert auditor is not None

    def test_auditor_with_label_names(self):
        """Test auditor with label names."""
        auditor = DatasetAuditor(label_names=["cat", "dog", "bird"])
        assert auditor.label_names == ["cat", "dog", "bird"]

    def test_audit_returns_dict(self):
        """Test that audit returns a dictionary."""
        auditor = DatasetAuditor()
        labels, pred_probs = generate_clean_dataset()
        result = auditor.audit(labels, pred_probs)
        assert isinstance(result, dict)
        assert "label_issues" in result
        assert "duplicates" in result
        assert "summary" in result

    def test_audit_finds_label_errors(self):
        """Test that audit finds label errors in noisy data."""
        auditor = DatasetAuditor()
        labels, pred_probs, noisy_indices = generate_noisy_dataset(
            n_samples=100, noise_rate=0.2, seed=42
        )
        result = auditor.audit(labels, pred_probs)

        # Should find some label issues
        assert len(result["label_issues"]) > 0

        # Found issues should overlap with actual noisy indices
        found_indices = set(issue["index"] for issue in result["label_issues"])
        actual_noisy = set(noisy_indices)

        # At least some overlap expected
        overlap = found_indices & actual_noisy
        assert len(overlap) > 0

    def test_label_issue_has_required_keys(self):
        """Test that label issues have required keys."""
        auditor = DatasetAuditor()
        labels, pred_probs, _ = generate_noisy_dataset()
        result = auditor.audit(labels, pred_probs)

        if result["label_issues"]:
            issue = result["label_issues"][0]
            assert "index" in issue
            assert "given_label" in issue
            assert "suggested_label" in issue
            assert "given_confidence" in issue
            assert "suggested_confidence" in issue

    def test_label_names_in_issues(self):
        """Test that label names appear in issues when provided."""
        auditor = DatasetAuditor(label_names=["cat", "dog", "bird"])
        labels, pred_probs, _ = generate_noisy_dataset(n_classes=3)
        result = auditor.audit(labels, pred_probs)

        if result["label_issues"]:
            issue = result["label_issues"][0]
            assert "given_label_name" in issue
            assert "suggested_label_name" in issue
            assert issue["given_label_name"] in ["cat", "dog", "bird"]

    def test_summary_statistics(self):
        """Test that summary statistics are computed."""
        auditor = DatasetAuditor()
        labels, pred_probs = generate_clean_dataset(n_samples=50, n_classes=3)
        result = auditor.audit(labels, pred_probs)

        summary = result["summary"]
        assert summary["total_samples"] == 50
        assert summary["unique_labels"] == 3
        assert "label_issue_count" in summary
        assert "label_issue_rate" in summary


class TestDuplicateDetection:
    """Test duplicate detection functionality."""

    def test_finds_duplicates(self):
        """Test that auditor finds near-duplicates."""
        auditor = DatasetAuditor()
        labels, pred_probs = generate_clean_dataset(n_samples=20)
        features, duplicate_pairs = generate_features_with_duplicates(
            n_samples=20, n_duplicate_pairs=3
        )

        result = auditor.audit(
            labels, pred_probs, features=features, check_duplicates=True
        )

        # Should find duplicates
        assert len(result["duplicates"]) >= len(duplicate_pairs)

    def test_duplicate_has_required_keys(self):
        """Test that duplicates have required keys."""
        auditor = DatasetAuditor()
        labels, pred_probs = generate_clean_dataset(n_samples=20)
        features, _ = generate_features_with_duplicates(n_samples=20)

        result = auditor.audit(
            labels, pred_probs, features=features, check_duplicates=True
        )

        if result["duplicates"]:
            dup = result["duplicates"][0]
            assert "index_a" in dup
            assert "index_b" in dup
            assert "similarity" in dup

    def test_duplicate_threshold(self):
        """Test that duplicate threshold works."""
        auditor = DatasetAuditor()
        labels, pred_probs = generate_clean_dataset(n_samples=20)
        features, _ = generate_features_with_duplicates(n_samples=20)

        # Higher threshold = fewer duplicates
        result_high = auditor.audit(
            labels, pred_probs, features=features, duplicate_threshold=0.99
        )
        result_low = auditor.audit(
            labels, pred_probs, features=features, duplicate_threshold=0.8
        )

        assert len(result_low["duplicates"]) >= len(result_high["duplicates"])

    def test_no_duplicates_when_disabled(self):
        """Test that duplicates are empty when check_duplicates=False."""
        auditor = DatasetAuditor()
        labels, pred_probs = generate_clean_dataset()
        features, _ = generate_features_with_duplicates()

        result = auditor.audit(
            labels, pred_probs, features=features, check_duplicates=False
        )

        assert result["duplicates"] == []


class TestLabelQualityScores:
    """Test label quality score computation."""

    def test_quality_scores_shape(self):
        """Test that quality scores have correct shape."""
        auditor = DatasetAuditor()
        labels, pred_probs = generate_clean_dataset(n_samples=50)
        scores = auditor.get_label_quality_scores(labels, pred_probs)

        assert len(scores) == 50

    def test_quality_scores_range(self):
        """Test that quality scores are in [0, 1]."""
        auditor = DatasetAuditor()
        labels, pred_probs = generate_clean_dataset()
        scores = auditor.get_label_quality_scores(labels, pred_probs)

        assert np.all(scores >= 0)
        assert np.all(scores <= 1)

    def test_noisy_labels_have_lower_quality(self):
        """Test that noisy labels have lower quality scores."""
        auditor = DatasetAuditor()
        labels, pred_probs, noisy_indices = generate_noisy_dataset(
            n_samples=100, noise_rate=0.2
        )
        scores = auditor.get_label_quality_scores(labels, pred_probs)

        # Average quality of noisy should be lower than clean
        noisy_scores = scores[noisy_indices]
        clean_indices = [i for i in range(100) if i not in noisy_indices]
        clean_scores = scores[clean_indices]

        assert np.mean(noisy_scores) < np.mean(clean_scores)


class TestSuggestCorrections:
    """Test label correction suggestions."""

    def test_suggest_corrections_format(self):
        """Test that corrections have correct format."""
        auditor = DatasetAuditor()
        labels, pred_probs, _ = generate_noisy_dataset()
        corrections = auditor.suggest_corrections(labels, pred_probs)

        if corrections:
            corr = corrections[0]
            assert "index" in corr
            assert "given_label" in corr
            assert "suggested_label" in corr
            assert "confidence_improvement" in corr

    def test_corrections_sorted_by_confidence(self):
        """Test that corrections are sorted by confidence improvement."""
        auditor = DatasetAuditor()
        labels, pred_probs, _ = generate_noisy_dataset()
        corrections = auditor.suggest_corrections(labels, pred_probs)

        if len(corrections) > 1:
            improvements = [c["confidence_improvement"] for c in corrections]
            assert improvements == sorted(improvements, reverse=True)

    def test_min_confidence_filter(self):
        """Test that min_confidence_diff filters corrections."""
        auditor = DatasetAuditor()
        labels, pred_probs, _ = generate_noisy_dataset()

        corr_low = auditor.suggest_corrections(
            labels, pred_probs, min_confidence_diff=0.1
        )
        corr_high = auditor.suggest_corrections(
            labels, pred_probs, min_confidence_diff=0.5
        )

        # Higher threshold = fewer corrections
        assert len(corr_high) <= len(corr_low)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        auditor = DatasetAuditor()
        labels: list[int] = []
        pred_probs = np.array([]).reshape(0, 3)

        # Should handle gracefully
        result = auditor.audit(labels, pred_probs)
        assert result["summary"]["total_samples"] == 0

    def test_single_sample(self):
        """Test handling of single sample."""
        auditor = DatasetAuditor()
        labels = [0]
        pred_probs = np.array([[0.9, 0.1]])

        result = auditor.audit(labels, pred_probs)
        assert result["summary"]["total_samples"] == 1

    def test_binary_classification(self):
        """Test with binary classification."""
        auditor = DatasetAuditor(label_names=["negative", "positive"])
        labels, pred_probs, _ = generate_noisy_dataset(n_classes=2)

        result = auditor.audit(labels, pred_probs)
        assert result["summary"]["unique_labels"] <= 2
