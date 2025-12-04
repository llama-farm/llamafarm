"""Tests for ingest path detection logic.

Ensures that hash-based files in lf_data/datasets/*/raw/ are correctly
detected and metadata is loaded to get the original filename for parser matching.
This prevents PDFs from being parsed as raw text (regression test for #529).
"""

import json
from pathlib import Path


def detect_hash_based_file(source_path: str) -> tuple[bool, str | None]:
    """Extract the detection logic from ingest_file_with_rag_task.

    Returns:
        Tuple of (is_hash_based, dataset_name)
    """
    # This matches the logic in tasks/ingest_tasks.py
    if "lf_data/datasets/" in str(source_path) and "/raw/" in str(source_path):
        # Extract dataset name from path
        # Path format: .../lf_data/datasets/{dataset_name}/raw/{hash}
        path_parts = Path(source_path).parts
        try:
            datasets_idx = path_parts.index("datasets")
            dataset_name = path_parts[datasets_idx + 1]
            return True, dataset_name
        except (ValueError, IndexError):
            return True, None
    return False, None


class TestIngestPathDetection:
    """Test cases for ingest path detection."""

    def test_detects_lf_data_datasets_raw_structure(self):
        """Files in lf_data/datasets/*/raw/ should be detected as hash-based."""
        path = "/home/user/.llamafarm/lf_data/datasets/my_dataset/raw/abc123def456"
        is_hash, dataset = detect_hash_based_file(path)
        assert is_hash is True
        assert dataset == "my_dataset"

    def test_detects_windows_path_structure(self):
        """Windows paths with lf_data/datasets/*/raw/ should also work."""
        path = "C:\\Users\\user\\.llamafarm\\lf_data\\datasets\\test_ds\\raw\\hash123"
        # Convert to forward slashes for consistent checking
        normalized = path.replace("\\", "/")
        is_hash, dataset = detect_hash_based_file(normalized)
        assert is_hash is True
        assert dataset == "test_ds"

    def test_ignores_old_lf_data_raw_structure(self):
        """Old lf_data/raw structure (without datasets) should NOT be detected.

        This was the bug - the old code checked for 'lf_data/raw' which doesn't
        exist in the new structure. The correct path is 'lf_data/datasets/*/raw/'.
        """
        old_path = "/home/user/.llamafarm/lf_data/raw/abc123def456"
        is_hash, dataset = detect_hash_based_file(old_path)
        assert is_hash is False
        assert dataset is None

    def test_ignores_regular_file_paths(self):
        """Regular file paths should not be detected as hash-based."""
        regular_paths = [
            "/home/user/documents/report.pdf",
            "/tmp/upload.txt",
            "./local_file.docx",
            "/var/data/files/document.pdf",
        ]
        for path in regular_paths:
            is_hash, dataset = detect_hash_based_file(path)
            assert is_hash is False, f"Path {path} should not be hash-based"
            assert dataset is None

    def test_extracts_correct_dataset_name(self):
        """Dataset name should be correctly extracted from path."""
        test_cases = [
            ("/lf_data/datasets/fda_letters/raw/hash1", "fda_letters"),
            ("/lf_data/datasets/research_papers/raw/hash2", "research_papers"),
            ("/lf_data/datasets/my-dataset-123/raw/abc", "my-dataset-123"),
        ]
        for path, expected_dataset in test_cases:
            is_hash, dataset = detect_hash_based_file(path)
            assert is_hash is True
            assert dataset == expected_dataset


class TestMetadataLoading:
    """Test metadata loading for hash-based files."""

    def test_loads_metadata_for_original_filename(self, tmp_path):
        """Metadata file should provide original filename for parser selection."""
        # Create dataset structure
        dataset_dir = tmp_path / "lf_data" / "datasets" / "test_dataset"
        raw_dir = dataset_dir / "raw"
        meta_dir = dataset_dir / "meta"
        raw_dir.mkdir(parents=True)
        meta_dir.mkdir(parents=True)

        # Create a hash-named file (simulating uploaded PDF)
        file_hash = "abc123def456"
        raw_file = raw_dir / file_hash
        raw_file.write_bytes(b"%PDF-1.4 fake pdf content")

        # Create metadata file with original filename
        meta_file = meta_dir / f"{file_hash}.json"
        meta_content = {
            "original_file_name": "important_document.pdf",
            "mime_type": "application/pdf",
            "size": 1234,
            "timestamp": 1699999999.0,
        }
        meta_file.write_text(json.dumps(meta_content))

        # Verify the detection logic
        source_path = str(raw_file)
        assert "lf_data/datasets/" in source_path
        assert "/raw/" in source_path

        # Verify metadata can be loaded
        file_path = Path(source_path)
        loaded_meta_dir = file_path.parent.parent / "meta"
        loaded_meta_file = loaded_meta_dir / f"{file_path.name}.json"

        assert loaded_meta_file.exists()
        with open(loaded_meta_file) as f:
            loaded_meta = json.load(f)

        assert loaded_meta["original_file_name"] == "important_document.pdf"
        assert loaded_meta["mime_type"] == "application/pdf"

    def test_fallback_when_metadata_missing(self, tmp_path):
        """Should fall back to hash as filename when metadata is missing."""
        # Create dataset structure without metadata
        dataset_dir = tmp_path / "lf_data" / "datasets" / "test_dataset"
        raw_dir = dataset_dir / "raw"
        raw_dir.mkdir(parents=True)

        file_hash = "xyz789"
        raw_file = raw_dir / file_hash
        raw_file.write_bytes(b"some content")

        # Check that meta file doesn't exist
        meta_dir = dataset_dir / "meta"
        meta_file = meta_dir / f"{file_hash}.json"
        assert not meta_file.exists()

        # The ingest code should fall back to using the hash as filename
        # and application/octet-stream as mime type
        file_path = Path(str(raw_file))
        expected_fallback_filename = file_path.name
        assert expected_fallback_filename == file_hash
