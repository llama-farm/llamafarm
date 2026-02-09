"""Tests for model packaging and distribution.

Verifies:
- Package creation with model weights + metadata
- Package reading (metadata only, no extraction)
- Package extraction with model path
- Class map and validation metrics in package
- Package listing and deletion
- Security (path traversal prevention)
"""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

import pytest

from vision_training.model_package import (
    ModelPackage,
    ModelPackager,
    PackageMetadata,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def packager(temp_dir):
    return ModelPackager(packages_dir=temp_dir / "packages")


@pytest.fixture
def model_file(temp_dir):
    """Create a fake model weights file."""
    model = temp_dir / "best.pt"
    model.write_bytes(b"fake_pytorch_model_weights_v1")
    return model


class TestCreatePackage:
    """Tests for package creation."""

    def test_creates_tar_gz(self, packager, model_file):
        """create_package() produces a .tar.gz file."""
        metadata = PackageMetadata(
            model_id="yolov8n-custom",
            version=1,
        )

        package = packager.create_package(
            model_path=str(model_file),
            metadata=metadata,
        )

        pkg_path = Path(package.package_path)
        assert pkg_path.exists()
        assert pkg_path.suffix == ".gz"
        assert ".tar" in pkg_path.stem

    def test_package_contains_model(self, packager, model_file):
        """Package contains the model weights."""
        package = packager.create_package(
            model_path=str(model_file),
            metadata=PackageMetadata(model_id="test", version=1),
        )

        with tarfile.open(package.package_path, "r:gz") as tar:
            names = tar.getnames()
            assert "model.pt" in names

    def test_package_contains_metadata(self, packager, model_file):
        """Package contains metadata.json with correct fields."""
        metadata = PackageMetadata(
            model_id="yolov8n-birds",
            version=3,
            base_model="yolov8n",
            task="detection",
            class_names=["robin", "sparrow", "eagle"],
            training_samples=500,
            description="Bird detector fine-tuned from yolov8n",
        )

        package = packager.create_package(
            model_path=str(model_file),
            metadata=metadata,
        )

        with tarfile.open(package.package_path, "r:gz") as tar:
            meta_file = tar.extractfile("metadata.json")
            assert meta_file is not None
            data = json.loads(meta_file.read().decode())

            assert data["model_id"] == "yolov8n-birds"
            assert data["version"] == 3
            assert data["base_model"] == "yolov8n"
            assert data["class_names"] == ["robin", "sparrow", "eagle"]
            assert data["training_samples"] == 500

    def test_package_contains_class_map(self, packager, model_file):
        """Package contains class_map.json."""
        class_map = {"person": 0, "car": 1, "bike": 2}
        package = packager.create_package(
            model_path=str(model_file),
            metadata=PackageMetadata(model_id="test", version=1),
            class_map=class_map,
        )

        with tarfile.open(package.package_path, "r:gz") as tar:
            cm_file = tar.extractfile("class_map.json")
            assert cm_file is not None
            data = json.loads(cm_file.read().decode())
            assert data == class_map

    def test_package_includes_validation_metrics(self, packager, model_file):
        """Package contains validation_metrics.json when provided."""
        val_metrics = {
            "accuracy": 0.92,
            "mean_iou": 0.85,
            "per_class": {"person": 0.95, "car": 0.89},
        }

        package = packager.create_package(
            model_path=str(model_file),
            metadata=PackageMetadata(model_id="test", version=1),
            validation_metrics=val_metrics,
        )

        with tarfile.open(package.package_path, "r:gz") as tar:
            val_file = tar.extractfile("validation_metrics.json")
            assert val_file is not None
            data = json.loads(val_file.read().decode())
            assert data["accuracy"] == 0.92

    def test_missing_model_file_raises(self, packager):
        """create_package() raises FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError):
            packager.create_package(
                model_path="/nonexistent/model.pt",
                metadata=PackageMetadata(model_id="test", version=1),
            )

    def test_class_map_updates_metadata(self, packager, model_file):
        """Providing class_map updates metadata.class_names."""
        class_map = {"bird": 0, "cat": 1}
        metadata = PackageMetadata(model_id="test", version=1)

        packager.create_package(
            model_path=str(model_file),
            metadata=metadata,
            class_map=class_map,
        )

        assert metadata.class_map == class_map
        assert set(metadata.class_names) == {"bird", "cat"}


class TestReadPackage:
    """Tests for reading package metadata."""

    def test_reads_metadata_without_extraction(self, packager, model_file):
        """read_package() returns metadata without extracting weights."""
        packager.create_package(
            model_path=str(model_file),
            metadata=PackageMetadata(
                model_id="yolov8n-custom",
                version=2,
                task="detection",
            ),
        )

        packages = packager.list_packages()
        assert len(packages) == 1

        pkg = packages[0]
        assert pkg.metadata.model_id == "yolov8n-custom"
        assert pkg.metadata.version == 2
        assert pkg.is_extracted is False

    def test_read_nonexistent_raises(self, packager):
        """read_package() raises for missing file."""
        with pytest.raises(FileNotFoundError):
            packager.read_package("/nonexistent/package.tar.gz")


class TestExtractPackage:
    """Tests for package extraction."""

    def test_extracts_model_weights(self, packager, model_file, temp_dir):
        """extract_package() extracts model weights to filesystem."""
        package = packager.create_package(
            model_path=str(model_file),
            metadata=PackageMetadata(model_id="test", version=1),
        )

        extracted = packager.extract_package(package.package_path)

        assert extracted.is_extracted is True
        assert extracted.model_path != ""
        assert Path(extracted.model_path).exists()
        assert Path(extracted.model_path).read_bytes() == model_file.read_bytes()

    def test_extracts_to_custom_dir(self, packager, model_file, temp_dir):
        """extract_package() extracts to specified directory."""
        package = packager.create_package(
            model_path=str(model_file),
            metadata=PackageMetadata(model_id="test", version=1),
        )

        custom_dir = temp_dir / "custom_extract"
        extracted = packager.extract_package(
            package.package_path,
            extract_dir=custom_dir,
        )

        assert str(custom_dir) in extracted.model_path

    def test_extraction_preserves_metadata(self, packager, model_file):
        """Extracted package has correct metadata."""
        package = packager.create_package(
            model_path=str(model_file),
            metadata=PackageMetadata(
                model_id="bird-detector",
                version=5,
                class_names=["robin", "eagle"],
            ),
        )

        extracted = packager.extract_package(package.package_path)
        assert extracted.metadata.model_id == "bird-detector"
        assert extracted.metadata.version == 5
        assert extracted.metadata.class_names == ["robin", "eagle"]


class TestListAndDelete:
    """Tests for package listing and deletion."""

    def test_list_empty(self, packager):
        """list_packages() returns empty list when no packages."""
        assert packager.list_packages() == []

    def test_list_multiple(self, packager, model_file):
        """list_packages() returns all packages."""
        for i in range(3):
            packager.create_package(
                model_path=str(model_file),
                metadata=PackageMetadata(model_id=f"model_{i}", version=1),
            )

        packages = packager.list_packages()
        assert len(packages) == 3

    def test_delete_package(self, packager, model_file):
        """delete_package() removes the file."""
        package = packager.create_package(
            model_path=str(model_file),
            metadata=PackageMetadata(model_id="test", version=1),
        )

        assert packager.delete_package(package.package_path) is True
        assert not Path(package.package_path).exists()
        assert packager.list_packages() == []

    def test_delete_nonexistent(self, packager):
        """delete_package() returns False for missing file."""
        assert packager.delete_package("/nonexistent.tar.gz") is False


class TestPackageMetadata:
    """Tests for PackageMetadata serialization."""

    def test_round_trip(self):
        """to_dict/from_dict preserves all fields."""
        meta = PackageMetadata(
            model_id="yolov8n-custom",
            version=3,
            base_model="yolov8n",
            task="detection",
            class_map={"person": 0, "car": 1},
            class_names=["person", "car"],
            training_samples=1000,
            accuracy=0.92,
        )

        data = meta.to_dict()
        restored = PackageMetadata.from_dict(data)

        assert restored.model_id == "yolov8n-custom"
        assert restored.version == 3
        assert restored.class_map == {"person": 0, "car": 1}
        assert restored.accuracy == 0.92

    def test_from_dict_ignores_unknown(self):
        """from_dict() ignores unknown fields."""
        data = {
            "model_id": "test",
            "version": 1,
            "unknown_field": "should be ignored",
        }
        meta = PackageMetadata.from_dict(data)
        assert meta.model_id == "test"


class TestSecurity:
    """Security tests for package handling."""

    def test_path_traversal_prevention(self, packager, temp_dir):
        """extract_package() rejects archives with path traversal."""
        # Create a malicious archive with ../ paths
        malicious_pkg = temp_dir / "evil.tar.gz"
        with tarfile.open(str(malicious_pkg), "w:gz") as tar:
            # Add a file with path traversal
            import io
            data = b"malicious content"
            info = tarfile.TarInfo(name="../../../etc/evil.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        with pytest.raises(ValueError, match="Unsafe path"):
            packager.extract_package(str(malicious_pkg))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
