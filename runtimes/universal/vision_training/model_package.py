"""Model packaging for distribution across LlamaFarm instances.

Package format (.tar.gz):
    model.pt / model.onnx      - Weights
    metadata.json               - version, base_model, class_map, training info
    class_map.json              - {0: "person", 1: "bird", ...}
    validation_metrics.json     - What it scored on the validation gate

Standalone: packages sit in ~/.llamafarm/vision/packages/ for manual transfer.
Mesh: packages get announced via Atmosphere MODEL_AVAILABLE gossip.
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PackageMetadata:
    """Metadata about a packaged model."""

    model_id: str
    version: int = 1
    base_model: str = ""            # What it was fine-tuned from
    task: str = "detection"          # detection, classification, segmentation
    class_map: dict[str, int] = field(default_factory=dict)  # class_name -> class_id
    class_names: list[str] = field(default_factory=list)      # ordered by class_id
    training_samples: int = 0       # How many samples were used
    training_source: str = ""       # "auto_trainer", "manual", "transfer"
    source_node: str = "local"      # Atmosphere node ID
    description: str = ""
    created_at: str = ""            # ISO timestamp
    lineage: list[str] = field(default_factory=list)  # Previous model IDs in chain

    # Validation metrics summary
    accuracy: float = 0.0
    mean_iou: float = 0.0
    validation_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PackageMetadata:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelPackage:
    """A packaged model ready for distribution."""

    package_path: str
    metadata: PackageMetadata
    model_path: str = ""        # Path to extracted model weights
    is_extracted: bool = False


class ModelPackager:
    """Creates and reads model packages for distribution.

    Example:
        ```python
        packager = ModelPackager(packages_dir="~/.llamafarm/vision/packages")

        # Create a package from a trained model
        package = packager.create_package(
            model_path="/path/to/best.pt",
            metadata=PackageMetadata(
                model_id="yolov8n-custom",
                version=3,
                class_map={"person": 0, "car": 1},
                class_names=["person", "car"],
            ),
        )
        print(f"Package: {package.package_path}")

        # Read a package
        info = packager.read_package(package.package_path)
        print(f"Model: {info.metadata.model_id} v{info.metadata.version}")

        # Extract and load
        extracted = packager.extract_package(package.package_path)
        print(f"Model at: {extracted.model_path}")
        ```
    """

    def __init__(
        self,
        packages_dir: Path | str | None = None,
    ):
        self._packages_dir = (
            Path(packages_dir) if packages_dir
            else Path.home() / ".llamafarm" / "vision" / "packages"
        )
        self._packages_dir.mkdir(parents=True, exist_ok=True)

    def create_package(
        self,
        model_path: str,
        metadata: PackageMetadata,
        validation_metrics: dict[str, Any] | None = None,
        class_map: dict[str, int] | None = None,
    ) -> ModelPackage:
        """Create a distributable model package.

        Args:
            model_path: Path to model weights (.pt, .onnx)
            metadata: Package metadata
            validation_metrics: Optional validation gate metrics
            class_map: Optional class name -> ID mapping

        Returns:
            ModelPackage with path to the created archive
        """
        src = Path(model_path)
        if not src.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Set timestamp
        if not metadata.created_at:
            metadata.created_at = datetime.utcnow().isoformat()

        # Use provided class map or metadata's
        if class_map:
            metadata.class_map = class_map
            metadata.class_names = sorted(class_map.keys(), key=lambda k: class_map[k])

        # Build package name
        pkg_name = f"{metadata.model_id}_v{metadata.version}"
        pkg_path = self._packages_dir / f"{pkg_name}.tar.gz"

        # Create temp staging directory
        with tempfile.TemporaryDirectory(prefix="model_pkg_") as staging:
            staging_dir = Path(staging)

            # Copy model weights
            model_ext = src.suffix or ".pt"
            dst_model = staging_dir / f"model{model_ext}"
            shutil.copy2(str(src), str(dst_model))

            # Write metadata
            metadata_file = staging_dir / "metadata.json"
            metadata_file.write_text(json.dumps(metadata.to_dict(), indent=2))

            # Write class map
            class_map_file = staging_dir / "class_map.json"
            class_map_data = metadata.class_map or {
                name: i for i, name in enumerate(metadata.class_names)
            }
            class_map_file.write_text(json.dumps(class_map_data, indent=2))

            # Write validation metrics
            if validation_metrics:
                val_file = staging_dir / "validation_metrics.json"
                val_file.write_text(json.dumps(validation_metrics, indent=2))

            # Create tar.gz
            with tarfile.open(str(pkg_path), "w:gz") as tar:
                for item in staging_dir.iterdir():
                    tar.add(str(item), arcname=item.name)

        logger.info(
            f"Created package: {pkg_path} "
            f"({metadata.model_id} v{metadata.version}, "
            f"{pkg_path.stat().st_size / 1024:.1f} KB)"
        )

        return ModelPackage(
            package_path=str(pkg_path),
            metadata=metadata,
        )

    def read_package(self, package_path: str) -> ModelPackage:
        """Read package metadata without extracting model weights.

        Args:
            package_path: Path to .tar.gz package

        Returns:
            ModelPackage with metadata
        """
        pkg = Path(package_path)
        if not pkg.exists():
            raise FileNotFoundError(f"Package not found: {package_path}")

        with tarfile.open(str(pkg), "r:gz") as tar:
            # Read metadata.json
            try:
                meta_member = tar.getmember("metadata.json")
                meta_file = tar.extractfile(meta_member)
                if meta_file:
                    metadata = PackageMetadata.from_dict(
                        json.loads(meta_file.read().decode("utf-8"))
                    )
                else:
                    metadata = PackageMetadata(model_id="unknown")
            except KeyError:
                metadata = PackageMetadata(model_id="unknown")

        return ModelPackage(
            package_path=str(pkg),
            metadata=metadata,
        )

    def extract_package(
        self,
        package_path: str,
        extract_dir: Path | str | None = None,
    ) -> ModelPackage:
        """Extract a package and return paths to contents.

        Args:
            package_path: Path to .tar.gz package
            extract_dir: Where to extract (default: packages_dir/extracted/<name>)

        Returns:
            ModelPackage with model_path pointing to extracted weights
        """
        pkg = Path(package_path)
        if not pkg.exists():
            raise FileNotFoundError(f"Package not found: {package_path}")

        # Determine extract directory
        pkg_name = pkg.stem.replace(".tar", "")
        if extract_dir:
            dest = Path(extract_dir)
        else:
            dest = self._packages_dir / "extracted" / pkg_name
        dest.mkdir(parents=True, exist_ok=True)

        # Extract
        with tarfile.open(str(pkg), "r:gz") as tar:
            # Security: check for path traversal
            for member in tar.getmembers():
                member_path = Path(member.name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe path in archive: {member.name}")
            tar.extractall(str(dest), filter="data")

        # Read metadata
        metadata_file = dest / "metadata.json"
        if metadata_file.exists():
            metadata = PackageMetadata.from_dict(
                json.loads(metadata_file.read_text())
            )
        else:
            metadata = PackageMetadata(model_id="unknown")

        # Find model file
        model_path = ""
        for ext in [".pt", ".onnx", ".tflite", ".bin"]:
            candidate = dest / f"model{ext}"
            if candidate.exists():
                model_path = str(candidate)
                break

        return ModelPackage(
            package_path=str(pkg),
            metadata=metadata,
            model_path=model_path,
            is_extracted=True,
        )

    def list_packages(self) -> list[ModelPackage]:
        """List all available packages."""
        packages = []
        for pkg_file in sorted(self._packages_dir.glob("*.tar.gz")):
            try:
                packages.append(self.read_package(str(pkg_file)))
            except Exception as e:
                logger.warning(f"Failed to read package {pkg_file}: {e}")
        return packages

    def delete_package(self, package_path: str) -> bool:
        """Delete a package file."""
        pkg = Path(package_path)
        if pkg.exists():
            pkg.unlink()
            logger.info(f"Deleted package: {pkg}")
            return True
        return False
