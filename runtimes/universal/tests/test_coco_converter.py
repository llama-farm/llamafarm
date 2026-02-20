"""Tests for COCO JSON → YOLO format converter."""

import json
from pathlib import Path

import pytest

from vision_training.coco_converter import (
    _validate_coco,
    convert_coco_to_yolo,
    is_coco_format,
)


def _make_coco(tmp_path: Path, images: list[dict] | None = None,
               categories: list[dict] | None = None,
               annotations: list[dict] | None = None,
               create_images: bool = True) -> tuple[Path, Path]:
    """Create a minimal COCO dataset in tmp_path. Returns (json_path, images_dir)."""
    if categories is None:
        categories = [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
    if images is None:
        images = [
            {"id": 1, "file_name": "img_001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img_002.jpg", "width": 640, "height": 480},
        ]
    if annotations is None:
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 50, 50]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [200, 200, 60, 40]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [50, 50, 100, 100]},
        ]

    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    if create_images:
        for img in images:
            (images_dir / Path(img["file_name"]).name).write_bytes(b"\xff\xd8\xff\xe0fake")

    coco = {"images": images, "annotations": annotations, "categories": categories}
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(coco))
    return json_path, images_dir


class TestConvertCocoToYolo:
    """Tests for convert_coco_to_yolo()."""

    def test_basic_conversion(self, tmp_path: Path):
        json_path, images_dir = _make_coco(tmp_path)
        out = tmp_path / "yolo_out"
        data_yaml = convert_coco_to_yolo(json_path, out, images_dir)

        assert data_yaml.exists()
        assert (out / "images" / "train").is_dir()
        assert (out / "images" / "val").is_dir()
        assert (out / "labels" / "train").is_dir()
        assert (out / "labels" / "val").is_dir()

        # Check at least one label file exists
        labels = list((out / "labels" / "train").glob("*.txt")) + \
                 list((out / "labels" / "val").glob("*.txt"))
        assert len(labels) == 2

    def test_bbox_normalization(self, tmp_path: Path):
        """COCO [x,y,w,h] absolute → YOLO [cx,cy,w,h] normalized."""
        images = [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}]
        annotations = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}]
        categories = [{"id": 1, "name": "obj"}]
        json_path, images_dir = _make_coco(tmp_path, images, categories, annotations)
        out = tmp_path / "out"
        convert_coco_to_yolo(json_path, out, images_dir, split_ratio=1.0)

        label = out / "labels" / "train" / "a.txt"
        assert label.exists()
        parts = label.read_text().strip().split()
        assert parts[0] == "0"  # class index
        assert abs(float(parts[1]) - 0.25) < 0.01  # x_center = (10+15)/100
        assert abs(float(parts[2]) - 0.40) < 0.01  # y_center = (20+20)/100
        assert abs(float(parts[3]) - 0.30) < 0.01  # width
        assert abs(float(parts[4]) - 0.40) < 0.01  # height

    def test_path_traversal_rejected(self, tmp_path: Path):
        """file_name with '..' should be skipped."""
        images = [{"id": 1, "file_name": "../../etc/passwd", "width": 640, "height": 480}]
        json_path, images_dir = _make_coco(tmp_path, images, annotations=[], create_images=False)
        out = tmp_path / "out"
        convert_coco_to_yolo(json_path, out, images_dir)
        # Should skip the traversal filename — no images copied
        all_images = list((out / "images" / "train").glob("*")) + \
                     list((out / "images" / "val").glob("*"))
        assert len(all_images) == 0

    def test_absolute_path_rejected(self, tmp_path: Path):
        """Absolute path in file_name should be skipped (contains '/')."""
        images = [{"id": 1, "file_name": "/etc/passwd", "width": 640, "height": 480}]
        json_path, images_dir = _make_coco(tmp_path, images, annotations=[], create_images=False)
        out = tmp_path / "out"
        convert_coco_to_yolo(json_path, out, images_dir)
        all_images = list((out / "images" / "train").glob("*")) + \
                     list((out / "images" / "val").glob("*"))
        assert len(all_images) == 0

    def test_missing_image_skipped(self, tmp_path: Path):
        json_path, images_dir = _make_coco(tmp_path, create_images=False)
        out = tmp_path / "out"
        convert_coco_to_yolo(json_path, out, images_dir)
        all_images = list((out / "images" / "train").glob("*")) + \
                     list((out / "images" / "val").glob("*"))
        assert len(all_images) == 0

    def test_basename_collision_detected(self, tmp_path: Path):
        """Same basename from different subdirs → second is skipped."""
        images = [
            {"id": 1, "file_name": "subdir_a/photo.jpg", "width": 100, "height": 100},
            {"id": 2, "file_name": "subdir_b/photo.jpg", "width": 100, "height": 100},
        ]
        categories = [{"id": 1, "name": "x"}]
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 20]},
        ]
        images_dir = tmp_path / "images"
        for sub in ["subdir_a", "subdir_b"]:
            (images_dir / sub).mkdir(parents=True)
            (images_dir / sub / "photo.jpg").write_bytes(b"\xff\xd8fake")

        json_path = tmp_path / "ann.json"
        json_path.write_text(json.dumps({
            "images": images, "annotations": annotations, "categories": categories
        }))
        out = tmp_path / "out"
        convert_coco_to_yolo(json_path, out, images_dir)
        # Only first should be converted, second skipped as collision
        all_images = list((out / "images" / "train").glob("*")) + \
                     list((out / "images" / "val").glob("*"))
        assert len(all_images) == 1

    def test_zero_dimensions_skipped(self, tmp_path: Path):
        images = [{"id": 1, "file_name": "a.jpg", "width": 0, "height": 480}]
        json_path, images_dir = _make_coco(tmp_path, images, annotations=[])
        out = tmp_path / "out"
        convert_coco_to_yolo(json_path, out, images_dir)
        labels = list((out / "labels" / "train").glob("*")) + \
                 list((out / "labels" / "val").glob("*"))
        assert len(labels) == 0

    def test_split_ratio(self, tmp_path: Path):
        """80/20 split with 5 images → 4 train, 1 val."""
        images = [{"id": i, "file_name": f"img_{i}.jpg", "width": 100, "height": 100}
                  for i in range(1, 6)]
        annotations = [{"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20]}
                       for i in range(1, 6)]
        categories = [{"id": 1, "name": "x"}]
        json_path, images_dir = _make_coco(tmp_path, images, categories, annotations)
        out = tmp_path / "out"
        convert_coco_to_yolo(json_path, out, images_dir, split_ratio=0.8)

        train = list((out / "images" / "train").glob("*"))
        val = list((out / "images" / "val").glob("*"))
        assert len(train) == 4
        assert len(val) == 1

    def test_data_yaml_content(self, tmp_path: Path):
        import yaml
        json_path, images_dir = _make_coco(tmp_path)
        out = tmp_path / "out"
        data_yaml = convert_coco_to_yolo(json_path, out, images_dir)
        cfg = yaml.safe_load(data_yaml.read_text())
        assert cfg["nc"] == 2
        assert cfg["train"] == "images/train"
        assert cfg["val"] == "images/val"
        assert 0 in cfg["names"]

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            convert_coco_to_yolo(tmp_path / "nope.json", tmp_path / "out")

    def test_invalid_coco_json(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"foo": "bar"}))
        with pytest.raises(ValueError, match="missing 'images'"):
            convert_coco_to_yolo(bad, tmp_path / "out")


class TestIsCocoFormat:
    def test_valid(self, tmp_path: Path):
        p = tmp_path / "coco.json"
        p.write_text(json.dumps({"images": [], "annotations": [], "categories": []}))
        assert is_coco_format(p) is True

    def test_not_json(self, tmp_path: Path):
        p = tmp_path / "readme.txt"
        p.write_text("not json")
        assert is_coco_format(p) is False

    def test_missing_keys(self, tmp_path: Path):
        p = tmp_path / "partial.json"
        p.write_text(json.dumps({"images": []}))
        assert is_coco_format(p) is False

    def test_nonexistent(self, tmp_path: Path):
        assert is_coco_format(tmp_path / "nope.json") is False


class TestValidateCoco:
    def test_missing_images(self):
        with pytest.raises(ValueError, match="missing 'images'"):
            _validate_coco({"categories": []})

    def test_images_not_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            _validate_coco({"images": "not a list", "categories": []})

    def test_category_missing_id(self):
        with pytest.raises(ValueError, match="each category needs"):
            _validate_coco({"images": [], "categories": [{"name": "x"}]})
