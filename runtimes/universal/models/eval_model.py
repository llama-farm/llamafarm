"""Vision model evaluator — runs ultralytics .val() with size-based recall."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVAL_DB_PATH = Path.home() / ".llamafarm" / "models" / "vision" / "eval.db"

# Object size thresholds (px² area)
SMALL_THRESH = 32 * 32    # < 1024 px²
MEDIUM_THRESH = 96 * 96   # < 9216 px²

# Default scoring weights
DEFAULT_WEIGHTS = {
    "mAP50_95": 0.35,
    "mAP50": 0.20,
    "small_object_recall": 0.20,
    "f1": 0.15,
    "speed": 0.10,
}


@dataclass
class EvalResult:
    """Evaluation result with metrics and composite score."""

    model_name: str
    model_path: str
    dataset: str
    timestamp: str = ""
    # Core COCO metrics
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    # Per-class mAP (class_name → mAP50)
    per_class: dict[str, float] = field(default_factory=dict)
    # Speed (ms)
    inference_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    # Model size
    model_size_mb: float = 0.0
    # Size-based recall
    small_object_recall: float = 0.0
    medium_object_recall: float = 0.0
    large_object_recall: float = 0.0
    # Composite score
    score: float = 0.0
    # DB row id (set after save)
    eval_id: int | None = None

    def compute_score(self, weights: dict[str, float] | None = None) -> float:
        """Compute weighted composite score."""
        w = weights or DEFAULT_WEIGHTS
        # Speed score: 1.0 at ≤50ms, 0.0 at ≥500ms
        speed_score = max(0.0, min(1.0, 1.0 - (self.inference_ms - 50) / 450))
        self.score = (
            w.get("mAP50_95", 0.35) * self.mAP50_95
            + w.get("mAP50", 0.20) * self.mAP50
            + w.get("small_object_recall", 0.20) * self.small_object_recall
            + w.get("f1", 0.15) * self.f1
            + w.get("speed", 0.10) * speed_score
        )
        return self.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "dataset": self.dataset,
            "timestamp": self.timestamp,
            "mAP50": self.mAP50,
            "mAP50_95": self.mAP50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "per_class": self.per_class,
            "inference_ms": self.inference_ms,
            "preprocess_ms": self.preprocess_ms,
            "postprocess_ms": self.postprocess_ms,
            "model_size_mb": self.model_size_mb,
            "small_object_recall": self.small_object_recall,
            "medium_object_recall": self.medium_object_recall,
            "large_object_recall": self.large_object_recall,
            "score": self.score,
            "eval_id": self.eval_id,
        }


class EvalDB:
    """SQLite store for evaluation results and promotions."""

    def __init__(self, db_path: Path | None = None):
        self._path = db_path or EVAL_DB_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        self._init_tables()

    def _init_tables(self) -> None:
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_path TEXT,
                dataset TEXT,
                timestamp TEXT,
                mAP50 REAL, mAP50_95 REAL,
                precision_val REAL, recall_val REAL, f1 REAL,
                per_class TEXT,
                inference_ms REAL, preprocess_ms REAL, postprocess_ms REAL,
                model_size_mb REAL,
                small_object_recall REAL,
                medium_object_recall REAL,
                large_object_recall REAL,
                score REAL
            );
            CREATE TABLE IF NOT EXISTS promotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                eval_id INTEGER REFERENCES eval_results(id),
                promoted_at TEXT,
                reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_eval_score ON eval_results(score DESC);
            CREATE INDEX IF NOT EXISTS idx_eval_dataset ON eval_results(dataset);
        """)
        self._db.commit()

    def save(self, result: EvalResult) -> int:
        with self._lock:
            cur = self._db.execute("""
                INSERT INTO eval_results (
                    model_name, model_path, dataset, timestamp,
                    mAP50, mAP50_95, precision_val, recall_val, f1, per_class,
                    inference_ms, preprocess_ms, postprocess_ms, model_size_mb,
                    small_object_recall, medium_object_recall, large_object_recall,
                    score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.model_name, result.model_path, result.dataset, result.timestamp,
                result.mAP50, result.mAP50_95, result.precision, result.recall, result.f1,
                json.dumps(result.per_class),
                result.inference_ms, result.preprocess_ms, result.postprocess_ms,
                result.model_size_mb,
                result.small_object_recall, result.medium_object_recall, result.large_object_recall,
                result.score,
            ))
            self._db.commit()
            result.eval_id = cur.lastrowid
            return cur.lastrowid

    def leaderboard(self, dataset: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        query = "SELECT * FROM eval_results"
        params: list[Any] = []
        if dataset:
            query += " WHERE dataset = ?"
            params.append(dataset)
        query += " ORDER BY score DESC LIMIT ?"
        params.append(limit)
        rows = self._db.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def best_model(self, dataset: str | None = None) -> dict[str, Any] | None:
        lb = self.leaderboard(dataset, limit=1)
        return lb[0] if lb else None

    def record_promotion(self, model_name: str, eval_id: int, reason: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO promotions (model_name, eval_id, promoted_at, reason) VALUES (?, ?, ?, ?)",
                (model_name, eval_id, datetime.now().isoformat(), reason),
            )
            self._db.commit()

    def model_history(self, model_name: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM eval_results WHERE model_name = ? ORDER BY timestamp DESC LIMIT ?",
            (model_name, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        # Rename precision_val/recall_val back to precision/recall for API
        d["precision"] = d.pop("precision_val", 0.0)
        d["recall"] = d.pop("recall_val", 0.0)
        if d.get("per_class"):
            try:
                d["per_class"] = json.loads(d["per_class"])
            except (json.JSONDecodeError, TypeError):
                d["per_class"] = {}
        return d


class ModelEvaluator:
    """Evaluate YOLO models using ultralytics .val()."""

    def __init__(self, db: EvalDB | None = None):
        self.db = db or EvalDB()

    async def evaluate(
        self,
        model_path: str,
        data_yaml: str,
        model_name: str | None = None,
        imgsz: int = 640,
        batch: int = 16,
        weights: dict[str, float] | None = None,
    ) -> EvalResult:
        """Run ultralytics .val() and return metrics."""
        import asyncio

        return await asyncio.to_thread(
            self._evaluate_sync, model_path, data_yaml, model_name, imgsz, batch, weights
        )

    def _evaluate_sync(
        self,
        model_path: str,
        data_yaml: str,
        model_name: str | None,
        imgsz: int,
        batch: int,
        weights: dict[str, float] | None,
    ) -> EvalResult:
        from ultralytics import YOLO

        name = model_name or Path(model_path).stem
        logger.info(f"Evaluating {name} on {data_yaml} (imgsz={imgsz}, batch={batch})")

        model = YOLO(model_path)
        metrics = model.val(data=data_yaml, imgsz=imgsz, batch=batch, verbose=False, plots=False)

        p = float(metrics.box.mp)
        r = float(metrics.box.mr)
        f1 = 2 * p * r / max(p + r, 1e-6)

        result = EvalResult(
            model_name=name,
            model_path=str(model_path),
            dataset=str(data_yaml),
            timestamp=datetime.now().isoformat(),
            mAP50=float(metrics.box.map50),
            mAP50_95=float(metrics.box.map),
            precision=p,
            recall=r,
            f1=f1,
            model_size_mb=Path(model_path).stat().st_size / (1024 * 1024),
        )

        # Per-class metrics
        if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
            names = getattr(metrics, "names", {})
            for i, m in enumerate(metrics.box.maps):
                cls_name = names.get(i, f"class_{i}")
                result.per_class[cls_name] = float(m)

        # Speed
        if hasattr(metrics, "speed") and isinstance(metrics.speed, dict):
            result.preprocess_ms = metrics.speed.get("preprocess", 0.0)
            result.inference_ms = metrics.speed.get("inference", 0.0)
            result.postprocess_ms = metrics.speed.get("postprocess", 0.0)

        # Size-based recall from validation predictions
        self._compute_size_recall(model, data_yaml, imgsz, result)

        result.compute_score(weights)
        self.db.save(result)
        logger.info(f"Eval complete: {name} score={result.score:.4f} mAP50={result.mAP50:.4f}")
        return result

    def _compute_size_recall(
        self, model: Any, data_yaml: str, imgsz: int, result: EvalResult
    ) -> None:
        """Compute actual size-based recall from validation predictions."""
        try:
            import yaml
            from PIL import Image

            with open(data_yaml) as f:
                cfg = yaml.safe_load(f)

            ds_path = Path(cfg.get("path", ""))
            val_rel = cfg.get("val", "images/val")
            val_dir = ds_path / val_rel if not Path(val_rel).is_absolute() else Path(val_rel)
            labels_dir = val_dir.parent.parent / "labels" / Path(val_rel).name

            if not labels_dir.exists():
                logger.debug(f"Labels dir not found: {labels_dir}, skipping size recall")
                return

            small_total = small_det = 0
            medium_total = medium_det = 0
            large_total = large_det = 0

            label_files = sorted(labels_dir.glob("*.txt"))[:200]  # Sample up to 200

            for label_file in label_files:
                img_path = val_dir / label_file.with_suffix(".jpg").name
                if not img_path.exists():
                    img_path = val_dir / label_file.with_suffix(".png").name
                if not img_path.exists():
                    continue

                img = Image.open(img_path)
                w, h = img.size

                # Count GT objects by size
                gt_by_size: dict[str, int] = {"small": 0, "medium": 0, "large": 0}
                for line in label_file.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    bw, bh = float(parts[3]) * w, float(parts[4]) * h
                    area = bw * bh
                    if area < SMALL_THRESH:
                        gt_by_size["small"] += 1
                        small_total += 1
                    elif area < MEDIUM_THRESH:
                        gt_by_size["medium"] += 1
                        medium_total += 1
                    else:
                        gt_by_size["large"] += 1
                        large_total += 1

                # Run inference on this image to count detections by size
                preds = model.predict(str(img_path), imgsz=imgsz, verbose=False, conf=0.25)
                if preds and preds[0].boxes is not None:
                    boxes = preds[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        area = (x2 - x1) * (y2 - y1)
                        if area < SMALL_THRESH:
                            small_det += 1
                        elif area < MEDIUM_THRESH:
                            medium_det += 1
                        else:
                            large_det += 1

            # Recall = min(detected, total) / total (capped at 1.0)
            result.small_object_recall = min(small_det, small_total) / max(small_total, 1)
            result.medium_object_recall = min(medium_det, medium_total) / max(medium_total, 1)
            result.large_object_recall = min(large_det, large_total) / max(large_total, 1)

            logger.info(
                f"Size recall: small={result.small_object_recall:.3f} ({small_det}/{small_total}), "
                f"med={result.medium_object_recall:.3f} ({medium_det}/{medium_total}), "
                f"large={result.large_object_recall:.3f} ({large_det}/{large_total})"
            )
        except Exception as e:
            logger.warning(f"Size recall computation failed: {e}")

    def compare(self, result_a: EvalResult, result_b: EvalResult) -> dict[str, Any]:
        """Head-to-head comparison of two eval results."""
        comparison: dict[str, Any] = {
            "model_a": result_a.model_name,
            "model_b": result_b.model_name,
            "winner": result_a.model_name if result_a.score > result_b.score else result_b.model_name,
            "score_diff": abs(result_a.score - result_b.score),
            "metrics": {},
        }
        for metric in ("mAP50", "mAP50_95", "precision", "recall", "f1",
                        "inference_ms", "small_object_recall", "score"):
            va = getattr(result_a, metric, 0.0)
            vb = getattr(result_b, metric, 0.0)
            # Lower is better for inference_ms
            better = "a" if (va > vb if metric != "inference_ms" else va < vb) else "b"
            comparison["metrics"][metric] = {"a": va, "b": vb, "diff": va - vb, "better": better}
        return comparison
