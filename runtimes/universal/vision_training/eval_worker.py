"""Subprocess eval worker — runs model evaluation in an isolated process.

Invoked by the eval router via subprocess. Writes results to a JSON file.
If this process OOMs or crashes, the runtime stays alive.

Usage:
    python -m vision_training.eval_worker \
        --model /path/to/model.pt \
        --dataset /path/to/data.yaml \
        --output /tmp/eval_result.json \
        [--model-name my-model] \
        [--imgsz 640] \
        [--batch 16] \
        [--weights '{"mAP50_95": 0.35, ...}']
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure runtime modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_worker")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vision model eval worker")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--dataset", required=True, help="Path to dataset YAML")
    parser.add_argument("--output", required=True, help="Path to write JSON result")
    parser.add_argument("--model-name", default=None, help="Human-readable model name")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--weights", default=None, help="JSON scoring weights")
    parser.add_argument("--db-path", default=None, help="Path to eval.db")
    args = parser.parse_args()

    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
        except json.JSONDecodeError:
            logger.warning(f"Invalid weights JSON: {args.weights}")

    # Validate inputs
    model_path = Path(args.model)
    if not model_path.exists():
        _write_error(args.output, f"Model not found: {args.model}")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        _write_error(args.output, f"Dataset not found: {args.dataset}")
        sys.exit(1)

    try:
        from models.eval_model import EvalDB, ModelEvaluator

        db_path = Path(args.db_path) if args.db_path else None
        db = EvalDB(db_path)
        evaluator = ModelEvaluator(db=db)

        # Run synchronously — this is the subprocess
        result = evaluator._evaluate_sync(
            model_path=args.model,
            data_yaml=args.dataset,
            model_name=args.model_name,
            imgsz=args.imgsz,
            batch=args.batch,
            weights=weights,
        )

        # Write result
        output = {"status": "completed", "result": result.to_dict()}
        Path(args.output).write_text(json.dumps(output))
        logger.info(f"Eval complete: score={result.score:.4f} mAP50={result.mAP50:.4f}")

    except Exception as e:
        logger.error(f"Eval failed: {e}", exc_info=True)
        _write_error(args.output, str(e))
        sys.exit(1)


def _write_error(output_path: str, error: str) -> None:
    """Write error result to output file."""
    Path(output_path).write_text(json.dumps({"status": "failed", "error": error}))


if __name__ == "__main__":
    main()
