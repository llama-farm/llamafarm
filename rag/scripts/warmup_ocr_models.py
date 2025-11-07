#!/usr/bin/env python3
"""Download and cache all OCR models for faster startup.

This script initializes all OCR preprocessors to trigger model downloads.
Run this once after installing OCR dependencies to cache models locally.

Usage:
    uv run python scripts/warmup_ocr_models.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.scripts.warmup_ocr_models")


def warmup_paddleocr():
    """Download PaddleOCR and PPStructureV3 models."""
    try:
        from paddleocr import PaddleOCR, PPStructureV3

        logger.info("Downloading PaddleOCR models...")
        ocr = PaddleOCR(lang="en")
        logger.info("✓ PaddleOCR models cached")

        logger.info("Downloading PPStructureV3 models (this may take 5-10 minutes)...")
        logger.info(
            "Models being downloaded: layout detection, OCR, table extraction, formula recognition, chart parsing"
        )
        structure = PPStructureV3(lang="en")
        logger.info("✓ PPStructureV3 models cached")

        return True
    except ImportError as e:
        logger.warning(f"PaddleOCR not installed: {e}")
        logger.info("Install with: uv pip install paddleocr paddlepaddle paddlex[ocr]")
        return False
    except Exception as e:
        logger.error(f"Failed to warm up PaddleOCR: {e}", exc_info=True)
        return False


def warmup_tesseract():
    """Check Tesseract availability."""
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        logger.info("✓ Tesseract OCR available")
        return True
    except Exception:
        logger.warning("Tesseract OCR not available")
        logger.info(
            "Install with: brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)"
        )
        return False


def main():
    """Warm up all OCR models."""
    logger.info("=== OCR Model Warmup ===")
    logger.info(
        "This script downloads and caches OCR models for faster subsequent use."
    )
    logger.info("")

    success_count = 0
    total_count = 0

    # Warm up PaddleOCR (primary OCR engine)
    total_count += 1
    if warmup_paddleocr():
        success_count += 1

    # Check Tesseract (fallback OCR engine)
    total_count += 1
    if warmup_tesseract():
        success_count += 1

    # Summary
    logger.info("")
    logger.info(f"=== Warmup Complete: {success_count}/{total_count} engines ready ===")

    if success_count == total_count:
        logger.info(
            "All OCR models cached! Subsequent runs will be much faster (~5-10 seconds)."
        )
        logger.info(f"Models cached in: ~/.paddlex/official_models/")
        return 0
    else:
        logger.warning(
            "Some OCR engines are not available. Install missing dependencies."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
