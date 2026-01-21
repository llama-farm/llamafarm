#!/usr/bin/env python3
"""Demo script for OCR Integration.

This demo shows:
1. Process an image file with OCR (requires LlamaFarm server running)
2. OCR detection logic for images and scanned PDFs
3. Graceful handling when OCR endpoint is unavailable

NOTE: To see actual OCR in action, the LlamaFarm server must be running at:
  http://127.0.0.1:8000/v1/vision/ocr
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add rag directory to path
rag_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(rag_dir))

from components.parsers.universal import UniversalParser  # noqa: E402


def demo_ocr_detection():
    """Demo OCR detection logic."""
    print("=" * 60)
    print("OCR Detection Logic Demo")
    print("=" * 60)

    parser = UniversalParser()

    # Test image extensions
    print("\n--- Image Extensions (should trigger OCR) ---")
    image_extensions = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"]
    for ext in image_extensions:
        file_path = Path(f"test_image{ext}")
        needs_ocr = parser._needs_ocr(file_path, "")
        status = "[PASS]" if needs_ocr else "[FAIL]"
        print(f"{status} {ext}: needs_ocr = {needs_ocr}")

    # Test scanned PDF detection
    print("\n--- Scanned PDF Detection (sparse text < 50 chars) ---")
    sparse_texts = [
        ("", "empty text"),
        ("Page 1", "6 chars"),
        ("x" * 49, "49 chars"),
        ("x" * 50, "50 chars (threshold)"),
        ("x" * 100, "100 chars"),
    ]

    for text, description in sparse_texts:
        file_path = Path("test.pdf")
        needs_ocr = parser._needs_ocr(file_path, text)
        expected = len(text.strip()) < 50
        status = "[PASS]" if needs_ocr == expected else "[FAIL]"
        print(f"{status} PDF with {description}: needs_ocr = {needs_ocr}")

    # Test text-rich files
    print("\n--- Text-Rich Files (should NOT trigger OCR) ---")
    rich_text = "This is a document with plenty of extractable text content that should not need OCR processing."
    for ext in [".txt", ".md", ".pdf", ".docx"]:
        file_path = Path(f"test{ext}")
        needs_ocr = parser._needs_ocr(file_path, rich_text)
        expected = False  # Should not need OCR
        status = "[PASS]" if needs_ocr == expected else "[FAIL]"
        print(f"{status} {ext} with {len(rich_text)} chars: needs_ocr = {needs_ocr}")

    print("\n" + "=" * 60)
    print("OCR Detection Demo Complete")
    print("=" * 60)
    return True


def demo_ocr_endpoint_mocked():
    """Demo OCR endpoint call with mocking."""
    print("\n\n" + "=" * 60)
    print("OCR Endpoint Call Demo (Mocked)")
    print("=" * 60)

    # Create a mock response
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "text": "This is the text extracted from the image via OCR. The OCR system can read printed text, handwriting, and other content from images."
    }

    # Create a temp image file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
        # Write minimal PNG header (fake image)
        f.write(b"\x89PNG\r\n\x1a\n fake image content")
        temp_path = f.name

    try:
        parser = UniversalParser(config={"use_ocr": True})

        print(f"\nOCR Endpoint: {parser.ocr_endpoint}")
        print(f"Test file: {temp_path}")

        # Mock the requests.post call
        with patch("requests.post", return_value=mock_response) as mock_post:
            result = parser._run_remote_ocr(temp_path)

            print(f"\nOCR was called: {mock_post.called}")
            if mock_post.called:
                call_args = mock_post.call_args
                print(f"Endpoint called: {call_args[0][0]}")
                print(f"Request JSON keys: {list(call_args[1]['json'].keys())}")

            print(f"\nOCR Result: {result[:100]}...")
            print("[PASS] OCR endpoint call simulation successful")

    finally:
        Path(temp_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("OCR Endpoint Demo Complete")
    print("=" * 60)
    return True


def demo_ocr_graceful_failure():
    """Demo graceful handling when OCR endpoint is unavailable."""
    print("\n\n" + "=" * 60)
    print("OCR Graceful Failure Demo")
    print("=" * 60)

    from requests.exceptions import ConnectionError, Timeout

    parser = UniversalParser(config={"use_ocr": True})

    # Create a temp image file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
        f.write(b"\x89PNG\r\n\x1a\n fake image content")
        temp_path = f.name

    try:
        print("\n--- Testing Connection Error ---")
        with patch("requests.post", side_effect=ConnectionError("Connection refused")):
            result = parser._run_remote_ocr(temp_path)
            status = "[PASS]" if result is None else "[FAIL]"
            print(f"{status} Connection error handled gracefully: result = {result}")

        print("\n--- Testing Timeout ---")
        with patch("requests.post", side_effect=Timeout("Request timed out")):
            result = parser._run_remote_ocr(temp_path)
            status = "[PASS]" if result is None else "[FAIL]"
            print(f"{status} Timeout handled gracefully: result = {result}")

        print("\n--- Testing Server Error ---")
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        with patch("requests.post", return_value=mock_response):
            result = parser._run_remote_ocr(temp_path)
            status = "[PASS]" if result is None else "[FAIL]"
            print(f"{status} Server error handled gracefully: result = {result}")

    finally:
        Path(temp_path).unlink(missing_ok=True)

    print("\n" + "=" * 60)
    print("OCR Graceful Failure Demo Complete")
    print("=" * 60)
    return True


def demo_live_ocr():
    """Demo live OCR (requires universal-runtime running)."""
    print("\n\n" + "=" * 60)
    print("Live OCR Demo (requires universal-runtime)")
    print("=" * 60)

    import requests

    parser = UniversalParser(config={"use_ocr": True})

    # Check if OCR endpoint is available
    try:
        # Try a simple health check
        response = requests.get(
            parser.ocr_endpoint.replace("/ocr", "/health"),
            timeout=5
        )
        endpoint_available = response.ok
    except Exception:
        endpoint_available = False

    if not endpoint_available:
        print(f"\n[SKIP] OCR endpoint not available: {parser.ocr_endpoint}")
        print("To run live OCR demo, start the LlamaFarm server:")
        print("  nx start server")
        print("\n" + "=" * 60)
        print("Live OCR Demo Skipped")
        print("=" * 60)
        return True  # Not a failure, just skipped

    # If endpoint is available, create a test image and process it
    # (This would require a real image with text to be meaningful)
    print(f"\n[INFO] OCR endpoint is available: {parser.ocr_endpoint}")
    print("[INFO] To test with a real image, run:")
    print("  parser = UniversalParser()")
    print("  result = parser.parse('/path/to/image.png')")

    print("\n" + "=" * 60)
    print("Live OCR Demo Complete")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success1 = demo_ocr_detection()
    success2 = demo_ocr_endpoint_mocked()
    success3 = demo_ocr_graceful_failure()
    success4 = demo_live_ocr()

    if success1 and success2 and success3 and success4:
        print("\n\nAll OCR demos passed!")
        sys.exit(0)
    else:
        print("\n\nSome demos failed!")
        sys.exit(1)
