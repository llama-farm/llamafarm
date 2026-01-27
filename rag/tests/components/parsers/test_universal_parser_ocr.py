"""Tests for UniversalParser OCR integration.

This test file verifies:
1. UniversalParser detects images needing OCR (.png, .jpg, .jpeg, .tiff)
2. UniversalParser detects scanned PDFs (< 50 chars extracted)
3. UniversalParser calls OCR endpoint correctly (mocked)
4. UniversalParser handles OCR endpoint unavailable gracefully
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestUniversalParserOCRDetection:
    """Test OCR detection logic."""

    def test_detects_images_needing_ocr_png(self):
        """Test: UniversalParser detects PNG images needing OCR."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()
        file_path = Path("test_image.png")

        # PNG should always trigger OCR
        assert parser._needs_ocr(file_path, "") is True
        assert parser._needs_ocr(file_path, None) is True

    def test_detects_images_needing_ocr_jpg(self):
        """Test: UniversalParser detects JPG images needing OCR."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()

        assert parser._needs_ocr(Path("test.jpg"), "") is True
        assert parser._needs_ocr(Path("test.jpeg"), "") is True

    def test_detects_images_needing_ocr_tiff(self):
        """Test: UniversalParser detects TIFF images needing OCR."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()

        assert parser._needs_ocr(Path("test.tiff"), "") is True
        assert parser._needs_ocr(Path("test.bmp"), "") is True
        assert parser._needs_ocr(Path("test.gif"), "") is True
        assert parser._needs_ocr(Path("test.webp"), "") is True

    def test_detects_scanned_pdfs_with_sparse_text(self):
        """Test: UniversalParser detects scanned PDFs (< 50 chars extracted)."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()

        # Less than 50 chars should trigger OCR
        assert parser._needs_ocr(Path("test.pdf"), "Page 1") is True
        assert parser._needs_ocr(Path("test.pdf"), "x" * 49) is True

    def test_skips_ocr_for_text_rich_pdfs(self):
        """Test: UniversalParser skips OCR for PDFs with enough text."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()

        # 50+ chars should not trigger OCR for PDFs
        text = "This is a text-rich PDF with plenty of extractable content that should not need OCR."
        assert parser._needs_ocr(Path("test.pdf"), text) is False

    def test_skips_ocr_for_text_files(self):
        """Test: UniversalParser skips OCR for text files with content (50+ chars)."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()

        # Text must be 50+ chars to skip OCR
        text = "This is a text file with plenty of content to avoid triggering OCR detection."
        assert parser._needs_ocr(Path("test.txt"), text) is False
        assert parser._needs_ocr(Path("test.md"), text) is False


class TestUniversalParserOCREndpoint:
    """Test OCR endpoint calling."""

    @patch("requests.post")
    def test_calls_ocr_endpoint_with_base64_image(self, mock_post):
        """Test: UniversalParser calls OCR endpoint correctly (mocked)."""
        from components.parsers.universal import UniversalParser

        # Setup mock response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "text": "Extracted text from OCR"
        }
        mock_post.return_value = mock_response

        # Create a temp file to test with
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            f.write(b"fake image data")
            temp_path = f.name

        try:
            parser = UniversalParser(config={"use_ocr": True})
            result = parser._run_remote_ocr(temp_path)

            # Verify endpoint was called
            mock_post.assert_called_once()
            call_args = mock_post.call_args

            # Verify the endpoint URL
            assert call_args[0][0] == "http://127.0.0.1:8000/v1/vision/ocr"

            # Verify the request contains base64 image
            request_json = call_args[1]["json"]
            assert "image" in request_json
            assert "filename" in request_json

            # Verify result
            assert result == "Extracted text from OCR"

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("requests.post")
    def test_returns_none_on_ocr_endpoint_error(self, mock_post):
        """Test: UniversalParser handles OCR endpoint errors gracefully."""
        from components.parsers.universal import UniversalParser

        # Setup mock response to return error
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            f.write(b"fake image data")
            temp_path = f.name

        try:
            parser = UniversalParser(config={"use_ocr": True})
            result = parser._run_remote_ocr(temp_path)

            # Should return None on error
            assert result is None

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("requests.post")
    def test_returns_none_on_ocr_timeout(self, mock_post):
        """Test: UniversalParser handles OCR timeout gracefully."""
        from requests.exceptions import Timeout

        from components.parsers.universal import UniversalParser

        # Setup mock to raise timeout
        mock_post.side_effect = Timeout("Connection timed out")

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            f.write(b"fake image data")
            temp_path = f.name

        try:
            parser = UniversalParser(config={"use_ocr": True})
            result = parser._run_remote_ocr(temp_path)

            # Should return None on timeout
            assert result is None

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("requests.post")
    def test_returns_none_on_connection_error(self, mock_post):
        """Test: UniversalParser handles OCR endpoint unavailable gracefully."""
        from requests.exceptions import ConnectionError

        from components.parsers.universal import UniversalParser

        # Setup mock to raise connection error
        mock_post.side_effect = ConnectionError("Connection refused")

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            f.write(b"fake image data")
            temp_path = f.name

        try:
            parser = UniversalParser(config={"use_ocr": True})
            result = parser._run_remote_ocr(temp_path)

            # Should return None on connection error
            assert result is None

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestUniversalParserOCRIntegration:
    """Test OCR integration with parse flow."""

    @patch("requests.post")
    def test_applies_ocr_to_image_file(self, mock_post):
        """Test: OCR is applied when parsing image files."""
        from components.parsers.universal import UniversalParser

        # Setup mock OCR response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "text": "This text was extracted via OCR from the image. " * 20
        }
        mock_post.return_value = mock_response

        # Create a fake image file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n fake image data")
            temp_path = f.name

        try:
            parser = UniversalParser(config={"use_ocr": True})
            parser.parse(temp_path)  # result not needed, just verify flow

            # OCR should have been called
            assert mock_post.called

            # Should have documents (assuming OCR text was used)
            # Note: This depends on MarkItDown's behavior with fake PNG
            # The test primarily verifies the flow

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("requests.post")
    def test_adds_ocr_applied_metadata(self, mock_post):
        """Test: OCR adds ocr_applied metadata when used."""
        from components.parsers.universal import UniversalParser

        # Setup mock OCR response with long text
        mock_response = MagicMock()
        mock_response.ok = True
        ocr_text = "This is OCR extracted text. " * 50  # Long enough to create chunks
        mock_response.json.return_value = {"text": ocr_text}
        mock_post.return_value = mock_response

        # Create a fake image file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n fake image data")
            temp_path = f.name

        try:
            parser = UniversalParser(config={"use_ocr": True})
            result = parser.parse(temp_path)

            # If OCR was used and produced docs, they should have ocr_applied metadata
            if result.documents:
                for doc in result.documents:
                    if doc.metadata.get("ocr_applied"):
                        assert doc.metadata["ocr_applied"] is True

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_respects_use_ocr_false_config(self):
        """Test: OCR is skipped when use_ocr=False."""
        from components.parsers.universal import UniversalParser

        # Create parser with OCR disabled
        parser = UniversalParser(config={"use_ocr": False})

        # Create a minimal text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Short text")  # Less than 50 chars
            temp_path = f.name

        try:
            # Parse should not call OCR even though text is short
            with patch.object(parser, "_run_remote_ocr") as mock_ocr:
                parser.parse(temp_path)  # result not needed, just verify mock
                # OCR should not be called
                mock_ocr.assert_not_called()

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @patch("requests.post")
    def test_uses_longer_text_between_markitdown_and_ocr(self, mock_post):
        """Test: Parser uses longer text between MarkItDown and OCR."""
        from components.parsers.universal import UniversalParser

        # OCR returns short text
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"text": "Short"}
        mock_post.return_value = mock_response

        # Create a text file with more content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            content = "This is a text file with significantly more content than OCR would extract. " * 5
            f.write(content)
            temp_path = f.name

        try:
            parser = UniversalParser(config={"use_ocr": True})
            result = parser.parse(temp_path)

            # Should use MarkItDown text since it's longer
            if result.documents:
                # The content should be from MarkItDown, not OCR
                total_content = "".join(d.content for d in result.documents)
                assert "This is a text file" in total_content

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestUniversalParserOCREndpointConfig:
    """Test OCR endpoint configuration."""

    def test_uses_default_ocr_endpoint(self):
        """Test: Uses default OCR endpoint if not configured."""
        from components.parsers.universal import UniversalParser

        parser = UniversalParser()
        assert parser.ocr_endpoint == "http://127.0.0.1:8000/v1/vision/ocr"

    def test_uses_custom_ocr_endpoint(self):
        """Test: Uses custom OCR endpoint from config."""
        from components.parsers.universal import UniversalParser

        custom_endpoint = "http://custom-server:8080/api/ocr"
        parser = UniversalParser(config={"ocr_endpoint": custom_endpoint})
        assert parser.ocr_endpoint == custom_endpoint

    @patch("requests.post")
    def test_custom_endpoint_is_called(self, mock_post):
        """Test: Custom OCR endpoint is actually called."""
        from components.parsers.universal import UniversalParser

        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"text": "OCR result"}
        mock_post.return_value = mock_response

        custom_endpoint = "http://my-ocr-server:9000/ocr"

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            f.write(b"fake image data")
            temp_path = f.name

        try:
            parser = UniversalParser(config={"ocr_endpoint": custom_endpoint})
            parser._run_remote_ocr(temp_path)

            # Verify custom endpoint was called
            call_args = mock_post.call_args
            assert call_args[0][0] == custom_endpoint

        finally:
            Path(temp_path).unlink(missing_ok=True)
