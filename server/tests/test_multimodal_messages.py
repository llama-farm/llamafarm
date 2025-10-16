"""Tests for multimodal message handling in the server."""

import base64

import pytest
from pydantic import ValidationError
from server.api.routers.inference.models import (
    AudioURLContent,
    ChatMessage,
    ChatRequest,
    ImageURLContent,
    MessageContentPart,
    VideoURLContent,
)


class TestImageURLContent:
    """Test ImageURLContent model."""

    def test_valid_image_url(self):
        """Test creating valid image URL content."""
        content = ImageURLContent(url="data:image/png;base64,iVBORw0KGgo=")
        assert content.url == "data:image/png;base64,iVBORw0KGgo="
        assert content.detail == "auto"

    def test_custom_detail_level(self):
        """Test custom detail levels."""
        for detail in ["auto", "low", "high"]:
            content = ImageURLContent(url="http://example.com/image.png", detail=detail)
            assert content.detail == detail

    def test_http_url(self):
        """Test HTTP(S) URLs."""
        content = ImageURLContent(url="https://example.com/image.jpg")
        assert content.url == "https://example.com/image.jpg"


class TestAudioURLContent:
    """Test AudioURLContent model."""

    def test_valid_audio_url(self):
        """Test creating valid audio URL content."""
        content = AudioURLContent(url="data:audio/mpeg;base64,SUQzBAAAAAA=")
        assert content.url == "data:audio/mpeg;base64,SUQzBAAAAAA="

    def test_http_url(self):
        """Test HTTP(S) URLs."""
        content = AudioURLContent(url="https://example.com/audio.mp3")
        assert content.url == "https://example.com/audio.mp3"


class TestVideoURLContent:
    """Test VideoURLContent model."""

    def test_valid_video_url(self):
        """Test creating valid video URL content."""
        content = VideoURLContent(url="data:video/mp4;base64,AAAAIGZ0eXBpc29t")
        assert content.url == "data:video/mp4;base64,AAAAIGZ0eXBpc29t"

    def test_http_url(self):
        """Test HTTP(S) URLs."""
        content = VideoURLContent(url="https://example.com/video.mp4")
        assert content.url == "https://example.com/video.mp4"


class TestMessageContentPart:
    """Test MessageContentPart model."""

    def test_text_content_part(self):
        """Test creating a text content part."""
        part = MessageContentPart(type="text", text="Hello, world!")
        assert part.type == "text"
        assert part.text == "Hello, world!"
        assert part.image_url is None
        assert part.audio_url is None
        assert part.video_url is None

    def test_image_url_content_part(self):
        """Test creating an image URL content part."""
        image_url = ImageURLContent(url="data:image/png;base64,iVBORw0KGgo=")
        part = MessageContentPart(type="image_url", image_url=image_url)
        assert part.type == "image_url"
        assert part.image_url.url == "data:image/png;base64,iVBORw0KGgo="
        assert part.text is None

    def test_audio_url_content_part(self):
        """Test creating an audio URL content part."""
        audio_url = AudioURLContent(url="data:audio/mpeg;base64,SUQzBAAAAAA=")
        part = MessageContentPart(type="audio_url", audio_url=audio_url)
        assert part.type == "audio_url"
        assert part.audio_url.url == "data:audio/mpeg;base64,SUQzBAAAAAA="

    def test_video_url_content_part(self):
        """Test creating a video URL content part."""
        video_url = VideoURLContent(url="data:video/mp4;base64,AAAAIGZ0eXBpc29t")
        part = MessageContentPart(type="video_url", video_url=video_url)
        assert part.type == "video_url"
        assert part.video_url.url == "data:video/mp4;base64,AAAAIGZ0eXBpc29t"

    def test_text_part_without_text_fails(self):
        """Test that text type requires text field."""
        with pytest.raises(ValidationError) as exc_info:
            MessageContentPart(type="text", text=None)
        assert "text content part must have 'text' field set" in str(exc_info.value)

    def test_image_url_part_without_url_fails(self):
        """Test that image_url type requires image_url field."""
        with pytest.raises(ValidationError) as exc_info:
            MessageContentPart(type="image_url", image_url=None)
        assert "image_url content part must have 'image_url' field set" in str(
            exc_info.value
        )

    def test_audio_url_part_without_url_fails(self):
        """Test that audio_url type requires audio_url field."""
        with pytest.raises(ValidationError) as exc_info:
            MessageContentPart(type="audio_url", audio_url=None)
        assert "audio_url content part must have 'audio_url' field set" in str(
            exc_info.value
        )

    def test_video_url_part_without_url_fails(self):
        """Test that video_url type requires video_url field."""
        with pytest.raises(ValidationError) as exc_info:
            MessageContentPart(type="video_url", video_url=None)
        assert "video_url content part must have 'video_url' field set" in str(
            exc_info.value
        )


class TestChatMessage:
    """Test ChatMessage model."""

    def test_simple_string_message(self):
        """Test creating a simple string content message."""
        msg = ChatMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_multimodal_message_with_text_and_image(self):
        """Test creating a multimodal message with text and image."""
        parts = [
            MessageContentPart(type="text", text="Describe this image:"),
            MessageContentPart(
                type="image_url",
                image_url=ImageURLContent(url="data:image/png;base64,iVBORw0KGgo="),
            ),
        ]
        msg = ChatMessage(role="user", content=parts)
        assert msg.role == "user"
        assert len(msg.content) == 2
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "image_url"

    def test_multimodal_message_with_audio(self):
        """Test creating a multimodal message with audio."""
        parts = [
            MessageContentPart(type="text", text="Transcribe this:"),
            MessageContentPart(
                type="audio_url",
                audio_url=AudioURLContent(url="data:audio/mpeg;base64,SUQzBAAAAAA="),
            ),
        ]
        msg = ChatMessage(role="user", content=parts)
        assert len(msg.content) == 2
        assert msg.content[1].type == "audio_url"

    def test_multimodal_message_with_video(self):
        """Test creating a multimodal message with video."""
        parts = [
            MessageContentPart(type="text", text="Analyze this video:"),
            MessageContentPart(
                type="video_url",
                video_url=VideoURLContent(url="data:video/mp4;base64,AAAAIGZ0eXBpc29t"),
            ),
        ]
        msg = ChatMessage(role="user", content=parts)
        assert len(msg.content) == 2
        assert msg.content[1].type == "video_url"

    def test_empty_content_list_fails(self):
        """Test that empty content list is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role="user", content=[])
        assert "content list cannot be empty" in str(exc_info.value)

    def test_multiple_images_in_message(self):
        """Test message with multiple images."""
        parts = [
            MessageContentPart(type="text", text="Compare these images:"),
            MessageContentPart(
                type="image_url",
                image_url=ImageURLContent(url="data:image/png;base64,iVBORw0KGgo="),
            ),
            MessageContentPart(
                type="image_url",
                image_url=ImageURLContent(url="data:image/jpeg;base64,/9j/4AAQSkZJ"),
            ),
        ]
        msg = ChatMessage(role="user", content=parts)
        assert len(msg.content) == 3
        assert msg.content[1].type == "image_url"
        assert msg.content[2].type == "image_url"

    def test_mixed_media_types(self):
        """Test message with mixed media types."""
        parts = [
            MessageContentPart(type="text", text="Analyze all of this:"),
            MessageContentPart(
                type="image_url",
                image_url=ImageURLContent(url="data:image/png;base64,iVBORw0KGgo="),
            ),
            MessageContentPart(
                type="audio_url",
                audio_url=AudioURLContent(url="data:audio/mpeg;base64,SUQzBAAAAAA="),
            ),
            MessageContentPart(
                type="video_url",
                video_url=VideoURLContent(url="data:video/mp4;base64,AAAAIGZ0eXBpc29t"),
            ),
        ]
        msg = ChatMessage(role="user", content=parts)
        assert len(msg.content) == 4


class TestChatRequest:
    """Test ChatRequest model."""

    def test_simple_chat_request(self):
        """Test creating a simple chat request."""
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="Hello!")], stream=True
        )
        assert len(req.messages) == 1
        assert req.stream is True

    def test_chat_request_with_multimodal_message(self):
        """Test chat request with multimodal message."""
        parts = [
            MessageContentPart(type="text", text="What's in this image?"),
            MessageContentPart(
                type="image_url",
                image_url=ImageURLContent(url="data:image/png;base64,iVBORw0KGgo="),
            ),
        ]
        req = ChatRequest(
            messages=[ChatMessage(role="user", content=parts)],
            model="gpt-4-vision-preview",
        )
        assert len(req.messages) == 1
        assert isinstance(req.messages[0].content, list)
        assert req.model == "gpt-4-vision-preview"

    def test_chat_request_with_rag_parameters(self):
        """Test chat request with RAG parameters."""
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="Query")],
            rag_enabled=True,
            database="main_db",
            rag_retrieval_strategy="hybrid",
            rag_top_k=5,
            rag_score_threshold=0.7,
        )
        assert req.rag_enabled is True
        assert req.database == "main_db"
        assert req.rag_retrieval_strategy == "hybrid"
        assert req.rag_top_k == 5
        assert req.rag_score_threshold == 0.7

    def test_conversation_with_multimodal_history(self):
        """Test conversation with multimodal message history."""
        req = ChatRequest(
            messages=[
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(
                    role="user",
                    content=[
                        MessageContentPart(type="text", text="What's in this?"),
                        MessageContentPart(
                            type="image_url",
                            image_url=ImageURLContent(
                                url="data:image/png;base64,iVBORw0KGgo="
                            ),
                        ),
                    ],
                ),
                ChatMessage(role="assistant", content="I see a test image."),
                ChatMessage(role="user", content="What color is it?"),
            ]
        )
        assert len(req.messages) == 4
        assert isinstance(req.messages[1].content, list)
        assert isinstance(req.messages[2].content, str)


class TestBase64Encoding:
    """Test base64 encoding/decoding for media."""

    def test_encode_png_image(self):
        """Test encoding a PNG image to base64."""
        # PNG magic bytes
        png_data = b"\x89PNG\r\n\x1a\n"
        encoded = base64.b64encode(png_data).decode("utf-8")
        data_url = f"data:image/png;base64,{encoded}"

        # Verify it can be used in a message
        part = MessageContentPart(
            type="image_url", image_url=ImageURLContent(url=data_url)
        )
        assert part.image_url.url.startswith("data:image/png;base64,")

    def test_encode_jpeg_image(self):
        """Test encoding a JPEG image to base64."""
        # JPEG magic bytes
        jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        encoded = base64.b64encode(jpeg_data).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{encoded}"

        part = MessageContentPart(
            type="image_url", image_url=ImageURLContent(url=data_url)
        )
        assert part.image_url.url.startswith("data:image/jpeg;base64,")

    def test_decode_base64_from_message(self):
        """Test decoding base64 data from a message."""
        png_data = b"\x89PNG\r\n\x1a\n"
        encoded = base64.b64encode(png_data).decode("utf-8")
        data_url = f"data:image/png;base64,{encoded}"

        part = MessageContentPart(
            type="image_url", image_url=ImageURLContent(url=data_url)
        )

        # Extract and decode
        _, base64_part = part.image_url.url.split(";base64,")
        decoded = base64.b64decode(base64_part)
        assert decoded == png_data


class TestMessageSerialization:
    """Test message serialization to/from dict."""

    def test_simple_message_to_dict(self):
        """Test serializing a simple message to dict."""
        msg = ChatMessage(role="user", content="Hello!")
        msg_dict = msg.model_dump()
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Hello!"

    def test_multimodal_message_to_dict(self):
        """Test serializing a multimodal message to dict."""
        parts = [
            MessageContentPart(type="text", text="Describe:"),
            MessageContentPart(
                type="image_url",
                image_url=ImageURLContent(url="data:image/png;base64,iVBORw0KGgo="),
            ),
        ]
        msg = ChatMessage(role="user", content=parts)
        msg_dict = msg.model_dump()

        assert msg_dict["role"] == "user"
        assert isinstance(msg_dict["content"], list)
        assert len(msg_dict["content"]) == 2
        assert msg_dict["content"][0]["type"] == "text"
        assert msg_dict["content"][1]["type"] == "image_url"

    def test_multimodal_message_from_dict(self):
        """Test deserializing a multimodal message from dict."""
        msg_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgo=",
                        "detail": "high",
                    },
                },
            ],
        }
        msg = ChatMessage(**msg_dict)
        assert msg.role == "user"
        assert len(msg.content) == 2
        assert msg.content[0].text == "What is this?"
        assert msg.content[1].image_url.detail == "high"

    def test_chat_request_to_dict(self):
        """Test serializing a chat request to dict."""
        req = ChatRequest(
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        MessageContentPart(type="text", text="Analyze:"),
                        MessageContentPart(
                            type="image_url",
                            image_url=ImageURLContent(
                                url="data:image/png;base64,iVBORw0KGgo="
                            ),
                        ),
                    ],
                )
            ],
            model="gpt-4-vision",
            stream=True,
        )
        req_dict = req.model_dump()

        assert req_dict["model"] == "gpt-4-vision"
        assert req_dict["stream"] is True
        assert len(req_dict["messages"]) == 1
        assert isinstance(req_dict["messages"][0]["content"], list)
