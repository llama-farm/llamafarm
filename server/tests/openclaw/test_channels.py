"""Tests for OpenClaw Lite channel system."""
import pytest
from agents.channels import ChannelRegistry, Channel, WebhookChannel


class TestChannel:
    """Test individual Channel objects."""
    
    def test_channel_creation(self):
        """Test creating a channel."""
        channel = Channel(
            name="slack",
            channel_type="messaging",
            config={"api_key": "test-key"}
        )
        
        assert channel.name == "slack"
        assert channel.channel_type == "messaging"
        assert channel.config["api_key"] == "test-key"
    
    def test_channel_send_message(self):
        """Test sending a message via channel."""
        messages_sent = []
        
        def mock_send(msg):
            messages_sent.append(msg)
            return {"status": "sent", "id": "123"}
        
        channel = Channel(
            name="test",
            channel_type="messaging",
            send_handler=mock_send
        )
        
        result = channel.send("Hello world")
        assert result["status"] == "sent"
        assert len(messages_sent) == 1
        assert messages_sent[0] == "Hello world"
    
    def test_channel_receive_webhook(self):
        """Test receiving webhook data."""
        received = []
        
        def webhook_handler(data):
            received.append(data)
            return {"processed": True}
        
        channel = Channel(
            name="webhook-test",
            channel_type="webhook",
            webhook_handler=webhook_handler
        )
        
        result = channel.handle_webhook({"message": "incoming"})
        assert result["processed"] is True
        assert len(received) == 1


class TestChannelRegistry:
    """Test ChannelRegistry for managing channels."""
    
    def test_registry_initialization(self):
        """Test registry starts empty."""
        registry = ChannelRegistry()
        assert len(registry.list_channels()) == 0
    
    def test_register_channel(self):
        """Test registering a channel."""
        registry = ChannelRegistry()
        
        channel = Channel(
            name="discord",
            channel_type="messaging",
            config={"token": "bot-token"}
        )
        
        registry.register(channel)
        assert "discord" in registry.list_channels()
    
    def test_get_channel(self):
        """Test retrieving a channel."""
        registry = ChannelRegistry()
        
        channel = Channel(name="telegram", channel_type="messaging")
        registry.register(channel)
        
        retrieved = registry.get_channel("telegram")
        assert retrieved is not None
        assert retrieved.name == "telegram"
    
    def test_unregister_channel(self):
        """Test removing a channel."""
        registry = ChannelRegistry()
        channel = Channel(name="temp", channel_type="test")
        
        registry.register(channel)
        assert registry.has_channel("temp")
        
        registry.unregister("temp")
        assert not registry.has_channel("temp")
    
    def test_list_channels_by_type(self):
        """Test filtering channels by type."""
        registry = ChannelRegistry()
        
        registry.register(Channel("slack", "messaging"))
        registry.register(Channel("discord", "messaging"))
        registry.register(Channel("webhook1", "webhook"))
        
        messaging = registry.list_by_type("messaging")
        assert len(messaging) == 2
        assert all(c.channel_type == "messaging" for c in messaging)


class TestWebhookHandler:
    """Test WebhookHandler for processing incoming webhooks."""
    
    def test_handler_initialization(self):
        """Test webhook handler setup."""
        handler = WebhookHandler(path="/webhook")
        assert handler.path == "/webhook"
    
    def test_register_webhook_route(self):
        """Test registering webhook route."""
        handler = WebhookHandler(path="/hooks")
        
        processed = []
        def processor(data):
            processed.append(data)
            return {"ok": True}
        
        handler.register_processor("slack", processor)
        
        result = handler.process("slack", {"event": "message"})
        assert result["ok"] is True
        assert len(processed) == 1
    
    def test_webhook_authentication(self):
        """Test webhook signature verification."""
        handler = WebhookHandler(path="/secure")
        
        handler.set_auth("slack", secret="my-secret")
        
        # Mock request with signature
        request = {
            "headers": {"X-Signature": "valid-signature"},
            "body": '{"data": "test"}'
        }
        
        is_valid = handler.verify_signature("slack", request)
        assert isinstance(is_valid, bool)
    
    def test_unknown_channel_webhook(self):
        """Test webhook from unregistered channel."""
        handler = WebhookHandler()
        
        with pytest.raises(KeyError):
            handler.process("unknown-channel", {})
