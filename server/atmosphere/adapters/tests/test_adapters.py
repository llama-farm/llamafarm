"""
Tests for the Atmosphere adapter system.
"""

import pytest
import asyncio
from atmosphere.adapters import (
    Adapter,
    Capability,
    AdapterManager,
    register_adapter,
    get_adapter,
    list_adapters,
    create_adapter,
    discover_adapters,
)


class TestAdapterRegistry:
    """Test adapter registration and discovery."""
    
    def test_list_adapters(self):
        """Should list registered adapters."""
        adapters = list_adapters()
        names = [a.name for a in adapters]
        
        # LlamaFarm and Ollama should be registered
        assert "llamafarm" in names
        assert "ollama" in names
    
    def test_get_adapter(self):
        """Should get adapter by name."""
        cls = get_adapter("llamafarm")
        assert cls is not None
        assert cls.NAME == "llamafarm"
        
        cls = get_adapter("ollama")
        assert cls is not None
        assert cls.NAME == "ollama"
        
        # Unknown adapter
        cls = get_adapter("nonexistent")
        assert cls is None
    
    def test_create_adapter(self):
        """Should create adapter instance."""
        adapter = create_adapter("llamafarm", url="http://localhost:9999")
        assert adapter is not None
        assert adapter.url == "http://localhost:9999"
        assert not adapter.is_connected
    
    def test_priority_ordering(self):
        """Adapters should be ordered by priority."""
        adapters = list_adapters()
        
        # LlamaFarm (priority=10) should come before Ollama (priority=5)
        llamafarm_idx = next(i for i, a in enumerate(adapters) if a.name == "llamafarm")
        ollama_idx = next(i for i, a in enumerate(adapters) if a.name == "ollama")
        
        assert llamafarm_idx < ollama_idx


class TestCapability:
    """Test Capability dataclass."""
    
    def test_create_capability(self):
        """Should create capability."""
        cap = Capability(
            name="embeddings",
            description="Text embeddings",
            models=["model-a", "model-b"]
        )
        
        assert cap.name == "embeddings"
        assert len(cap.models) == 2
    
    def test_capability_defaults(self):
        """Should have sensible defaults."""
        cap = Capability(name="test", description="Test cap")
        
        assert cap.models == []
        assert cap.constraints == {}
        assert cap.vector is None


class TestAdapterInterface:
    """Test the Adapter base class interface."""
    
    def test_adapter_init(self):
        """Should initialize with URL."""
        adapter = create_adapter("llamafarm", url="http://custom:8000")
        
        assert adapter.url == "http://custom:8000"
        assert not adapter.is_connected
        assert adapter.capabilities == []
    
    def test_adapter_default_url(self):
        """Should use default URL if not provided."""
        adapter = create_adapter("llamafarm")
        assert adapter.url == "http://localhost:14345"
        
        adapter = create_adapter("ollama")
        assert adapter.url == "http://localhost:11434"
    
    def test_adapter_to_dict(self):
        """Should serialize to dict."""
        adapter = create_adapter("llamafarm")
        d = adapter.to_dict()
        
        assert d["name"] == "llamafarm"
        assert "url" in d
        assert "connected" in d
        assert "capabilities" in d


class TestAdapterManager:
    """Test the AdapterManager."""
    
    def test_manager_init(self):
        """Should initialize manager."""
        manager = AdapterManager()
        assert manager.connected_adapters == []
    
    @pytest.mark.asyncio
    async def test_discover_handles_failures(self):
        """Should handle connection failures gracefully."""
        manager = AdapterManager()
        
        # Try to discover with invalid URLs
        await manager.discover_all(urls={
            "llamafarm": "http://localhost:99999",
            "ollama": "http://localhost:99998"
        })
        
        # Should not crash, just have no connections
        # (actual connections depend on what's running)
    
    def test_get_status(self):
        """Should return status dict."""
        manager = AdapterManager()
        status = manager.get_status()
        
        assert "adapters" in status
        assert "capabilities" in status


class TestCustomAdapter:
    """Test creating custom adapters."""
    
    def test_register_custom_adapter(self):
        """Should be able to register a custom adapter."""
        
        @register_adapter("test-custom", description="Test adapter", priority=1)
        class TestAdapter(Adapter):
            NAME = "test-custom"
            
            async def connect(self):
                return True
            
            async def disconnect(self):
                pass
            
            async def health_check(self):
                return True
            
            async def get_capabilities(self):
                return [Capability(name="test", description="Test")]
            
            async def execute(self, capability, **kwargs):
                return "test result"
        
        # Should be registered
        assert get_adapter("test-custom") is not None
        
        # Should be in list
        adapters = list_adapters()
        assert any(a.name == "test-custom" for a in adapters)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
