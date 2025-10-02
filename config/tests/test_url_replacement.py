"""
Tests for URL replacement functionality in config loader.
"""

import socket
from unittest.mock import patch

from config.helpers import loader
from config.helpers.loader import (
    _is_host_docker_internal_resolvable,
    _replace_localhost_url,
    _replace_urls_in_config,
    _reset_host_docker_internal_cache,
    load_config_dict,
)


class TestURLReplacement:
    """Test URL replacement functionality."""

    def setup_method(self):
        """Reset cache before each test."""
        _reset_host_docker_internal_cache()

    def test_is_host_docker_internal_resolvable_success(self):
        """Test successful DNS resolution of host.docker.internal."""
        with patch("socket.gethostbyname") as mock_gethostbyname:
            mock_gethostbyname.return_value = "192.168.65.254"
            assert _is_host_docker_internal_resolvable() is True
            mock_gethostbyname.assert_called_once_with("host.docker.internal")

    def test_is_host_docker_internal_resolvable_failure(self):
        """Test failed DNS resolution of host.docker.internal."""
        with patch("socket.gethostbyname") as mock_gethostbyname:
            mock_gethostbyname.side_effect = socket.gaierror("Name resolution failed")
            assert _is_host_docker_internal_resolvable() is False
            mock_gethostbyname.assert_called_once_with("host.docker.internal")

    def _assert_dns_call_and_result(
        self, mock_gethostbyname, expected_result=True, expected_call_count=1
    ):
        """Helper method to assert DNS call result and count."""
        result = _is_host_docker_internal_resolvable()
        assert result is expected_result
        assert mock_gethostbyname.call_count == expected_call_count

    def test_is_host_docker_internal_resolvable_caching(self):
        """Test that DNS resolution result is cached."""
        with patch("socket.gethostbyname") as mock_gethostbyname:
            mock_gethostbyname.return_value = "192.168.65.254"

            # First call should hit the DNS
            self._assert_dns_call_and_result(
                mock_gethostbyname, expected_result=True, expected_call_count=1
            )

            # Second call should use cache, no additional DNS call
            self._assert_dns_call_and_result(
                mock_gethostbyname, expected_result=True, expected_call_count=1
            )

            # Third call should also use cache
            self._assert_dns_call_and_result(
                mock_gethostbyname, expected_result=True, expected_call_count=1
            )

    def test_reset_host_docker_internal_cache(self):
        """Test that cache reset works correctly."""
        with patch("socket.gethostbyname") as mock_gethostbyname:
            mock_gethostbyname.return_value = "192.168.65.254"

            # First call should hit DNS
            self._assert_dns_call_and_result(
                mock_gethostbyname, expected_result=True, expected_call_count=1
            )

            # Reset cache
            _reset_host_docker_internal_cache()

            # Next call should hit DNS again
            self._assert_dns_call_and_result(
                mock_gethostbyname, expected_result=True, expected_call_count=2
            )

    def test_replace_localhost_url_http(self):
        """Test replacing HTTP localhost URL."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_localhost_url("http://localhost:8080/api")
            assert result == "http://host.docker.internal:8080/api"

    def test_replace_localhost_url_https(self):
        """Test replacing HTTPS localhost URL."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_localhost_url("https://localhost:443/secure")
            assert result == "https://host.docker.internal:443/secure"

    def test_replace_localhost_url_no_port(self):
        """Test replacing localhost URL without port."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_localhost_url("http://localhost/api")
            assert result == "http://host.docker.internal/api"

    def test_replace_localhost_url_no_path(self):
        """Test replacing localhost URL without path."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_localhost_url("http://localhost:8080")
            assert result == "http://host.docker.internal:8080"

    def test_replace_localhost_url_not_resolvable(self):
        """Test that URL is not replaced when host.docker.internal is not resolvable."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=False):
            result = _replace_localhost_url("http://localhost:8080/api")
            assert result == "http://localhost:8080/api"

    def test_replace_localhost_url_127_0_0_1_http(self):
        """Test replacing HTTP 127.0.0.1 URL."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_localhost_url("http://127.0.0.1:8080/api")
            assert result == "http://host.docker.internal:8080/api"

    def test_replace_localhost_url_127_0_0_1_https(self):
        """Test replacing HTTPS 127.0.0.1 URL."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_localhost_url("https://127.0.0.1:443/secure")
            assert result == "https://host.docker.internal:443/secure"

    def test_replace_localhost_url_127_0_0_1_no_port(self):
        """Test replacing 127.0.0.1 URL without port."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_localhost_url("http://127.0.0.1/api")
            assert result == "http://host.docker.internal/api"

    def test_replace_localhost_url_127_0_0_1_no_path(self):
        """Test replacing 127.0.0.1 URL without path."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_localhost_url("http://127.0.0.1:8080")
            assert result == "http://host.docker.internal:8080"

    def test_replace_localhost_url_127_0_0_1_not_resolvable(self):
        """Test that 127.0.0.1 URL is not replaced when host.docker.internal is not resolvable."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=False):
            result = _replace_localhost_url("http://127.0.0.1:8080/api")
            assert result == "http://127.0.0.1:8080/api"

    def test_replace_localhost_url_non_localhost(self):
        """Test that non-localhost URLs are not modified."""
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_localhost_url("http://example.com:8080/api")
            assert result == "http://example.com:8080/api"

    def test_replace_localhost_url_non_string(self):
        """Test that non-string values are returned unchanged."""
        # Integer
        result = _replace_localhost_url(123)
        assert result == 123
        # None
        result_none = _replace_localhost_url(None)
        assert result_none is None
        # Bytes
        result_bytes = _replace_localhost_url(b"http://localhost:8080/api")
        assert result_bytes == b"http://localhost:8080/api"
        # List
        result_list = _replace_localhost_url(["http://localhost:8080/api"])
        assert result_list == ["http://localhost:8080/api"]

    def test_replace_urls_in_config_simple_dict(self):
        """Test URL replacement in a simple dictionary."""
        config = {
            "base_url": "http://localhost:8080",
            "api_url": "https://localhost:9000/api",
            "other_field": "not a url",
        }

        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_urls_in_config(config)

        expected = {
            "base_url": "http://host.docker.internal:8080",
            "api_url": "https://host.docker.internal:9000/api",
            "other_field": "not a url",
        }
        assert result == expected

    def test_replace_urls_in_config_with_127_0_0_1(self):
        """Test URL replacement with 127.0.0.1 addresses."""
        config = {
            "localhost_url": "http://localhost:8080",
            "ip_url": "https://127.0.0.1:9000/api",
            "mixed_list": ["http://localhost:3000", "https://127.0.0.1:4000/path", "not a url"],
            "other_field": "not a url",
        }

        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_urls_in_config(config)

        expected = {
            "localhost_url": "http://host.docker.internal:8080",
            "ip_url": "https://host.docker.internal:9000/api",
            "mixed_list": [
                "http://host.docker.internal:3000",
                "https://host.docker.internal:4000/path",
                "not a url",
            ],
            "other_field": "not a url",
        }
        assert result == expected

    def test_replace_urls_in_config_nested_dict(self):
        """Test URL replacement in nested dictionaries."""
        config = {
            "runtime": {"base_url": "http://localhost:11434"},
            "rag": {
                "databases": [
                    {"embedding_strategies": [{"config": {"base_url": "http://localhost:8080"}}]}
                ]
            },
        }

        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_urls_in_config(config)

        expected = {
            "runtime": {"base_url": "http://host.docker.internal:11434"},
            "rag": {
                "databases": [
                    {
                        "embedding_strategies": [
                            {"config": {"base_url": "http://host.docker.internal:8080"}}
                        ]
                    }
                ]
            },
        }
        assert result == expected

    def test_replace_urls_in_config_list(self):
        """Test URL replacement in lists."""
        config = [{"base_url": "http://localhost:8080"}, {"api_url": "https://localhost:9000"}]

        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_urls_in_config(config)

        expected = [
            {"base_url": "http://host.docker.internal:8080"},
            {"api_url": "https://host.docker.internal:9000"},
        ]
        assert result == expected

    def test_replace_urls_in_config_primitive_values(self):
        """Test that primitive values are returned unchanged."""
        assert _replace_urls_in_config("string") == "string"
        assert _replace_urls_in_config(123) == 123
        assert _replace_urls_in_config(True) is True
        assert _replace_urls_in_config(None) is None

    def test_replace_urls_in_config_deeply_nested(self):
        """Test that deeply nested structures with mixed types are handled recursively."""
        config = {
            "level1": [
                {
                    "level2": {
                        "url": "http://localhost:8000",
                        "value": 42,
                        "none_value": None,
                        "list": [
                            "http://localhost:9000",
                            False,
                            None,
                            {"deep_url": "http://localhost:7000"},
                        ],
                    }
                },
                "string_value",
                None,
            ],
            "simple_url": "http://localhost:5000",
        }
        expected = {
            "level1": [
                {
                    "level2": {
                        "url": "http://host.docker.internal:8000",
                        "value": 42,
                        "none_value": None,
                        "list": [
                            "http://host.docker.internal:9000",
                            False,
                            None,
                            {"deep_url": "http://host.docker.internal:7000"},
                        ],
                    }
                },
                "string_value",
                None,
            ],
            "simple_url": "http://host.docker.internal:5000",
        }
        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = _replace_urls_in_config(config)
        assert result == expected

    def test_load_config_dict_with_url_replacement(self, temp_config_file):
        """Test that load_config_dict applies URL replacement."""
        config_content = """
version: v1
name: test_config
namespace: test

runtime:
  provider: ollama
  model: test-model
  base_url: http://localhost:11434

rag:
  databases:
    - name: test_db
      type: ChromaStore
      embedding_strategies:
        - name: test_embedding
          type: OllamaEmbedder
          config:
            model: test-model
            base_url: http://localhost:8080
"""

        config_file = temp_config_file(config_content)

        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = load_config_dict(config_file, validate=False)

        # Check that URLs were replaced
        assert result["runtime"]["base_url"] == "http://host.docker.internal:11434"
        assert (
            result["rag"]["databases"][0]["embedding_strategies"][0]["config"]["base_url"]
            == "http://host.docker.internal:8080"
        )

    def test_load_config_dict_no_replacement_when_not_resolvable(self, temp_config_file):
        """Test that URLs are not replaced when host.docker.internal is not resolvable."""
        config_content = """
version: v1
name: test_config
namespace: test

runtime:
  provider: ollama
  model: test-model
  base_url: http://localhost:11434
"""

        config_file = temp_config_file(config_content)

        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=False):
            result = load_config_dict(config_file, validate=False)

        # Check that URLs were NOT replaced
        assert result["runtime"]["base_url"] == "http://localhost:11434"

    def test_load_config_dict_with_127_0_0_1_replacement(self, temp_config_file):
        """Test that load_config_dict applies URL replacement for 127.0.0.1."""
        config_content = """
version: v1
name: test_config
namespace: test

runtime:
  provider: ollama
  model: test-model
  base_url: http://127.0.0.1:11434

rag:
  databases:
    - name: test_db
      type: ChromaStore
      embedding_strategies:
        - name: test_embedding
          type: OllamaEmbedder
          config:
            model: test-model
            base_url: https://127.0.0.1:8080/v1
"""

        config_file = temp_config_file(config_content)

        with patch.object(loader, "_is_host_docker_internal_resolvable", return_value=True):
            result = load_config_dict(config_file, validate=False)

        # Check that 127.0.0.1 URLs were replaced
        assert result["runtime"]["base_url"] == "http://host.docker.internal:11434"
        assert (
            result["rag"]["databases"][0]["embedding_strategies"][0]["config"]["base_url"]
            == "https://host.docker.internal:8080/v1"
        )
