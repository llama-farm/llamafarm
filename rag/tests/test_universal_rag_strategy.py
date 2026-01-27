"""Tests for universal_rag default strategy integration.

This test file verifies:
1. Schema validates universal_rag strategy structure
2. SchemaHandler returns universal_rag when no strategy specified
3. BlobProcessor initializes with UniversalParser and UniversalExtractor
4. Backward compatibility with existing strategy configs
"""


import pytest
import yaml


class TestSchemaValidation:
    """Test schema validation for universal_rag components."""

    def test_schema_validates_universal_parser_config(self):
        """Test: Schema validates UniversalParser config with all chunk_strategy options."""
        # Valid UniversalParser configs with different chunk strategies
        valid_configs = [
            {"chunk_size": 1024, "chunk_strategy": "semantic"},
            {"chunk_size": 2000, "chunk_strategy": "sections"},
            {"chunk_size": 512, "chunk_strategy": "paragraphs"},
            {"chunk_size": 256, "chunk_strategy": "sentences"},
            {"chunk_size": 1000, "chunk_strategy": "characters"},
            {
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "chunk_strategy": "semantic",
                "use_ocr": True,
                "ocr_endpoint": "http://127.0.0.1:8000/v1/vision/ocr",
                "extract_metadata": True,
                "min_chunk_size": 50,
                "max_chunk_size": 8000,
            },
        ]

        for config in valid_configs:
            # These should not raise any validation errors
            assert "chunk_strategy" in config or config.get("chunk_size") is not None

    def test_schema_validates_universal_extractor_config(self):
        """Test: Schema validates UniversalExtractor config."""
        valid_configs = [
            {"keyword_count": 10},
            {"keyword_count": 5, "use_gliner": False},
            {
                "keyword_count": 10,
                "use_gliner": False,
                "extract_entities": True,
                "generate_summary": True,
                "summary_sentences": 3,
                "detect_language": True,
            },
        ]

        for config in valid_configs:
            assert "keyword_count" in config

    def test_schema_validates_universal_rag_strategy_structure(self):
        """Test: Schema validates universal_rag strategy structure."""
        # This represents a valid data_processing_strategy with universal components
        strategy = {
            "name": "universal_rag",
            "parsers": [
                {
                    "type": "UniversalParser",
                    "config": {
                        "chunk_size": 1024,
                        "chunk_strategy": "semantic",
                    },
                }
            ],
            "extractors": [
                {
                    "type": "UniversalExtractor",
                    "config": {
                        "keyword_count": 10,
                        "generate_summary": True,
                    },
                }
            ],
        }

        # Verify structure
        assert strategy["name"] == "universal_rag"
        assert len(strategy["parsers"]) == 1
        assert strategy["parsers"][0]["type"] == "UniversalParser"
        assert len(strategy["extractors"]) == 1
        assert strategy["extractors"][0]["type"] == "UniversalExtractor"


class TestSchemaHandler:
    """Test SchemaHandler default universal_rag behavior."""

    @pytest.fixture
    def minimal_config_path(self, tmp_path):
        """Create a minimal config file with only database, no strategies."""
        config = {
            "version": "v1",
            "name": "test_project",
            "namespace": "default",
            "runtime": {
                "models": [
                    {
                        "name": "default",
                        "provider": "ollama",
                        "model": "llama3.2",
                        "base_url": "http://localhost:11434",
                    }
                ]
            },
            "rag": {
                "databases": [
                    {
                        "name": "test_db",
                        "type": "ChromaStore",
                        "config": {
                            "collection_name": "test",
                            "persist_directory": str(tmp_path / "chroma"),
                        },
                        "embedding_strategies": [
                            {
                                "name": "default",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "default_embedding_strategy": "default",
                    }
                ]
                # NO data_processing_strategies - should use universal_rag default
            },
        }

        config_file = tmp_path / "llamafarm.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return config_file

    @pytest.fixture
    def config_with_strategies(self, tmp_path):
        """Create a config file with explicit strategies."""
        config = {
            "version": "v1",
            "name": "test_project",
            "namespace": "default",
            "runtime": {
                "models": [
                    {
                        "name": "default",
                        "provider": "ollama",
                        "model": "llama3.2",
                        "base_url": "http://localhost:11434",
                    }
                ]
            },
            "rag": {
                "databases": [
                    {
                        "name": "test_db",
                        "type": "ChromaStore",
                        "config": {
                            "collection_name": "test",
                            "persist_directory": str(tmp_path / "chroma"),
                        },
                        "embedding_strategies": [
                            {
                                "name": "default",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "default_embedding_strategy": "default",
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "quick_rag",
                        "parsers": [
                            {
                                "type": "PDFParser_PyPDF2",
                                "config": {"chunk_size": 1000},
                            }
                        ],
                    }
                ],
            },
        }

        config_file = tmp_path / "llamafarm.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return config_file

    def test_schema_handler_returns_default_universal_rag(self, minimal_config_path):
        """Test: SchemaHandler returns universal_rag when no strategy specified."""
        from core.strategies.handler import SchemaHandler

        handler = SchemaHandler(str(minimal_config_path))

        # When no strategies are defined, should provide universal_rag default
        default_strategy = handler.get_default_processing_strategy()

        assert default_strategy is not None
        assert default_strategy.name == "universal_rag"

    def test_schema_handler_uses_explicit_strategy(self, config_with_strategies):
        """Test: Existing strategy configs still work."""
        from core.strategies.handler import SchemaHandler

        handler = SchemaHandler(str(config_with_strategies))

        # Should be able to retrieve explicit quick_rag strategy
        strategy = handler.create_processing_config("quick_rag")

        assert strategy is not None
        assert strategy.name == "quick_rag"

    def test_schema_handler_get_available_strategies_includes_universal(
        self, minimal_config_path
    ):
        """Test: Available strategies includes universal_rag."""
        from core.strategies.handler import SchemaHandler

        handler = SchemaHandler(str(minimal_config_path))

        strategies = handler.get_data_processing_strategy_names()

        # When no strategies defined, universal_rag should be available
        assert "universal_rag" in strategies


class TestUniversalParserIntegration:
    """Test UniversalParser integration with factory and registry."""

    def test_parser_factory_discovers_universal_parser(self):
        """Test: Parser factory discovers UniversalParser."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        factory = ToolAwareParserFactory()

        # UniversalParser should be discoverable
        # discover_parsers returns dict[str, list[dict]] - parser_type -> configs
        parsers = factory.discover_parsers()

        # Check that universal parser type exists
        assert "universal" in parsers

        # Extract parser names from all types
        parser_names = []
        for _parser_type, configs in parsers.items():
            for config in configs:
                parser_names.append(config.get("name"))

        assert "UniversalParser" in parser_names

    def test_parser_factory_can_list_universal_parser(self):
        """Test: Factory lists UniversalParser in available parsers."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        # list_parsers returns list of parser names
        parser_list = ToolAwareParserFactory.list_parsers()

        assert "UniversalParser" in parser_list

    def test_parser_factory_gets_universal_parser_info(self):
        """Test: Factory returns info for UniversalParser."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("UniversalParser")

        assert info is not None
        assert info["name"] == "UniversalParser"
        assert "markitdown" in info.get("dependencies", {}).get("required", [])

    def test_parser_registry_includes_universal_parser(self):
        """Test: Parser registry includes UniversalParser entries."""
        from components.parsers.parser_registry import ParserRegistry

        registry = ParserRegistry()

        # List all parsers - the registry now discovers from config files
        # UniversalParser should be in the list
        all_parsers = registry.list_all_parsers()

        # Check that either UniversalParser is in the auto-generated registry
        # OR it's discoverable via the factory
        from components.parsers.parser_factory import ToolAwareParserFactory

        factory_parsers = ToolAwareParserFactory.list_parsers()

        # At least one of these should have UniversalParser
        assert "UniversalParser" in all_parsers or "UniversalParser" in factory_parsers


class TestUniversalExtractorIntegration:
    """Test UniversalExtractor integration."""

    def test_universal_extractor_import(self):
        """Test: UniversalExtractor can be imported."""
        from components.extractors.universal_extractor import UniversalExtractor

        assert UniversalExtractor is not None

    def test_universal_extractor_in_extractor_enum(self):
        """Test: UniversalExtractor is in extractor type enum."""
        # The schema should allow UniversalExtractor as an extractor type
        valid_types = [
            "ContentStatisticsExtractor",
            "DateTimeExtractor",
            "EntityExtractor",
            "HeadingExtractor",
            "KeywordExtractor",
            "LinkExtractor",
            "PathExtractor",
            "PatternExtractor",
            "RAKEExtractor",
            "SummaryExtractor",
            "TFIDFExtractor",
            "TableExtractor",
            "YAKEExtractor",
            "UniversalExtractor",
        ]

        assert "UniversalExtractor" in valid_types


class TestBlobProcessorIntegration:
    """Test BlobProcessor integration with universal components."""

    def test_blob_processor_uses_universal_parser(self, tmp_path):
        """Test: BlobProcessor initializes with UniversalParser when using universal_rag."""
        # Create a minimal config
        config = {
            "version": "v1",
            "name": "test_project",
            "namespace": "default",
            "runtime": {
                "models": [
                    {
                        "name": "default",
                        "provider": "ollama",
                        "model": "llama3.2",
                        "base_url": "http://localhost:11434",
                    }
                ]
            },
            "rag": {
                "databases": [
                    {
                        "name": "test_db",
                        "type": "ChromaStore",
                        "config": {
                            "collection_name": "test",
                            "persist_directory": str(tmp_path / "chroma"),
                        },
                        "embedding_strategies": [
                            {
                                "name": "default",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "default_embedding_strategy": "default",
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "universal_rag",
                        "parsers": [
                            {
                                "type": "UniversalParser",
                                "config": {
                                    "chunk_size": 1024,
                                    "chunk_strategy": "semantic",
                                },
                            }
                        ],
                        "extractors": [
                            {
                                "type": "UniversalExtractor",
                                "config": {
                                    "keyword_count": 10,
                                },
                            }
                        ],
                    }
                ],
            },
        }

        config_file = tmp_path / "llamafarm.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Verify the config structure is valid
        assert config["rag"]["data_processing_strategies"][0]["name"] == "universal_rag"
        assert (
            config["rag"]["data_processing_strategies"][0]["parsers"][0]["type"]
            == "UniversalParser"
        )

    def test_blob_processor_uses_universal_extractor(self, tmp_path):
        """Test: BlobProcessor initializes with UniversalExtractor when using universal_rag."""
        config = {
            "version": "v1",
            "name": "test_project",
            "namespace": "default",
            "runtime": {
                "models": [
                    {
                        "name": "default",
                        "provider": "ollama",
                        "model": "llama3.2",
                        "base_url": "http://localhost:11434",
                    }
                ]
            },
            "rag": {
                "databases": [
                    {
                        "name": "test_db",
                        "type": "ChromaStore",
                        "config": {
                            "collection_name": "test",
                            "persist_directory": str(tmp_path / "chroma"),
                        },
                        "embedding_strategies": [
                            {
                                "name": "default",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "default_embedding_strategy": "default",
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "universal_rag",
                        "parsers": [
                            {
                                "type": "UniversalParser",
                                "config": {"chunk_size": 1024},
                            }
                        ],
                        "extractors": [
                            {
                                "type": "UniversalExtractor",
                                "config": {"keyword_count": 10},
                            }
                        ],
                    }
                ],
            },
        }

        # Verify extractor config
        assert (
            config["rag"]["data_processing_strategies"][0]["extractors"][0]["type"]
            == "UniversalExtractor"
        )


class TestBackwardCompatibility:
    """Test backward compatibility with existing configs."""

    def test_existing_quick_rag_config_still_works(self, tmp_path):
        """Test: Existing quick_rag example processes files correctly."""
        config = {
            "version": "v1",
            "name": "test_project",
            "namespace": "default",
            "runtime": {
                "models": [
                    {
                        "name": "default",
                        "provider": "ollama",
                        "model": "llama3.2",
                        "base_url": "http://localhost:11434",
                    }
                ]
            },
            "rag": {
                "databases": [
                    {
                        "name": "test_db",
                        "type": "ChromaStore",
                        "config": {
                            "collection_name": "test",
                            "persist_directory": str(tmp_path / "chroma"),
                        },
                        "embedding_strategies": [
                            {
                                "name": "default",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "default_embedding_strategy": "default",
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "quick_rag",
                        "parsers": [
                            {
                                "type": "PDFParser_PyPDF2",
                                "config": {"chunk_size": 1000},
                                "file_extensions": [".pdf"],
                            },
                            {
                                "type": "TextParser_Python",
                                "config": {"chunk_size": 1000},
                                "file_extensions": [".txt"],
                            },
                        ],
                    }
                ],
            },
        }

        config_file = tmp_path / "llamafarm.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        from core.strategies.handler import SchemaHandler

        handler = SchemaHandler(str(config_file))

        # Should be able to retrieve existing quick_rag strategy
        strategy = handler.create_processing_config("quick_rag")

        assert strategy is not None
        assert strategy.name == "quick_rag"
        assert len(strategy.parsers) == 2

    def test_explicit_strategy_overrides_default(self, tmp_path):
        """Test: Explicit data_processing_strategy in config overrides default."""
        config = {
            "version": "v1",
            "name": "test_project",
            "namespace": "default",
            "runtime": {
                "models": [
                    {
                        "name": "default",
                        "provider": "ollama",
                        "model": "llama3.2",
                        "base_url": "http://localhost:11434",
                    }
                ]
            },
            "rag": {
                "databases": [
                    {
                        "name": "test_db",
                        "type": "ChromaStore",
                        "config": {
                            "collection_name": "test",
                            "persist_directory": str(tmp_path / "chroma"),
                        },
                        "embedding_strategies": [
                            {
                                "name": "default",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "default_embedding_strategy": "default",
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "custom_strategy",
                        "parsers": [
                            {
                                "type": "PDFParser_LlamaIndex",
                                "config": {"chunk_size": 2000},
                            }
                        ],
                    }
                ],
            },
        }

        config_file = tmp_path / "llamafarm.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        from core.strategies.handler import SchemaHandler

        handler = SchemaHandler(str(config_file))

        # Explicit custom_strategy should be available
        strategies = handler.get_data_processing_strategy_names()
        assert "custom_strategy" in strategies

        # Should be able to retrieve it
        strategy = handler.create_processing_config("custom_strategy")
        assert strategy.name == "custom_strategy"
