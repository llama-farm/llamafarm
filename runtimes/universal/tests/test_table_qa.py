"""Tests for Table Question Answering Model (TAPAS).

Phase 13 of Universal Runtime ML Enhancements.
"""

import pytest

from models.table_qa_model import AGGREGATION_TYPES, TableQAModel


def create_simple_table() -> dict:
    """Create a simple test table."""
    return {
        "columns": ["Name", "Age", "City"],
        "rows": [
            ["Alice", "30", "New York"],
            ["Bob", "25", "Los Angeles"],
            ["Charlie", "35", "Chicago"],
        ],
    }


def create_numeric_table() -> dict:
    """Create a table with numeric data for aggregation."""
    return {
        "columns": ["Product", "Price", "Quantity"],
        "rows": [
            ["Apple", "1.50", "10"],
            ["Banana", "0.75", "20"],
            ["Orange", "2.00", "15"],
        ],
    }


def create_server_metrics_table() -> dict:
    """Create a table simulating server metrics."""
    return {
        "columns": ["Server", "CPU%", "Memory%", "Status"],
        "rows": [
            ["web-01", "45", "60", "healthy"],
            ["web-02", "85", "70", "warning"],
            ["db-01", "30", "80", "healthy"],
            ["cache-01", "15", "40", "healthy"],
        ],
    }


class TestTableQAModelLoading:
    """Test TAPAS model loading."""

    @pytest.mark.asyncio
    async def test_model_loads_successfully(self):
        """Test that TAPAS model loads."""
        model = TableQAModel(model_id="test-tapas", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()

    @pytest.mark.asyncio
    async def test_model_unload(self):
        """Test that model can be unloaded."""
        model = TableQAModel(model_id="test-tapas", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()
        assert not model.is_loaded

    @pytest.mark.asyncio
    async def test_model_reloads(self):
        """Test that model can be reloaded after unload."""
        model = TableQAModel(model_id="test-tapas", device="cpu")
        await model.load()
        await model.unload()
        await model.load()
        assert model.is_loaded
        await model.unload()


class TestQuestionAnswering:
    """Test question answering functionality."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = TableQAModel(model_id="test-tapas", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_answer_returns_dict(self, model):
        """Test that answer returns a dictionary."""
        table = create_simple_table()
        result = await model.answer(table, "Who lives in Chicago?")
        assert isinstance(result, dict)
        assert "answer" in result

    @pytest.mark.asyncio
    async def test_answer_has_required_keys(self, model):
        """Test that answer has required keys."""
        table = create_simple_table()
        result = await model.answer(table, "Who lives in New York?")
        assert "answer" in result
        assert "cells" in result
        assert "aggregation" in result
        assert "question" in result

    @pytest.mark.asyncio
    async def test_cells_format(self, model):
        """Test that cells have correct format."""
        table = create_simple_table()
        result = await model.answer(table, "Who is the oldest?")

        for cell in result["cells"]:
            assert "row" in cell
            assert "column" in cell
            assert isinstance(cell["row"], int)
            assert isinstance(cell["column"], int)

    @pytest.mark.asyncio
    async def test_question_preserved(self, model):
        """Test that question is preserved in result."""
        table = create_simple_table()
        question = "What is Alice's age?"
        result = await model.answer(table, question)
        assert result["question"] == question


class TestAggregation:
    """Test aggregation queries."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = TableQAModel(model_id="test-tapas", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_aggregation_type_valid(self, model):
        """Test that aggregation type is valid."""
        table = create_numeric_table()
        result = await model.answer(table, "What is the total quantity?")
        assert result["aggregation"] in AGGREGATION_TYPES.values()

    @pytest.mark.asyncio
    async def test_cell_values_included(self, model):
        """Test that cell values are included."""
        table = create_simple_table()
        result = await model.answer(table, "Who lives in Chicago?")
        assert "cell_values" in result


class TestBatchQuestions:
    """Test batch question answering."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = TableQAModel(model_id="test-tapas", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_batch_returns_list(self, model):
        """Test that batch returns a list."""
        table = create_simple_table()
        questions = ["Who is the oldest?", "Who lives in New York?"]
        results = await model.answer_batch(table, questions)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_batch_correct_count(self, model):
        """Test that batch returns correct number of results."""
        table = create_simple_table()
        questions = ["Who is Alice?", "Who is Bob?", "Who is Charlie?"]
        results = await model.answer_batch(table, questions)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_each_result_complete(self, model):
        """Test that each batch result has all fields."""
        table = create_simple_table()
        questions = ["Who is the oldest?", "Who is the youngest?"]
        results = await model.answer_batch(table, questions)

        for result in results:
            assert "answer" in result
            assert "cells" in result
            assert "aggregation" in result


class TestValidation:
    """Test input validation."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = TableQAModel(model_id="test-tapas", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_missing_columns_raises(self, model):
        """Test that missing columns raises error."""
        table = {"rows": [["a", "b"]]}
        with pytest.raises(ValueError, match="columns"):
            await model.answer(table, "Test?")

    @pytest.mark.asyncio
    async def test_missing_rows_raises(self, model):
        """Test that missing rows raises error."""
        table = {"columns": ["A", "B"]}
        with pytest.raises(ValueError, match="rows"):
            await model.answer(table, "Test?")

    @pytest.mark.asyncio
    async def test_empty_columns_raises(self, model):
        """Test that empty columns raises error."""
        table = {"columns": [], "rows": [["a", "b"]]}
        with pytest.raises(ValueError, match="at least one column"):
            await model.answer(table, "Test?")

    @pytest.mark.asyncio
    async def test_empty_rows_raises(self, model):
        """Test that empty rows raises error."""
        table = {"columns": ["A", "B"], "rows": []}
        with pytest.raises(ValueError, match="at least one row"):
            await model.answer(table, "Test?")

    @pytest.mark.asyncio
    async def test_empty_question_raises(self, model):
        """Test that empty question raises error."""
        table = create_simple_table()
        with pytest.raises(ValueError, match="empty"):
            await model.answer(table, "")

    @pytest.mark.asyncio
    async def test_whitespace_question_raises(self, model):
        """Test that whitespace-only question raises error."""
        table = create_simple_table()
        with pytest.raises(ValueError, match="empty"):
            await model.answer(table, "   ")


class TestModelNotLoaded:
    """Test behavior when model is not loaded."""

    @pytest.mark.asyncio
    async def test_answer_without_load_raises(self):
        """Test that answer without load raises error."""
        model = TableQAModel(model_id="test-tapas", device="cpu")
        table = create_simple_table()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await model.answer(table, "Test?")


class TestModelInfo:
    """Test model information retrieval."""

    @pytest.mark.asyncio
    async def test_model_info(self):
        """Test getting model info."""
        model = TableQAModel(model_id="test-tapas", device="cpu")
        await model.load()

        info = model.get_model_info()
        assert "hf_model_name" in info
        assert "is_loaded" in info
        assert "supported_aggregations" in info
        assert info["is_loaded"] is True

        await model.unload()


class TestSupportedOptions:
    """Test supported aggregation types."""

    def test_aggregation_types_defined(self):
        """Test that AGGREGATION_TYPES is defined."""
        assert "NONE" in AGGREGATION_TYPES.values()
        assert "SUM" in AGGREGATION_TYPES.values()
        assert "AVERAGE" in AGGREGATION_TYPES.values()
        assert "COUNT" in AGGREGATION_TYPES.values()


class TestRealisticQueries:
    """Test realistic query scenarios."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = TableQAModel(model_id="test-tapas", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_server_metrics_query(self, model):
        """Test query on server metrics table."""
        table = create_server_metrics_table()
        result = await model.answer(table, "Which server has the highest CPU?")
        assert "answer" in result
        # web-02 has 85% CPU
        assert result["answer"] is not None

    @pytest.mark.asyncio
    async def test_product_query(self, model):
        """Test query on product table."""
        table = create_numeric_table()
        result = await model.answer(table, "What is the most expensive product?")
        assert "answer" in result
