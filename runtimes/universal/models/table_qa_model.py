"""Table Question Answering Model using TAPAS.

Provides natural language question answering over tabular data using
Google's TAPAS model (Table Parser).

Usage:
    model = TableQAModel(model_id="tapas")
    await model.load()
    result = await model.answer(
        table={"columns": ["Name", "Age"], "rows": [["Alice", "30"], ["Bob", "25"]]},
        question="Who is older?"
    )
    # Returns: {"answer": "Alice", "cells": [...], "aggregation": "NONE"}
"""

import asyncio
import logging
from typing import Any

import pandas as pd
import torch

from models.base import BaseModel

logger = logging.getLogger(__name__)

# Default TAPAS model for table QA
DEFAULT_TAPAS_MODEL = "google/tapas-base-finetuned-wtq"

# Aggregation types from TAPAS
AGGREGATION_TYPES = {
    0: "NONE",
    1: "SUM",
    2: "AVERAGE",
    3: "COUNT",
}


class TableQAModel(BaseModel):
    """Table question answering model using TAPAS.

    TAPAS (Table Parser) is a BERT-based model trained for question answering
    over tables. It can answer questions requiring cell selection, aggregation,
    or counting.

    Attributes:
        model_id: Unique identifier for this model instance
        device: Device to run inference on (cpu, cuda, mps)
        hf_model_name: HuggingFace model name
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hf_model_name: str = DEFAULT_TAPAS_MODEL,
        token: str | None = None,
    ):
        """Initialize table QA model.

        Args:
            model_id: Unique identifier for this model instance
            device: Device to run on (cpu, cuda, mps)
            hf_model_name: HuggingFace model to load
            token: Optional HuggingFace token
        """
        super().__init__(model_id=model_id, device=device, token=token)
        self.hf_model_name = hf_model_name
        self.model_type = "table_qa"
        self.supports_streaming = False
        self._is_loaded = False
        self.tokenizer = None
        self.model = None

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded and self.model is not None

    async def load(self) -> None:
        """Load the TAPAS model and tokenizer."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} already loaded")
            return

        logger.info(f"Loading TAPAS model: {self.hf_model_name}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

        self._is_loaded = True
        logger.info(f"TAPAS model loaded: {self.model_id}")

    def _load_sync(self) -> None:
        """Synchronous model loading (run in executor)."""
        from transformers import TapasForQuestionAnswering, TapasTokenizer

        self.tokenizer = TapasTokenizer.from_pretrained(
            self.hf_model_name,
            token=self.token,
        )

        # TAPAS works best on CPU, has issues with MPS
        device_to_use = self.device
        if self.device == "mps":
            device_to_use = "cpu"

        self.model = TapasForQuestionAnswering.from_pretrained(
            self.hf_model_name,
            token=self.token,
        )
        self.model.to(device_to_use)
        self.model.eval()

    async def unload(self) -> None:
        """Unload the model and free resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        await super().unload()
        self._is_loaded = False

    async def answer(
        self,
        table: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        """Answer a question about a table.

        Args:
            table: Table data with 'columns' and 'rows' keys
                   columns: List of column names
                   rows: List of rows, each row is a list of cell values
            question: Natural language question about the table

        Returns:
            Dictionary with:
                - answer: The answer to the question
                - cells: Cell coordinates used in the answer
                - aggregation: Aggregation type if applicable
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if "columns" not in table or "rows" not in table:
            raise ValueError("Table must have 'columns' and 'rows' keys")

        if not table["columns"]:
            raise ValueError("Table must have at least one column")

        if not table["rows"]:
            raise ValueError("Table must have at least one row")

        if not question.strip():
            raise ValueError("Question cannot be empty")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._answer_sync,
            table,
            question,
        )

    def _answer_sync(
        self,
        table: dict[str, Any],
        question: str,
    ) -> dict[str, Any]:
        """Synchronous question answering."""
        # Convert to pandas DataFrame
        df = pd.DataFrame(table["rows"], columns=table["columns"])

        # Ensure all values are strings for TAPAS
        df = df.astype(str)

        # Tokenize
        inputs = self.tokenizer(
            table=df,
            queries=question,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Move to device
        device_to_use = self.device if self.device != "mps" else "cpu"
        inputs = {k: v.to(device_to_use) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process outputs
        predicted_answer_coordinates, predicted_aggregation_indices = self.tokenizer.convert_logits_to_predictions(
            inputs,
            outputs.logits.detach().cpu(),
            outputs.logits_aggregation.detach().cpu(),
        )

        # Get answer cells
        coordinates = predicted_answer_coordinates[0]
        aggregation_idx = predicted_aggregation_indices[0]

        # Extract cell values
        cell_values = []
        cells = []
        for coord in coordinates:
            row_idx, col_idx = coord
            cells.append({"row": row_idx, "column": col_idx})
            cell_values.append(df.iloc[row_idx, col_idx])

        # Get aggregation type
        aggregation = AGGREGATION_TYPES.get(aggregation_idx, "NONE")

        # Compute final answer
        if aggregation == "NONE":
            answer = ", ".join(cell_values) if cell_values else "No answer found"
        elif aggregation == "COUNT":
            answer = str(len(cell_values))
        elif aggregation == "SUM":
            try:
                total = sum(float(v) for v in cell_values)
                answer = str(total)
            except ValueError:
                answer = ", ".join(cell_values)
        elif aggregation == "AVERAGE":
            try:
                values = [float(v) for v in cell_values]
                avg = sum(values) / len(values) if values else 0
                answer = str(round(avg, 2))
            except ValueError:
                answer = ", ".join(cell_values)
        else:
            answer = ", ".join(cell_values) if cell_values else "No answer found"

        return {
            "answer": answer,
            "cells": cells,
            "cell_values": cell_values,
            "aggregation": aggregation,
            "question": question,
        }

    async def answer_batch(
        self,
        table: dict[str, Any],
        questions: list[str],
    ) -> list[dict[str, Any]]:
        """Answer multiple questions about the same table.

        Args:
            table: Table data with 'columns' and 'rows' keys
            questions: List of questions

        Returns:
            List of answer results (one per question)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        for question in questions:
            result = await self.answer(table, question)
            results.append(result)
        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "hf_model_name": self.hf_model_name,
            "is_loaded": self.is_loaded,
            "supported_aggregations": list(AGGREGATION_TYPES.values()),
        })
        return info
