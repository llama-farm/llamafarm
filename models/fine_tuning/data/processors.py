"""
Dataset processors for fine-tuning.

This module provides processors for different dataset formats used in fine-tuning.
"""

import json
import csv
from typing import Dict, Any, List, Union, Optional
from pathlib import Path
import logging

try:
    from datasets import Dataset, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    # Create dummy Dataset class for type hints
    class Dataset:
        pass

from ..core.base import BaseDataProcessor

logger = logging.getLogger(__name__)


class DatasetProcessor(BaseDataProcessor):
    """Base dataset processor with common functionality."""
    
    def get_dataset_stats(self, dataset) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        if not DATASETS_AVAILABLE:
            return {"error": "datasets library not available"}
        
        if not isinstance(dataset, Dataset):
            return {"error": "Invalid dataset type"}
        
        stats = {
            "num_examples": len(dataset),
            "columns": dataset.column_names,
            "features": str(dataset.features)
        }
        
        # Get sample lengths if text column exists
        if "text" in dataset.column_names:
            texts = dataset["text"][:100]  # Sample first 100
            lengths = [len(text.split()) for text in texts]
            stats["avg_length_words"] = sum(lengths) / len(lengths)
            stats["max_length_words"] = max(lengths)
            stats["min_length_words"] = min(lengths)
        
        return stats


class JSONLProcessor(DatasetProcessor):
    """Processor for JSONL (JSON Lines) datasets."""
    
    def process_dataset(self, dataset_path: Union[str, Path]) -> Dataset:
        """Process JSONL dataset."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library is required for processing")
        
        path = Path(dataset_path)
        logger.info(f"Processing JSONL dataset: {path}")
        
        # Load JSONL file
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        
        # Create dataset
        dataset = Dataset.from_list(data)
        
        # Apply any configured preprocessing
        dataset = self._preprocess_dataset(dataset)
        
        logger.info(f"Processed {len(dataset)} examples from JSONL")
        return dataset
    
    def validate_dataset(self, dataset_path: Union[str, Path]) -> List[str]:
        """Validate JSONL dataset."""
        errors = []
        path = Path(dataset_path)
        
        if not path.exists():
            errors.append(f"Dataset file does not exist: {path}")
            return errors
        
        if not path.is_file():
            errors.append(f"Dataset path is not a file: {path}")
            return errors
        
        # Check file contents
        try:
            with open(path, 'r', encoding='utf-8') as f:
                line_count = 0
                valid_lines = 0
                
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    try:
                        json.loads(line.strip())
                        valid_lines += 1
                    except json.JSONDecodeError:
                        if line_num <= 10:  # Only report first 10 errors
                            errors.append(f"Invalid JSON on line {line_num}")
                
                if line_count == 0:
                    errors.append("Dataset file is empty")
                elif valid_lines == 0:
                    errors.append("No valid JSON lines found")
                elif valid_lines < line_count * 0.8:  # Less than 80% valid
                    errors.append(f"Many invalid lines: {valid_lines}/{line_count} valid")
                
        except Exception as e:
            errors.append(f"Could not read dataset file: {e}")
        
        return errors
    
    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Apply preprocessing to the dataset."""
        config = self.config
        
        # Filter by length
        min_length = config.get("min_length", 0)
        max_length = config.get("max_length", float('inf'))
        
        if min_length > 0 or max_length < float('inf'):
            def filter_by_length(example):
                text = example.get("text", "")
                length = len(text.split())
                return min_length <= length <= max_length
            
            dataset = dataset.filter(filter_by_length)
        
        # Filter empty examples
        if config.get("filter_empty", True):
            def filter_empty(example):
                text = example.get("text", "").strip()
                return len(text) > 0
            
            dataset = dataset.filter(filter_empty)
        
        return dataset


class AlpacaProcessor(DatasetProcessor):
    """Processor for Alpaca-format datasets."""
    
    def process_dataset(self, dataset_path: Union[str, Path]) -> Dataset:
        """Process Alpaca format dataset."""
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library is required for processing")
        
        path = Path(dataset_path)
        logger.info(f"Processing Alpaca dataset: {path}")
        
        # Load data
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif path.suffix.lower() == '.jsonl':
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Convert to conversation format
        converted_data = []
        template = self.config.get("conversation_template", "alpaca")
        
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            if template == "alpaca":
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            elif template == "llama3":
                if input_text:
                    text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
                else:
                    text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
            else:
                # Keep original format
                converted_data.append(item)
                continue
            
            converted_data.append({"text": text, **item})
        
        # Create dataset
        dataset = Dataset.from_list(converted_data)
        
        # Apply preprocessing
        dataset = self._preprocess_dataset(dataset)
        
        logger.info(f"Processed {len(dataset)} examples from Alpaca format")
        return dataset
    
    def validate_dataset(self, dataset_path: Union[str, Path]) -> List[str]:
        """Validate Alpaca dataset."""
        errors = []
        path = Path(dataset_path)
        
        if not path.exists():
            errors.append(f"Dataset file does not exist: {path}")
            return errors
        
        try:
            # Load and check structure
            if path.suffix.lower() == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif path.suffix.lower() == '.jsonl':
                data = []
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                errors.append(f"Unsupported file format: {path.suffix}")
                return errors
            
            if not isinstance(data, list):
                errors.append("Dataset must be a list of examples")
                return errors
            
            if len(data) == 0:
                errors.append("Dataset is empty")
                return errors
            
            # Check required fields
            required_fields = ["instruction", "output"]
            sample_size = min(10, len(data))
            
            for i, item in enumerate(data[:sample_size]):
                if not isinstance(item, dict):
                    errors.append(f"Example {i} is not a dictionary")
                    continue
                
                for field in required_fields:
                    if field not in item:
                        errors.append(f"Example {i} missing required field: {field}")
                    elif not isinstance(item[field], str):
                        errors.append(f"Example {i} field {field} is not a string")
        
        except Exception as e:
            errors.append(f"Could not validate dataset: {e}")
        
        return errors
    
    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Apply Alpaca-specific preprocessing."""
        config = self.config
        
        # Filter by length
        min_length = config.get("min_length", 0)
        max_length = config.get("max_length", float('inf'))
        
        if min_length > 0 or max_length < float('inf'):
            def filter_by_length(example):
                text = example.get("text", "")
                length = len(text.split())
                return min_length <= length <= max_length
            
            dataset = dataset.filter(filter_by_length)
        
        # Filter empty examples
        if config.get("filter_empty", True):
            def filter_empty(example):
                text = example.get("text", "").strip()
                return len(text) > 0
            
            dataset = dataset.filter(filter_empty)
        
        return dataset


# Don't auto-register to avoid circular imports
# Registration will be done in the factory module