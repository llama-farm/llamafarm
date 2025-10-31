"""Example Python module for testing RAG ingestion."""

import numpy as np
from typing import List, Dict, Any

class DataProcessor:
    """Process and transform data for machine learning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.batch_size = config.get('batch_size', 32)
        self.normalize = config.get('normalize', True)
    
    def process_batch(self, data: np.ndarray) -> np.ndarray:
        """Process a batch of data."""
        if self.normalize:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            data = (data - mean) / (std + 1e-8)
        return data
    
    def transform(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        """Transform multiple inputs."""
        results = []
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            processed = [self.process_batch(item) for item in batch]
            results.extend(processed)
        return results

def main():
    """Main entry point."""
    processor = DataProcessor({'batch_size': 16, 'normalize': True})
    test_data = [np.random.randn(100, 10) for _ in range(5)]
    results = processor.transform(test_data)
    print(f"Processed {len(results)} batches")

if __name__ == "__main__":
    main()
