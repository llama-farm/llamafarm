#!/usr/bin/env python3
"""
Utility to create train/eval splits for datasets.
Usage: python create_data_split.py --input medical_qa_cleaned.jsonl --eval-percent 10
"""

import json
import random
import argparse
from pathlib import Path

def create_split(input_file, eval_percent=10, seed=42, output_dir=None):
    """
    Create train/eval split from a dataset.
    
    Args:
        input_file: Input JSONL file
        eval_percent: Percentage for evaluation (1-50)
        seed: Random seed for reproducibility
        output_dir: Output directory (defaults to input file directory)
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Validate eval percentage
    if not 1 <= eval_percent <= 50:
        raise ValueError("Eval percentage must be between 1 and 50")
    
    # Set output directory
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read all lines
    print(f"Reading {input_path}...")
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total examples: {len(lines)}")
    
    # Shuffle with seed for reproducibility
    random.seed(seed)
    random.shuffle(lines)
    
    # Calculate split point
    split_point = int(len(lines) * (1 - eval_percent / 100))
    train_lines = lines[:split_point]
    eval_lines = lines[split_point:]
    
    # Generate output filenames
    base_name = input_path.stem
    train_file = output_dir / f"{base_name}_train.jsonl"
    eval_file = output_dir / f"{base_name}_eval.jsonl"
    
    # Write train dataset
    with open(train_file, 'w') as f:
        f.writelines(train_lines)
    
    # Write eval dataset
    with open(eval_file, 'w') as f:
        f.writelines(eval_lines)
    
    # Print statistics
    print(f"\n✅ Split created successfully!")
    print(f"📊 Statistics:")
    print(f"  • Train: {len(train_lines)} examples ({100-eval_percent}%)")
    print(f"  • Eval:  {len(eval_lines)} examples ({eval_percent}%)")
    print(f"  • Ratio: {len(train_lines):.1f}:{len(eval_lines):.1f}")
    print(f"\n📁 Output files:")
    print(f"  • Train: {train_file}")
    print(f"  • Eval:  {eval_file}")
    
    # Show sample distribution
    if len(lines) > 0:
        print(f"\n🎲 Random seed: {seed} (use same seed for reproducible splits)")
    
    return train_file, eval_file

def main():
    parser = argparse.ArgumentParser(
        description="Create train/eval splits for fine-tuning datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default 10% evaluation split
  python create_data_split.py --input medical_qa_cleaned.jsonl
  
  # Custom 15% evaluation split
  python create_data_split.py --input medical_qa_cleaned.jsonl --eval-percent 15
  
  # Very small 5% evaluation (more training data)
  python create_data_split.py --input medical_qa_cleaned.jsonl --eval-percent 5
  
  # Large 20% evaluation (more robust evaluation)
  python create_data_split.py --input medical_qa_cleaned.jsonl --eval-percent 20
  
  # Custom output directory
  python create_data_split.py --input data.jsonl --output-dir splits/
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file to split"
    )
    
    parser.add_argument(
        "--eval-percent", "-e",
        type=int,
        default=10,
        help="Percentage for evaluation (1-50, default: 10)"
    )
    
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (defaults to input file directory)"
    )
    
    args = parser.parse_args()
    
    try:
        create_split(
            args.input,
            args.eval_percent,
            args.seed,
            args.output_dir
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())