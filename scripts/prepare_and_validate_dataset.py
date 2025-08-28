#!/usr/bin/env python3
"""
Simplified dataset preparation and validation for MNIST and Hugging Face datasets.

This script validates dataset inputs for the training pipeline.
Supports only:
- Standard MNIST (no validation required - handled by torchvision)
- Hugging Face datasets (validates dataset exists and has required format)

Inputs (env):
  DATASET_NAME             HF dataset name (e.g., 'ylecun/mnist') or 'mnist' for PyTorch

Outputs (GITHUB_OUTPUT):
  prepared_data_dir        Directory to use for training
  detected_format          Detected/validated dataset format
"""
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.append(str(ROOT_DIR / 'src'))


def eprint(msg: str):
    print(msg, file=sys.stderr)


def validate_hf_dataset(dataset_name):
    """Validate Hugging Face dataset"""
    try:
        from datasets import load_dataset
    except ImportError:
        eprint("❌ Hugging Face 'datasets' library not installed. Install with: pip install datasets")
        sys.exit(1)
    
    print(f"🤗 Validating Hugging Face dataset: {dataset_name}")
    try:
        ds = load_dataset(dataset_name)
        
        # Check for splits
        if 'train' not in ds or 'test' not in ds:
            eprint(f"❌ Dataset '{dataset_name}' missing required splits ('train', 'test'). Found: {list(ds.keys())}")
            sys.exit(1)
        
        # Check for columns
        sample = ds['train'][0]
        if not ('image' in sample and 'label' in sample):
            eprint(f"❌ Dataset '{dataset_name}' missing required columns ('image', 'label'). Found: {list(sample.keys())}")
            sys.exit(1)
        
        print(f"✅ Hugging Face dataset '{dataset_name}' is valid. Train samples: {len(ds['train'])}, Test samples: {len(ds['test'])}")
        
    except Exception as e:
        eprint(f"❌ Failed to load Hugging Face dataset '{dataset_name}': {e}")
        sys.exit(1)


def main():
    """Main validation logic"""
    dataset_name = os.environ.get('DATASET_NAME', '').strip()
    
    # If no dataset specified, default to MNIST
    if not dataset_name:
        dataset_name = 'mnist'
        print("ℹ️  No dataset specified, defaulting to MNIST")
    
    # Handle different dataset types
    if dataset_name.lower() == 'mnist':
        print("📁 Using PyTorch MNIST dataset - no validation required")
        detected_format = 'pytorch'
        prepared_data_dir = ''
        
    elif '/' in dataset_name:
        print(f"� Validating Hugging Face dataset: {dataset_name}")
        validate_hf_dataset(dataset_name)
        detected_format = 'huggingface'
        prepared_data_dir = ''
        
    else:
        eprint(f"❌ Unsupported dataset: '{dataset_name}'")
        eprint("✅ Supported datasets:")
        eprint("- 'mnist' for PyTorch MNIST dataset")
        eprint("- 'author/dataset' for Hugging Face datasets (e.g., 'ylecun/mnist')")
        sys.exit(1)
    
    # Emit outputs for GitHub Actions
    out = os.environ.get('GITHUB_OUTPUT')
    if out:
        with open(out, 'a') as f:
            f.write(f"prepared_data_dir={prepared_data_dir}\n")
            f.write(f"detected_format={detected_format}\n")
    
    print(f"✅ Dataset validation completed")
    print(f"� Format: {detected_format}")
    if prepared_data_dir:
        print(f"� Data directory: {prepared_data_dir}")


if __name__ == '__main__':
    main()
