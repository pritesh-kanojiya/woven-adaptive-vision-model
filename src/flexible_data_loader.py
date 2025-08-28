"""
Simplified Data Loader for MNIST and Hugging Face datasets.

Supported sources:
- Hugging Face datasets: load_dataset("author/dataset_name") e.g., "ylecun/mnist"
- PyTorch MNIST: use "mnist" to load standard torchvision MNIST dataset
"""
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from pathlib import Path

# Hugging Face datasets support
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    load_dataset = None

def should_pin_memory():
    """Determine if pin_memory should be enabled based on CUDA availability"""
    return torch.cuda.is_available()

"""
Simplified Data Loader for MNIST and Hugging Face datasets.

Supported sources:
- Hugging Face datasets: load_dataset("author/dataset_name") e.g., "ylecun/mnist"
- PyTorch MNIST: use "mnist" to load standard torchvision MNIST dataset
"""
import torch
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Hugging Face datasets support
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    load_dataset = None

def should_pin_memory():
    """Determine if pin_memory should be enabled based on CUDA availability"""
    return torch.cuda.is_available()

def load_hf_dataset(dataset_name, config, split_train='train', split_test='test'):
    """
    Load dataset from Hugging Face datasets library
    
    Args:
        dataset_name (str): HF dataset name like 'ylecun/mnist' or 'fashion_mnist'
        config (dict): Configuration dictionary
        split_train (str): Training split name
        split_test (str): Test split name
    
    Returns:
        tuple: (train_loader, test_loader, dataset_info)
    """
    if not HF_DATASETS_AVAILABLE:
        raise RuntimeError("❌ Hugging Face 'datasets' library not installed. Install with: pip install datasets")
    
    print(f"🤗 Loading Hugging Face dataset: {dataset_name}")
    
    try:
        # Load dataset
        dataset = load_dataset(dataset_name)
        print(f"✅ Dataset loaded: {dataset}")
        
        # Extract splits
        if split_train not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(f"❌ Training split '{split_train}' not found. Available splits: {available_splits}")
        
        if split_test not in dataset:
            available_splits = list(dataset.keys())
            raise ValueError(f"❌ Test split '{split_test}' not found. Available splits: {available_splits}")
        
        train_dataset = dataset[split_train]
        test_dataset = dataset[split_test]
        
        print(f"📊 Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
        # Convert to PyTorch format
        data_config = config['data']
        batch_size = data_config.get('batch_size', 64)
        num_workers = data_config.get('num_workers', 2)
        
        # Extract features and labels - assume 'image' and 'label' columns (common HF format)
        if 'image' in train_dataset.features and 'label' in train_dataset.features:
            print("🖼️  Detected image dataset format")
            train_images = []
            train_labels = []
            test_images = []
            test_labels = []
            
            # Convert images to tensors with standardized preprocessing
            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
                transforms.Resize((28, 28)),  # Standardize size
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST-like normalization
            ])
            
            for item in train_dataset:
                img = transform(item['image'])
                train_images.append(img)
                train_labels.append(item['label'])
            
            for item in test_dataset:
                img = transform(item['image'])
                test_images.append(img)
                test_labels.append(item['label'])
            
            train_images = torch.stack(train_images)
            train_labels = torch.tensor(train_labels, dtype=torch.long)
            test_images = torch.stack(test_images)
            test_labels = torch.tensor(test_labels, dtype=torch.long)
            
        else:
            raise ValueError(f"❌ Unsupported HF dataset format. Expected 'image' and 'label' columns, got: {train_dataset.features}")
        
        # Create datasets and loaders
        train_torch_dataset = TensorDataset(train_images, train_labels)
        test_torch_dataset = TensorDataset(test_images, test_labels)
        
        use_pin_memory = should_pin_memory()
        train_loader = DataLoader(train_torch_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=use_pin_memory)
        test_loader = DataLoader(test_torch_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=use_pin_memory)
        
        # Get number of classes
        num_classes = len(torch.unique(train_labels))
        
        dataset_info = {
            'name': dataset_name,
            'format': 'huggingface',
            'train_samples': len(train_torch_dataset),
            'test_samples': len(test_torch_dataset),
            'num_classes': num_classes,
            'input_shape': (1, 28, 28),  # Standardized to grayscale 28x28
            'classes': list(range(num_classes))
        }
        
        print("✅ Hugging Face dataset loaded successfully:")
        print(f"   Train samples: {dataset_info['train_samples']}")
        print(f"   Test samples: {dataset_info['test_samples']}")
        print(f"   Classes: {dataset_info['num_classes']}")
        
        return train_loader, test_loader, dataset_info
        
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load Hugging Face dataset '{dataset_name}': {e}")

def load_mnist_data_pytorch(config):
    """Load standard MNIST dataset using torchvision"""
    print("📁 Loading MNIST dataset...")
    
    # Extract configuration
    data_config = config['data']
    batch_size = data_config.get('batch_size', 64)
    num_workers = data_config.get('num_workers', 2)
    data_dir = data_config.get('data_dir', './data')
    download = data_config.get('download', True)
    
    print(f"   Batch size: {batch_size}")
    print(f"   Data directory: {data_dir}")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define data transformations
    print("🔄 Setting up data transformations...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    print("📥 Downloading/loading training data...")
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    
    print("📥 Downloading/loading test data...")
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform
    )
    
    # Check for quick mode to create smaller subsets
    if os.environ.get('QUICK_MODE') == 'true':
        from torch.utils.data import Subset
        print("🧪 Quick mode enabled - creating smaller dataset subsets")
        
        # Create smaller subsets for quick testing
        train_subset_size = min(1000, len(train_dataset))
        test_subset_size = min(500, len(test_dataset))
        
        train_dataset = Subset(train_dataset, range(train_subset_size))
        test_dataset = Subset(test_dataset, range(test_subset_size))
    
    # Create data loaders
    print("⚙️  Creating data loaders...")
    use_pin_memory = should_pin_memory()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    # Create dataset info
    dataset_info = {
        'name': 'MNIST',
        'format': 'pytorch',
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'classes': list(range(10))
    }
    
    print("✅ Data loading complete!")
    print(f"   Training samples: {dataset_info['train_samples']:,}")
    print(f"   Test samples: {dataset_info['test_samples']:,}")
    print(f"   Classes: {dataset_info['num_classes']}")
    
    return train_loader, test_loader, dataset_info

def load_flexible_data(config, dataset_name=''):
    """
    Load data - supports MNIST and Hugging Face datasets only
    
    Args:
        config (dict): Configuration dictionary
        dataset_name (str): Either 'mnist' for PyTorch MNIST or HF dataset name (e.g., 'ylecun/mnist')
    
    Returns:
        tuple: (train_loader, test_loader, dataset_info)
    """
    if not dataset_name:
        raise RuntimeError("❌ No dataset specified. Provide 'mnist' for PyTorch MNIST or HF dataset name (e.g., 'ylecun/mnist')")
    
    # If dataset_name is "mnist", use PyTorch MNIST
    if dataset_name.lower() == 'mnist':
        print("📁 Using PyTorch MNIST dataset")
        return load_mnist_data_pytorch(config)
    
    # Otherwise, treat as Hugging Face dataset (should contain "/" like "author/dataset")
    elif "/" in dataset_name:
        print(f"🤗 Using Hugging Face dataset: {dataset_name}")
        return load_hf_dataset(dataset_name, config)
    
    else:
        raise RuntimeError(f"❌ Unsupported dataset format: '{dataset_name}'. Use 'mnist' or HF format 'author/dataset'")
