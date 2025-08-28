"""
Main Entry Point for MNIST MLOps Pipeline
"""
import sys
import os
import argparse

# Add src directory to path so we can import our modules
sys.path.append('src')

from model import create_model
from flexible_data_loader import load_flexible_data
from trainer import SimpleTrainer
from utils import (
    load_config, setup_logging, create_run_id, save_metrics,
    check_accuracy_threshold, print_system_info, validate_config,
    create_artifact_directories
)


def main():
    """
    Main function that orchestrates the complete ML pipeline
    """
    parser = argparse.ArgumentParser(description='MNIST MLOps Pipeline')
    parser.add_argument('command', choices=['train', 'quick-test', 'serve'], 
                       help='Command to execute')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset-format', default=None,
                       help='Dataset format to use (auto-detected if not specified)')
    parser.add_argument('--dataset-url', default='',
                       help='Custom dataset URL to download from')
    parser.add_argument('--dataset-name', default='',
                       help='Hugging Face dataset name (e.g., ylecun/mnist, Voxel51/emnist-letters-tiny)')
    
    args = parser.parse_args()
    
    # Get dataset parameters from environment variables (set by GitHub Actions)
    dataset_format = os.environ.get('DATASET_FORMAT', args.dataset_format)
    dataset_url = os.environ.get('DATASET_URL', args.dataset_url)
    dataset_name = os.environ.get('DATASET_NAME', args.dataset_name)
    
    # Auto-detection message
    if dataset_format is None and dataset_url and not dataset_name:
        print("🤖 Dataset format not specified - will auto-detect from downloaded data")
    elif dataset_format is None and dataset_name:
        print("🤖 Using Hugging Face dataset - format will be auto-handled")
    elif dataset_format is None:
        print("🤖 Dataset format not specified - will auto-detect from local data")
    
    if args.command == 'serve':
        print("🚀 Starting Woven Adaptive Vision Model Server...")
        import uvicorn
        uvicorn.run("inference.api:app", host="0.0.0.0", port=8000, reload=True)
        return True
    
    # Handle quick-test command separately
    if args.command == 'quick-test':
        print("🧪 Running in quick test mode (reduced dataset)")
        return quick_test()
    
    print("🚀 MNIST MLOps Pipeline - Starting...")
    print("=" * 60)
    if dataset_name:
        print(f"🤗 Hugging Face Dataset: {dataset_name}")
    elif dataset_url:
        print(f"🔗 Dataset URL: {dataset_url}")
    else:
        print(f"📁 Using standard/local dataset")
    print(f"📊 Dataset Format: {dataset_format}")
    print("=" * 60)
    
    # Step 1: Load and validate configuration
    print("\n📋 Step 1: Loading Configuration")
    try:
        config = load_config(args.config)
        
        # Update config with environment variables (from GitHub Actions)
        if 'LEARNING_RATE' in os.environ:
            learning_rate = float(os.environ['LEARNING_RATE'])
            config['training']['learning_rate'] = learning_rate
            print(f"🔧 Updated learning_rate from environment: {learning_rate}")
        
        if 'MAX_EPOCHS' in os.environ:
            max_epochs = int(os.environ['MAX_EPOCHS'])
            config['training']['max_epochs'] = max_epochs
            print(f"🔧 Updated max_epochs from environment: {max_epochs}")
        
        if 'REQUIRED_ACCURACY' in os.environ:
            required_accuracy = float(os.environ['REQUIRED_ACCURACY'])
            config['evaluation']['required_accuracy'] = required_accuracy
            print(f"🔧 Updated required_accuracy from environment: {required_accuracy}")
        
        if not validate_config(config):
            print("❌ Invalid configuration. Exiting.")
            return False
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # Step 2: Set up environment
    print("\n🔧 Step 2: Setting Up Environment")
    run_id = create_run_id()
    create_artifact_directories(config)
    setup_logging(config)
    print_system_info()
    
    # Log training configuration
    import logging
    logging.info("=" * 50)
    logging.info("TRAINING CONFIGURATION")
    logging.info("=" * 50)
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Learning Rate: {config['training']['learning_rate']}")
    logging.info(f"Max Epochs: {config['training']['max_epochs']}")
    logging.info(f"Required Accuracy: {config['evaluation']['required_accuracy']}")
    logging.info(f"Device: {config['training']['device']}")
    logging.info(f"Run ID: {run_id}")
    logging.info("=" * 50)
    
    # Step 3: Load data
    print("\n📁 Step 3: Loading Data")
    try:
        # Use simplified data loader that supports MNIST and Hugging Face datasets
        train_loader, test_loader, dataset_info = load_flexible_data(
            config, 
            dataset_name=dataset_name
        )
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    # Step 4: Create model
    print("\n🏗️  Step 4: Creating Model")
    try:
        # Update config with dynamic number of classes from dataset
        config['model']['num_classes'] = dataset_info['num_classes']
        print(f"🔧 Updated model config: num_classes = {dataset_info['num_classes']}")
        
        model = create_model(config)
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False
    
    # Step 5: Set up trainer
    print("\n🚀 Step 5: Setting Up Trainer")
    try:
        trainer = SimpleTrainer(model, config)
    except Exception as e:
        print(f"❌ Failed to set up trainer: {e}")
        return False
    
    # Step 6: Train model
    print("\n🎯 Step 6: Training Model")
    try:
        history = trainer.train(train_loader, test_loader)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Step 7: Final validation
    print("\n📊 Step 7: Final Validation")
    try:
        final_loss, final_accuracy, final_f1 = trainer.validate(test_loader)
        
        # Check if model meets requirements
        meets_threshold = check_accuracy_threshold(final_accuracy, config)
        
        # Create metrics summary
        metrics = {
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'final_f1_score': final_f1,
            'training_history': history,
            'dataset_info': dataset_info,
            'meets_threshold': meets_threshold
        }
        
        # Save metrics
        metrics_path = save_metrics(metrics, config, run_id)
        
    except Exception as e:
        print(f"❌ Final validation failed: {e}")
        return False
    
    # Step 8: Generate plots and summary
    print("\n📈 Step 8: Generating Reports")
    try:
        trainer.plot_training_history()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Final Results:")
        print(f"   Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"   Loss: {final_loss:.4f}")
        print(f"   F1-Score: {final_f1:.4f}")
        print(f"   Training Time: {history['total_time']:.1f}s")
        print(f"   Epochs: {history['epochs_trained']}")
        print(f"   Final Accuracy: {history['final_accuracy']:.4f}")
        
        print(f"\n📁 Artifacts saved in: {config['artifacts']['save_dir']}")
        print(f"📊 Metrics file: {metrics_path}")
        
        if meets_threshold:
            print("\n✅ MODEL READY FOR DEPLOYMENT!")
            print("   The model meets the required accuracy threshold.")
        else:
            print("\n⚠️  MODEL NEEDS IMPROVEMENT")
            print("   Consider adjusting hyperparameters or training longer.")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return False
    
    # Clean up artifacts and data if in quick mode
    if os.environ.get('QUICK_MODE') == 'true':
        print("\n🧹 Quick mode cleanup - removing artifacts and data...")
        try:
            import shutil
            cleaned_items = []
            
            # Remove artifacts directory
            if os.path.exists('./artifacts'):
                shutil.rmtree('./artifacts')
                cleaned_items.append("artifacts directory")
            
            # Remove data directory
            if os.path.exists('./data'):
                shutil.rmtree('./data')
                cleaned_items.append("data directory")
            
            if cleaned_items:
                print(f"   ✅ Removed: {', '.join(cleaned_items)}")
            else:
                print("   ℹ️  No artifacts or data directories found")
                
        except Exception as e:
            print(f"   ⚠️  Cleanup warning: {e}")
    
    return True


def train_command():
    """Command for training the model"""
    print("🎯 Starting training process...")
    success = main()
    if success:
        print("✅ Training completed successfully!")
        return True
    else:
        print("❌ Training failed!")
        return False


def evaluate_command():
    """Command for evaluating a trained model"""
    print("📊 Starting evaluation process...")
    # For now, this just runs the full pipeline
    # In a more advanced version, this would load a saved model
    success = main()
    if success:
        print("✅ Evaluation completed successfully!")
        return True
    else:
        print("❌ Evaluation failed!")
        return False


def quick_test():
    """
    Quick test to verify everything is working with minimal dataset
    """
    print("🧪 Running quick test with minimal dataset...")
    
    # Test with minimal configuration
    test_config = {
        'data': {'batch_size': 32, 'num_workers': 0, 'data_dir': './data', 'download': True},  # No workers for speed
        'model': {'input_channels': 1, 'num_classes': 10},
        'training': {'learning_rate': 0.01, 'max_epochs': 1, 'device': 'cpu'},
        'evaluation': {'required_accuracy': 0.1, 'metrics': ['accuracy']},
        'artifacts': {'save_dir': './test_artifacts'},
        'logging': {'level': 'INFO', 'save_logs': False}
    }
    
    try:
        # Create test directories
        os.makedirs('./test_artifacts', exist_ok=True)
        
        # Load minimal dataset for quick testing
        print("   Loading minimal dataset (1000 train, 500 test samples)...")
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, transforms
        
        # Create transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load full datasets but create subsets
        full_train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        full_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # Create small subsets for quick testing
        train_subset = Subset(full_train_dataset, range(1000))  # 1000 samples
        test_subset = Subset(full_test_dataset, range(500))     # 500 samples
        
        # Create data loaders with no pin_memory to avoid warnings
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
        
        print(f"✅ Quick test dataset loaded:")
        print(f"   Training samples: {len(train_subset):,}")
        print(f"   Test samples: {len(test_subset):,}")
        print(f"   Classes: 10")
        print(f"   Input shape: (1, 28, 28)")
        
        print("   Creating model...")
        model = create_model(test_config)
        
        print("   Setting up trainer...")
        trainer = SimpleTrainer(model, test_config)
        
        print("   Running 1 epoch...")
        history = trainer.train(train_loader, test_loader)
        
        print("✅ Quick test passed! Everything is working.")
        
        # Clean up test artifacts
        print("🧹 Cleaning up test artifacts...")
        import shutil
        if os.path.exists('./test_artifacts'):
            shutil.rmtree('./test_artifacts')
        if os.path.exists('./artifacts'):
            # Only remove quick test artifacts, keep structure for actual training
            quick_test_files = ['training_history.png', 'metrics_*.json']
            for pattern in quick_test_files:
                import glob
                for file in glob.glob(f'./artifacts/{pattern}'):
                    try:
                        os.remove(file)
                        print(f"   Removed: {file}")
                    except OSError:
                        pass
        print("🧹 Quick test cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    # Check if we have command line arguments
    if len(sys.argv) > 1:
        success = main()  # Use the new argument-parsing main function
        sys.exit(0 if success else 1)  # Exit with proper exit code
    else:
        # Default behavior - run training
        success = train_command()
        sys.exit(0 if success else 1)  # Exit with proper exit code
