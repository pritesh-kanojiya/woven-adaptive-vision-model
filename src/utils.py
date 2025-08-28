"""
Utility Functions for MNIST MLOps Pipeline
"""
import yaml
import os
import logging
import json
from datetime import datetime


def load_config(config_path='config/config.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    print(f"📋 Loading configuration from {config_path}...")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        print("✅ Configuration loaded successfully!")
        return config
        
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML file: {e}")
        raise


def setup_logging(config):
    """
    Set up logging based on configuration
    
    Args:
        config (dict): Configuration dictionary
    """
    log_config = config['logging']
    log_level = log_config.get('level', 'INFO')
    save_logs = log_config.get('save_logs', False)
    log_dir = log_config.get('log_dir', './artifacts/logs')
    
    # Create log directory if saving logs
    if save_logs:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(
                os.path.join(log_dir, 'training.log')
            ) if save_logs else logging.NullHandler()
        ]
    )
    
    print(f"📝 Logging configured - Level: {log_level}")
    if save_logs:
        print(f"   Log file: {os.path.join(log_dir, 'training.log')}")


def create_run_id():
    """
    Create a unique run ID for this training session
    
    Returns:
        str: Unique run ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"mnist_run_{timestamp}"
    print(f"🆔 Run ID: {run_id}")
    return run_id


def save_metrics(metrics, config, run_id):
    """
    Save training metrics to file
    
    Args:
        metrics (dict): Metrics dictionary
        config (dict): Configuration dictionary
        run_id (str): Unique run identifier
    """
    artifacts_dir = config['artifacts']['save_dir']
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Add run info to metrics
    metrics['run_id'] = run_id
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['config'] = config
    
    # Save metrics
    metrics_path = os.path.join(artifacts_dir, f'metrics_{run_id}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Metrics saved to {metrics_path}")
    return metrics_path


def check_accuracy_threshold(accuracy, config):
    """
    Check if model meets accuracy requirements
    
    Args:
        accuracy (float): Model accuracy
        config (dict): Configuration dictionary
        
    Returns:
        bool: Whether accuracy meets threshold
    """
    required_accuracy = config['evaluation']['required_accuracy']
    
    print(f"🎯 Accuracy check:")
    print(f"   Achieved: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Required: {required_accuracy:.4f} ({required_accuracy*100:.2f}%)")
    
    meets_threshold = accuracy >= required_accuracy
    
    if meets_threshold:
        print("✅ Model meets accuracy threshold!")
    else:
        print("❌ Model does not meet accuracy threshold")
        print(f"   Gap: {(required_accuracy - accuracy)*100:.2f} percentage points")
    
    return meets_threshold


def print_system_info():
    """
    Print system information for debugging
    """
    import torch
    import platform
    
    print("🖥️  System Information:")
    print(f"   Python version: {platform.python_version()}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Platform: {platform.platform()}")


def validate_config(config):
    """
    Validate configuration dictionary
    
    Args:
        config (dict): Configuration to validate
        
    Returns:
        bool: Whether configuration is valid
    """
    print("🔍 Validating configuration...")
    
    required_sections = ['data', 'model', 'training', 'evaluation', 'artifacts']
    
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required section: {section}")
            return False
    
    # Validate specific fields
    if config['training']['learning_rate'] <= 0:
        print("❌ Learning rate must be positive")
        return False
    
    if config['training']['max_epochs'] <= 0:
        print("❌ Max epochs must be positive")
        return False
    
    if config['data']['batch_size'] <= 0:
        print("❌ Batch size must be positive")
        return False
    
    print("✅ Configuration is valid!")
    return True


def create_artifact_directories(config):
    """
    Create necessary directories for artifacts
    
    Args:
        config (dict): Configuration dictionary
    """
    print("📁 Creating artifact directories...")
    
    base_dir = config['artifacts']['save_dir']
    subdirs = ['checkpoints', 'logs', 'plots', 'reports']
    
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)
        print(f"   Created: {dir_path}")
    
    print("✅ Artifact directories ready!")


if __name__ == "__main__":
    print("🧪 Testing utility functions...")
    
    # Test config loading
    try:
        config = load_config('config/config.yaml')
        print("✅ Config loading test passed!")
        
        # Test validation
        is_valid = validate_config(config)
        print(f"✅ Config validation test: {'passed' if is_valid else 'failed'}!")
        
        # Test directory creation
        create_artifact_directories(config)
        print("✅ Directory creation test passed!")
        
        # Test run ID creation
        run_id = create_run_id()
        print("✅ Run ID creation test passed!")
        
        # Print system info
        print_system_info()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
