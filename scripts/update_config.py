#!/usr/bin/env python3
"""
Configuration update script for woven adaptive vision models training pipeline.
Updates runtime configuration parameters from environment variables.
"""

import yaml
import os
import sys

def update_config():
    """Update configuration with runtime parameters from environment variables."""
    try:
        print("🔧 Updating configuration with runtime parameters...")
        
        # Load existing config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Update with runtime parameters
        config['training']['learning_rate'] = float(os.environ.get('LEARNING_RATE', '0.001'))
        config['training']['max_epochs'] = int(os.environ.get('MAX_EPOCHS', '10'))
        config['evaluation']['required_accuracy'] = float(os.environ.get('REQUIRED_ACCURACY', '0.95'))
        
        # Update model name
        config['model']['name'] = os.environ.get('MODEL_NAME')
        config['model']['version'] = f"v1.0.{os.environ.get('MODEL_VERSION', '1')}"
        
        # Save updated config
        with open('config/config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print('✅ Configuration updated successfully')
        print(f"📋 Model: {config['model']['name']} {config['model']['version']}")
        print(f"📊 Learning Rate: {config['training']['learning_rate']}")
        print(f"🔄 Max Epochs: {config['training']['max_epochs']}")
        print(f"🎯 Required Accuracy: {config['evaluation']['required_accuracy']}")
        
    except Exception as e:
        print(f"❌ Error updating configuration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    update_config()
