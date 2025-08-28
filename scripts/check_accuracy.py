#!/usr/bin/env python3
"""
Model accuracy validation script for woven adaptive vision models pipeline.
Checks model performance and sets outputs for workflow decisions.
"""

import json
import glob
import os
import sys

def check_model_accuracy():
    """Check model accuracy against requirements and set GitHub Actions outputs."""
    try:
        print("📊 Checking model accuracy...")
        
        # Find the metrics file
        metrics_files = glob.glob('artifacts/metrics_*.json')
        if not metrics_files:
            print('❌ No metrics file found!')
            sys.exit(1)
            
        with open(metrics_files[0], 'r') as f:
            metrics = json.load(f)
            
        accuracy = metrics['final_accuracy']
        required_accuracy = float(os.environ.get('REQUIRED_ACCURACY', '0.95'))
        model_version = os.environ.get('MODEL_VERSION', 'v2025.01.01-unknown_1')
        loss = round(metrics.get('final_loss', 0.0), 3)
        training_time = round(metrics.get('training_history', {}).get('total_time', 0.0), 3)
        accuracy_pct = round(accuracy * 100, 2)
        required_accuracy_pct = round(required_accuracy * 100, 2)

        print(f'Model: woven-adaptive-mnist-model {model_version}')
        print(f'Accuracy: {accuracy_pct:.2f}%')
        print(f'Required: {required_accuracy_pct:.2f}%')
        print(f'Loss: {loss:.3f}')
        print(f'Training Time: {training_time:.3f} seconds')

        # Set outputs for next steps
        github_output = os.environ.get('GITHUB_OUTPUT')
        if github_output:
            with open(github_output, 'a') as f:
                f.write(f'accuracy={accuracy_pct}\n')
                f.write(f'meets_threshold={accuracy >= required_accuracy}\n')
                f.write(f'loss={loss}\n')
                f.write(f'training_time={training_time}\n')
                f.write(f'model_name=woven-adaptive-mnist-model\n')
                f.write(f'model_version={model_version}\n')
        
        if accuracy >= required_accuracy:
            print('Model meets accuracy requirements!')
            return True
        else:
            print('Model does not meet accuracy requirements!')
            sys.exit(1)
            
    except Exception as e:
        print(f"Error checking accuracy: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_model_accuracy()
