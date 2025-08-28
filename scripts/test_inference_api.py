#!/usr/bin/env python3
"""
API Inference Test Script for MNIST Model
Designed to work with GitHub Actions workflow
"""

import numpy as np
from PIL import Image
import json
import requests
import sys
import os
import time
import argparse

def setup_logging():
    """Setup basic logging"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def convert_image_to_array(image_path):
    """Convert PNG image to normalized 784-element array"""
    logger.info(f"📸 Converting {image_path} to normalized array...")
    
    # Load image as grayscale and resize to 28x28
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    
    # Convert to numpy array and normalize to 0-1
    img_array = np.array(img) / 255.0
    
    # Flatten to 784 values
    flattened = img_array.flatten().tolist()
    
    return flattened

def test_single_image(image_path, expected_digit, image_id, api_url="http://localhost:8000"):
    """Test a single image against the API"""
    logger.info(f"🎯 Testing image {image_id}: {os.path.basename(image_path)} (expected: {expected_digit})")
    
    try:
        # Convert image to array
        start_time = time.time()
        image_data = convert_image_to_array(image_path)
        
        # Create payload
        payload = {'data': image_data}
        
        # Make API call
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        response_time = time.time() - start_time
        
        # Check response
        if response.status_code != 200:
            logger.error(f"  ❌ API Error: {response.status_code} - {response.text}")
            return None
        
        result = response.json()
        prediction = result['prediction']
        confidence = result['confidence']
        
        # Check if prediction is correct
        is_correct = prediction == expected_digit
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        
        logger.info(f"  Result: True={expected_digit}, Predicted={prediction}, Confidence={confidence:.3f}, {status}, Time={response_time:.3f}s")
        
        return {
            'image_id': image_id,
            'true_label': expected_digit,
            'predicted_label': prediction,
            'confidence': confidence,
            'is_correct': is_correct,
            'response_time': response_time
        }
            
    except Exception as e:
        logger.error(f"  ❌ Error testing {image_path}: {e}")
        return None

def check_api_health(api_url="http://localhost:8000"):
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"✅ API Health: {health_data}")
            return health_data.get('model_loaded', False)
        else:
            logger.error(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Cannot reach API: {e}")
        return False

def load_ground_truth(ground_truth_path):
    """Load ground truth data"""
    try:
        with open(ground_truth_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Cannot load ground truth from {ground_truth_path}: {e}")
        return None

def save_results_csv(results, output_path="results.csv"):
    """Save results to CSV file"""
    with open(output_path, 'w') as f:
        f.write('image_id,true_label,predicted_label,confidence,is_correct,response_time\n')
        for result in results:
            f.write(f"{result['image_id']},{result['true_label']},{result['predicted_label']},{result['confidence']},{result['is_correct']},{result['response_time']:.3f}\n")
    
    logger.info(f"📄 Results saved to {output_path}")

def write_github_env(success_count, total_count, accuracy, threshold, passed):
    """Write results to GitHub environment file"""
    github_env = os.environ.get('GITHUB_ENV')
    if github_env:
        with open(github_env, 'a') as f:
            f.write(f"SUCCESS_COUNT={success_count}\n")
            f.write(f"TOTAL_COUNT={total_count}\n")
            f.write(f"ACCURACY={accuracy:.3f}\n")
            f.write(f"RESULT={'PASS' if passed else 'FAIL'}\n")
        logger.info(f"📝 Results written to GitHub environment")

def main():
    parser = argparse.ArgumentParser(description='Test MNIST API inference')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--images-dir', default='test_images', help='Directory containing test images')
    parser.add_argument('--ground-truth', default='test_images/ground_truth.json', help='Ground truth JSON file')
    parser.add_argument('--threshold', type=float, help='Accuracy threshold (overrides environment variable)')
    parser.add_argument('--output', default='results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    logger.info("🧪 Testing MNIST API with Python script...")
    
    # Check if API is healthy
    if not check_api_health(args.api_url):
        logger.error("❌ API is not healthy. Please check the container.")
        sys.exit(1)
    
    # Load ground truth
    ground_truth = load_ground_truth(args.ground_truth)
    if ground_truth is None:
        sys.exit(1)
    
    logger.info(f"\n🧪 Starting tests on {len(ground_truth)} images...")
    
    results = []
    success_count = 0
    total_count = 0
    
    # Test each image
    for item in ground_truth:
        filename = os.path.basename(item['image_path'])
        expected_digit = item['true_label']
        image_path = os.path.join(args.images_dir, filename)
        
        if not os.path.exists(image_path):
            logger.warning(f"⚠️  Image not found: {image_path}")
            continue
            
        result = test_single_image(image_path, expected_digit, total_count, args.api_url)
        if result:
            results.append(result)
            if result['is_correct']:
                success_count += 1
            total_count += 1
    
    # Save results to CSV
    if results:
        save_results_csv(results, args.output)
    
    # Summary
    logger.info(f"\n==========================================")
    logger.info(f"📊 INFERENCE TEST RESULTS:")
    logger.info(f"✅ Successful tests: {success_count}/{total_count}")
    
    if total_count == 0:
        logger.error("❌ CRITICAL: No test images found!")
        sys.exit(1)
    
    accuracy = success_count / total_count
    accuracy_percent = int(accuracy * 100)
    
    logger.info(f"📈 Accuracy: {accuracy:.3f} ({accuracy_percent}%)")
    
    # Check threshold
    if args.threshold is not None:
        expected_accuracy = args.threshold
    else:
        expected_accuracy = float(os.environ.get('EXPECTED_ACCURACY', '0.8'))
    
    expected_percent = int(expected_accuracy * 100)
    passed = accuracy >= expected_accuracy
    
    if passed:
        logger.info(f"✅ PASS: Accuracy {accuracy_percent}% meets threshold {expected_percent}%")
    else:
        logger.error(f"❌ FAIL: Accuracy {accuracy_percent}% below threshold {expected_percent}%")
    
    # Write to GitHub environment if available
    write_github_env(success_count, total_count, accuracy, expected_accuracy, passed)
    
    # Exit with appropriate code
    if passed:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error("⚠️  Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
