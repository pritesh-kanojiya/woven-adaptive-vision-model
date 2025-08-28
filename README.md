# Woven Adaptive Vision Model - MLOps Pipeline

A comprehensive, production-ready MLOps pipeline for training and deploying adaptive vision models with support for multiple datasets, automated workflows, and Docker deployment.

## 🎯 Overview

This repository provides an end-to-end machine learning pipeline designed for scalability, modularity, and ease of use across different teams and models. The pipeline supports both MNIST and custom HuggingFace datasets, automated GitHub Actions workflows, manual approval processes, and containerized deployment.

### Key Features

- **🔄 Automated MLOps Pipeline**: Complete CI/CD with GitHub Actions
- **📊 Multi-Dataset Support**: MNIST and HuggingFace datasets
- **🚀 Containerized Deployment**: Docker and Docker Compose support
- **✅ Manual Approval Process**: Human validation before production deployment
- **📈 Comprehensive Monitoring**: Training metrics, logs, and visualizations
- **🧪 Quick Testing**: Fast validation mode for development
- **🔍 Inference Testing**: Automated API testing workflows
- **📦 Artifact Management**: Model checkpoints, logs, and reports

---

## 🏗️ Architecture

```
├── 📁 .github/workflows/          # Automated CI/CD pipelines
├── 📁 src/                        # Core ML components
├── 📁 scripts/                    # Utility and validation scripts
├── 📁 inference/                  # FastAPI inference server
├── 📁 config/                     # Configuration files
├── 📁 artifacts/                  # Training outputs (auto-generated)
├── 🐳 Dockerfile                  # Container definition
├── 🐳 docker-compose.yml          # Multi-service deployment
└── 📋 main.py                     # Main entry point
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** *(for local testing only)*
- **Git** with GitHub repository access
- **Clean/New Github repository**
- **Access to push docker images to GHCR for github repository** *(by default they have)*
- **Docker** *(for local testing only)*

### Code Setup

1. **Unarchive the repository data**
   ```bash
   unzip woven-adaptive-vision-model.zip
   ```
2. **Create a new github repository and Upload the contents to it**
   ```bash
   rm -rf woven-adaptive-vision-model/.git # Removing .git repository to avoid conflicts with existing one
   cp -avr woven-adaptive-vision-model/. <your-repository-directory>/
   ```

3. **Make sure the .github directory is copied as well**
   ```bash
   ls -lah <your-repository-directory>/.github/workflows/
   ```

4. **Commit the code and push it the main branch**
   ```bash
   git add .
   git commit -m "Woven Adaptive Vision Model Setup"
   git push main
   ```

5. **Now you should be able to see the below github workflows in your Github Console's Actions tab**
   ```
   MLOps Pipeline - Woven Adaptive Vision Models
   Test - Model Inference (MNIST DataSet Model Only)
   ```

---

## 🎮 Usage Guide

## 🔧 GitHub Actions Workflows

### 1. MLOps Pipeline Workflow

**File**: `.github/workflows/woven-mlops-pipeline.yml`

#### Trigger
Navigate to **Actions** → **MLOps Pipeline** → **Run workflow**

#### Input Parameters

| Parameter | Description | Type | Default | Required |
|-----------|-------------|------|---------|----------|
| **Learning Rate** | Training learning rate | String | `0.001` | ✅ |
| **Max Epochs** | Maximum training epochs | String | `5` | ✅ |
| **Required Accuracy** | Minimum accuracy threshold | String | `0.95` | ✅ |
| **Dataset** | Dataset source | String | `mnist` | ✅ |
| **Model Name** | Model and Docker image name | String | `woven-adaptive-mnist-model` | ✅ |
| **Build Docker** | Build and push Docker image | Choice | `true` | ❌ |

#### Workflow Steps

1. **🚀 Initialize Run**
   - Cache dependencies for faster execution
   - Generate model version (format: `vYYYY.MM.DD-{sha}_{run_number}`)
   - Display workflow input summary

2. **🧪 Unit Tests**
   - Validate dataset configuration
   - Run quick integration tests
   - Prepare data loaders

3. **🎯 Training & Testing**
   - Full model training with specified parameters
   - Continuous progress logging
   - Model evaluation and metrics calculation

4. **✅ Manual Validation** *(Conditional: only if accuracy meets threshold)*
   - **Automated Issue Creation**: GitHub issue with performance summary
   - **Human Review Required**: Model metrics, training configuration, artifacts
   - **Approval Options**:
     - ✅ **Approve**: Comment `approve`, `approved`, `lgtm`, or `yes`
     - ❌ **Reject**: Comment `deny`, `denied`, or `no` (with feedback)
   - **Timeout**: 15 minutes (auto-reject if no response)

5. **🐳 Docker Build** *(Conditional: if approved + Docker enabled)*
   - Build optimized CPU-only Docker image
   - Push to GitHub Container Registry (`ghcr.io`)
   - Verify image functionality with health checks

6. **📦 Release & Deployment**
   - Create GitHub release with version tag
   - Upload model artifacts and documentation
   - Include deployment instructions and API testing examples

7. **☁️ AWS ECS Deployment** *(Mock implementation)*
   - Demonstrates production deployment workflow
   - Task definition updates and service deployment
   - Health monitoring and verification

#### Manual Approval Process

When your model meets the accuracy threshold, the workflow automatically creates a **GitHub Issue** for human validation:

1. **📧 Notification**: You'll receive a notification on GitHub Console
2. **📊 Review**: Issue contains:
   - Model performance metrics and logs
   - Training configuration details
   - Artifact links and download instructions
   - Performance analysis checklist
3. **💬 Decision**: Comment on the issue: *(The pipeline proceeds to docker build and deployment only if approved)*
   - **Approve**: `approve`, `approved`, `lgtm`, `yes`
   - **Reject**: `deny`, `denied`, `no`
4. **⏱️ Timeout**: 15 minutes for response (auto-rejects if missed)

### 2. Inference Testing Workflow *(For MNIST model docker image only!)*

**File**: `.github/workflows/inference-tester.yml`

#### Trigger
Navigate to **Actions** → **Test Model Inference** → **Run workflow**

#### Input Parameters

| Parameter | Description | Default | Required |
|-----------|-------------|---------|----------|
| **Image Tag** | Docker image version to test | `latest` | ✅ |
| **Expected Accuracy** | Minimum accuracy threshold | `0.8` | ✅ |

#### What It Does
- Pulls the `woven-adaptive-mnist-model` image from github registry
- Starts inference container
- Runs comprehensive API tests (`/health`, `/predict`)
- Validates prediction accuracy against test dataset
- Reports performance metrics

---

## 📊 Outputs & Artifacts

### Training Artifacts

After successful workflow execution, find the below artifacts in the workflow summary available for download:

```
artifacts/
├── 📁 checkpoints/                 # Saved model files
│   └── {model_name}.pt            # PyTorch model checkpoint
├── 📁 logs/                       # Training logs
│   └── training.log               # Detailed training progress
├── 📁 plots/                      # Visualizations
│   └── training_history.png       # Loss and accuracy curves
├── 📁 reports/                    # Analysis reports
└── metrics_{run_id}.json          # Performance metrics
```

### GitHub Release Artifacts

Each successful training creates a **GitHub Release** containing:

- **📦 Model Checkpoint**: Ready-to-deploy `.pt` file
- **📊 Metrics Report**: JSON with detailed performance data
- **📋 Validation Summary**: Human approval decision details
- **🐳 Docker Instructions**: Complete deployment guide
- **🔧 API Testing Examples**: Ready-to-use curl commands

### Workflow Logs

Access detailed execution logs in **GitHub Actions**:

1. Navigate to **Actions** tab
2. Select your workflow run
3. Expand job steps for detailed logs
4. Download logs using "Download log archive"

---

### Local Environment Setup (Optional)

1. **Change directory to woven-adaptive-vision-model (if not already)**
   ```bash
   cd woven-adaptive-vision-model
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Quick test the setup**
   ```bash
   python main.py quick-test
   ```

### Command Line Interface

The pipeline supports three main commands:

#### 1. **Training** (`train`)
```bash
# Local MNIST training
DATASET_NAME=mnist LEARNING_RATE=0.001 MAX_EPOCHS=5 REQUIRED_ACCURACY=0.95 \
python main.py train

# HuggingFace dataset training
DATASET_NAME=ylecun/mnist LEARNING_RATE=0.001 MAX_EPOCHS=10 REQUIRED_ACCURACY=0.98 \
python main.py train
```

#### 2. **Quick Testing** (`quick-test`)
```bash
python main.py quick-test
```
- Runs 1 epoch with reduced dataset (1000 training, 500 test samples)
- Perfect for development and CI/CD validation
- Automatically cleans up artifacts after completion

#### 3. **Inference Server** (`serve`)
*(Note: The woven-adaptive-mnist-model.pt should exist under artifacts/checkpoints directory for this to work!)*

```bash
MODEL_NAME=woven-adaptive-mnist-model python main.py serve
```
- Starts FastAPI server on `http://localhost:8000`
- Provides `/health` and `/predict` endpoints
- Supports real-time model inference

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATASET_NAME` | Dataset source (`mnist` or HuggingFace path) | - | ✅ |
| `LEARNING_RATE` | Training learning rate | `0.001` | ❌ |
| `MAX_EPOCHS` | Maximum training epochs | `5` | ❌ |
| `REQUIRED_ACCURACY` | Minimum accuracy threshold | `0.95` | ❌ |
| `MODEL_NAME` | Name for model and Docker image | `woven-adaptive-mnist-model` | ❌ |

---

## 🐳 Docker Deployment

### Local Docker Testing (Optional)

1. **Build image locally**
   ```bash
   docker build -t woven-model:local .
   ```

2. **Run inference server**
   ```bash
   docker run -p 8000:8000 woven-model:local
   ```

3. **Test endpoints**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Prediction test
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}'
   ```

### Production Deployment with Docker Compose *(Recommended for local testing)*

1. **Create `docker-compose.yml`**
   ```yaml
   version: '3.8'
   services:
     woven-model:
       image: ghcr.io/{username}/{model-name}:{version}
       ports:
         - "8000:8000"
       environment:
         - MODEL_NAME={model-name}
         - LOG_LEVEL=INFO
       restart: unless-stopped
   ```

2. **Deploy**
   ```bash
   docker-compose up -d
   ```

3. **Monitor**
   ```bash
   docker-compose logs -f
   docker-compose ps
   ```

### Registry Images

After successful workflow completion, Docker images are available at:

```
ghcr.io/{github-username}/{model-name}:{version-tag}
ghcr.io/{github-username}/{model-name}:latest
```

---

## 🔍 API Reference

The inference server provides a RESTful API for model predictions:

### Endpoints

#### **GET** `/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### **POST** `/predict`
Model inference endpoint

**Request Body:**
```json
{
  "data": [/* 784 float values for 28x28 MNIST image */]
}
```

**Response:**
```json
{
  "prediction": 7,
  "confidence": 0.9847,
  "all_probabilities": [0.001, 0.002, ..., 0.9847, ...]
}
```

### Testing Examples

#### Quick Format Check
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}'
```
*Note: Returns validation error showing correct format requirements*

#### Complete Working Example
```bash
python3 -c "
import requests
data = [0.0] * 784
for i in range(350, 450): data[i] = 0.5 + (i % 10) * 0.05
response = requests.post('http://localhost:8000/predict', json={'data': data})
print('Prediction result:', response.json()['prediction'], 'confidence:', round(response.json()['confidence'], 3))
"
```

---

## ⚙️ Configuration

### Model Configuration

Edit `config/config.yaml` to customize:

```yaml
# Model architecture settings
model:
  input_channels: 1      # Input channels (1 for grayscale, 3 for RGB)
  num_classes: 10        # Number of output classes (Auto-detected for HuggingFace datasets)
  type: SimpleCNN        # Model architecture type
  name: woven-adaptive-mnist-model  # Model name
  version: v1.0.1        # Model version
  input_shape: [1, 28, 28]  # Input tensor shape

# Training configuration
training:
  learning_rate: 0.001   # Adam optimizer learning rate
  max_epochs: 5          # Maximum training epochs
  device: cpu            # Training device (cpu/cuda)

# Data loading settings
data:
  batch_size: 64         # Training batch size
  num_workers: 2         # Data loader workers
  data_dir: ./data       # Data storage directory
  dataset_name: mnist    # Default dataset name
  download: true         # Auto-download datasets
  format: pytorch        # Data format
  image_size: [28, 28]   # Expected image dimensions
  normalization:         # MNIST normalization values
    mean: [0.1307]
    std: [0.3081]

# Evaluation criteria
evaluation:
  required_accuracy: 0.95  # Minimum accuracy for approval
  metrics: [accuracy, loss, f1_score]  # Metrics to track

# Logging configuration
logging:
  level: INFO            # Log level (DEBUG, INFO, WARNING, ERROR)
  log_dir: ./artifacts/logs  # Log file directory
  save_logs: true        # Save logs to file

# Artifact management
artifacts:
  save_dir: ./artifacts  # Base directory for outputs
  save_model: true       # Save model checkpoints
  save_plots: true       # Save training plots
  save_reports: true     # Save evaluation reports
```

**Note**: Environment variables (`LEARNING_RATE`, `MAX_EPOCHS`, `REQUIRED_ACCURACY`, `MODEL_NAME`) will override these configuration values during workflow execution.

### Dataset Support

#### MNIST (Built-in)
```bash
DATASET_NAME=mnist python main.py train
```

#### HuggingFace Datasets
```bash
DATASET_NAME=ylecun/mnist python main.py train
DATASET_NAME=pittawat/letter_recognition python main.py train  # Any HF image dataset
```

---

## 📈 Monitoring & Logging

### Training Progress

Real-time training progress includes:

- **Batch-level progress**: Loss values every 100 batches
- **Epoch summaries**: Average loss, accuracy, timing
- **Validation metrics**: Test accuracy, F1-score
- **Convergence tracking**: Training history plots

### Log Files

- **Console Output**: Real-time progress and status
- **training.log**: Detailed file-based logging in `artifacts/logs/`
- **GitHub Actions Logs**: Complete workflow execution history

### Metrics Tracking

Comprehensive metrics saved in JSON format:

```json
{
  "run_id": "mnist_run_20250827_232547",
  "model_name": "woven-adaptive-mnist-model",
  "final_accuracy": 0.9835,
  "final_loss": 0.0490,
  "final_f1": 0.9835,
  "training_time": 21.8,
  "epochs_trained": 1,
  "configuration": {
    "learning_rate": 0.001,
    "max_epochs": 1,
    "dataset": "mnist"
  }
}
```

---

## 🛠️ Development

### Project Structure

```
src/
├── model.py                    # PyTorch model definitions
├── trainer.py                  # Training logic and metrics
├── flexible_data_loader.py     # Multi-format data handling
└── utils.py                    # Utilities and helpers

scripts/
├── check_accuracy.py           # Accuracy validation
├── prepare_and_validate_dataset.py  # Dataset preprocessing
├── test_inference_api.py       # API testing utilities
└── update_config.py            # Configuration management

inference/
├── api.py                      # FastAPI inference server
└── __init__.py                 # Package initialization
```

### Adding New Datasets

1. **HuggingFace Integration**: Use any HF dataset path as `DATASET_NAME`
2. **Custom Formats**: Extend `flexible_data_loader.py`
3. **Configuration**: Update `config.yaml` for new data specifications

---

## 🚨 Troubleshooting

### Common Issues

#### 1. **Workflow Fails at Dataset Loading**
```
❌ No dataset specified. Provide 'mnist' for PyTorch MNIST or HF dataset name
```
**Solution**: Ensure `DATASET_NAME` environment variable is set

#### 2. **Manual Approval Timeout**
```
❌ Manual approval timeout - no response within 15 minutes
```
**Solution**: Respond to GitHub issue faster, or re-run workflow

#### 3. **Docker Build Fails**
```
❌ Model file not found in artifacts
```
**Solution**: Ensure training completed successfully before Docker build

#### 4. **Inference API Errors**
```
{"detail":"Expected 784 values (shape 1x28x28), got 10"}
```
**Solution**: Provide exactly 784 float values for MNIST input