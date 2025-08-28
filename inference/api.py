"""
FastAPI Inference API for MNIST Model
Simple and clean API for model serving (now metadata-aware for modularity)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np
import os
import sys
from typing import List
import logging

# Add src to path for model imports
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and runtime metadata
model = None
device = None
metadata = {
    'input_shape': [1, 28, 28],
    'normalization': {'mean': [0.1307], 'std': [0.3081]},
    'num_classes': 10,
    'model_type': 'SimpleCNN'
}

# Initialize FastAPI app
app = FastAPI(
    title="Woven Adaptive Vision Model API",
    description="Adaptive vision classification API supporting multiple datasets",
    version="1.0.0"
)


def load_model():
    """Load the trained model and its metadata if available."""
    global model, device, metadata
    try:
        from model import SimpleCNN

        device = torch.device('cpu')  # Use CPU for serving

        # Get model name from environment variable, default to original name
        model_name = os.environ.get('MODEL_NAME', 'woven-adaptive-mnist-model')

        # Load model weights first to get metadata
        model_path = f'model/{model_name}.pt'
        fallback_path = f'artifacts/checkpoints/{model_name}.pt'
        checkpoint = None

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            logger.info(f"✅ Model loaded from {model_path}")
        elif os.path.exists(fallback_path):
            checkpoint = torch.load(fallback_path, map_location=device)
            logger.info(f"✅ Model loaded from {fallback_path}")
        else:
            raise FileNotFoundError("No model file found")

        # Load metadata if present to get model configuration
        if isinstance(checkpoint, dict) and 'metadata' in checkpoint and isinstance(checkpoint['metadata'], dict):
            metadata.update(checkpoint['metadata'])
            logger.info(f"✅ Loaded model metadata: {metadata}")

        # Create model with correct number of classes from metadata
        input_channels = metadata.get('input_shape', [1, 28, 28])[0]
        num_classes = metadata.get('num_classes', 10)
        
        logger.info(f"🏗️  Creating model with {input_channels} input channels and {num_classes} output classes")
        model = SimpleCNN(input_channels=input_channels, num_classes=num_classes)

        # Apply state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Allow direct state-dict checkpoints as well
            model.load_state_dict(checkpoint)

        # Load metadata if present
        if isinstance(checkpoint, dict) and 'metadata' in checkpoint and isinstance(checkpoint['metadata'], dict):
            metadata.update(checkpoint['metadata'])

        model.eval()
        logger.info("🚀 Model ready for inference")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    load_model()


class PredictionInput(BaseModel):
    """Input data for prediction"""
    data: List[float] = Field(..., description="Flattened image data (C*H*W values per model)")

    class Config:
        schema_extra = {
            "example": {
                "data": [0.0] * 784  # Default MNIST example; actual size is dynamic
            }
        }


class PredictionOutput(BaseModel):
    """Prediction output"""
    prediction: int = Field(..., description="Predicted class index")
    confidence: float = Field(..., description="Confidence score (0-1)")
    all_probabilities: List[float] = Field(..., description="Probabilities for all classes")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Predict class from input tensor serialized as a flat list."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Validate input size using metadata
        c, h, w = metadata.get('input_shape', [1, 28, 28])
        expected = int(c) * int(h) * int(w)
        if len(input_data.data) != expected:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected} values (shape {c}x{h}x{w}), got {len(input_data.data)}"
            )

        # Convert to tensor and reshape to NCHW
        data = np.array(input_data.data, dtype=np.float32).reshape(1, c, h, w)
        tensor = torch.from_numpy(data).to(device)

        # Apply normalization if dimensions align
        norm = metadata.get('normalization', {'mean': [0.1307], 'std': [0.3081]})
        mean = torch.tensor(norm.get('mean', [0.1307]), dtype=torch.float32).view(1, -1, 1, 1).to(device)
        std = torch.tensor(norm.get('std', [0.3081]), dtype=torch.float32).view(1, -1, 1, 1).to(device)
        if tensor.shape[1] == mean.shape[1]:
            tensor = (tensor - mean) / std

        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0][predicted_class].item())
            all_probs = [float(x) for x in probabilities[0].tolist()]

        return PredictionOutput(
            prediction=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "model_type": metadata.get('model_type', 'SimpleCNN'),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_shape": metadata.get('input_shape', [1, 28, 28]),
            "output_classes": metadata.get('num_classes', 10),
            "device": str(device),
            "normalization": metadata.get('normalization', {'mean': [0.1307], 'std': [0.3081]})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Woven Adaptive Vision Model API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "available_endpoints": ["/", "/predict", "/health", "/model/info", "/docs"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
