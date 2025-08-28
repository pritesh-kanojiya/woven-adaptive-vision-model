FROM python:3.10-slim

# Accept build argument for model name
ARG MODEL_NAME=woven-adaptive-mnist-model

WORKDIR /app

# Create non-root user for security first
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/model /app/logs && \
    chown -R appuser:appuser /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user for package installation
USER appuser

# Install CPU-only PyTorch to avoid CUDA dependencies (force CPU-only)
RUN --mount=type=cache,target=/home/appuser/.cache/pip \
    pip install --user \
    torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2+cpu \
    typing_extensions \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Verify CPU-only installation and check size
RUN python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); assert not torch.cuda.is_available(), 'CUDA should not be available!'" && \
    echo "📦 Installed package sizes:" && \
    du -sh /home/appuser/.local/lib/python*/site-packages/torch* | head -10

# Install minimal dependencies for inference only
RUN --mount=type=cache,target=/home/appuser/.cache/pip \
    pip install --user \
    typing_extensions \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    numpy==1.24.3 \
    pyyaml==6.0.1 \
    pillow==10.0.1

# Copy application code and model
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser artifacts/checkpoints/*.pt ./model/
COPY --chown=appuser:appuser inference/ ./inference/

# Add local python packages to PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app/src
ENV MODEL_NAME=$MODEL_NAME

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
