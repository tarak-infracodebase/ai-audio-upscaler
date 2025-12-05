# Multi-stage build for AI Audio Upscaler Pro - Production Ready
# Stage 1: Build dependencies and compile requirements
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime AS builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETPLATFORM=linux/amd64

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for cleaner dependency management
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
COPY requirements-prod.txt /tmp/requirements-prod.txt

# Install Python packages with optimizations
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements-prod.txt

# Pre-compile Python bytecode
RUN python -c "import compileall; compileall.compile_dir('/opt/venv/lib/python3.10/site-packages', force=True)"

# Stage 2: Runtime image - Optimized for production
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime AS runtime

# Metadata
LABEL maintainer="AI Audio Upscaler Team <dev@aiupscaler.com>"
LABEL description="Production-ready AI Audio Upscaler SaaS"
LABEL version="2.0.0"

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    libportaudio2 \
    # System utilities
    curl \
    ca-certificates \
    gosu \
    tini \
    # Monitoring and debugging
    htop \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

# Create application directories with proper permissions
RUN mkdir -p /app /app/data /app/logs /app/models /tmp/audio \
    && chown -R appuser:appuser /app /tmp/audio

# Copy application code
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Install the application in development mode for easier updates
RUN pip install -e .

# Create entrypoint script for flexible startup
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8001 8002

# Set default command with proper signal handling
ENTRYPOINT ["/usr/sbin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["api"]