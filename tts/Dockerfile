# Use a base image with Python, CUDA 11.8, and PyTorch pre-installed
# You might need to find a specific image tag that matches your needs.
# Example using an official nvidia image + manual python/pytorch install:
# Or find a suitable PyTorch image: https://hub.docker.com/r/pytorch/pytorch/
# We need Python 3.10+ for Style-Bert-VITS2 usually.

# Stage 1: Download BERT model using Git LFS (Robust approach)
FROM ubuntu:22.04 AS bert_downloader

# Install git, git-lfs, and ca-certificates
RUN apt-get update && \
    apt-get install -y --no-install-recommends git git-lfs ca-certificates && \
    # Update certificate store
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Initialize git-lfs for the container environment (try without --system)
RUN git lfs install

# Clone the BERT model repository structure
WORKDIR /bert_model
ARG HF_MODEL_NAME="ku-nlp/deberta-v2-large-japanese-char-wwm"
RUN echo "Cloning BERT model repository ${HF_MODEL_NAME}..." && \
    # Clone without downloading LFS files initially (optional, depends on git version)
    # GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/${HF_MODEL_NAME} .
    git clone https://huggingface.co/${HF_MODEL_NAME} . && \
    echo "Repository cloned."

# Fetch LFS objects
RUN echo "Fetching LFS objects..." && \
    git lfs fetch --all && \
    echo "LFS objects fetched."

# Checkout LFS files (replace pointers with actual files)
RUN echo "Checking out LFS files..." && \
    git lfs checkout && \
    echo "LFS files checked out."

# Stage 2: Build the final TTS image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (excluding git/git-lfs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    cmake \
    mecab-ipadic-utf8 \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the pre-downloaded BERT model from the first stage
# Ensure the target directory exists
RUN mkdir -p /app/bert
COPY --from=bert_downloader /bert_model /app/bert/deberta-v2-large-japanese-char-wwm

# Copy the requirements file
COPY requirements.txt /app/

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies using the absolute path
RUN pip install --no-cache-dir -r /app/requirements.txt

# Remove .dockerignore within the build context just in case it exists and causes issues
RUN rm -f /app/.dockerignore

# Copy the rest of the application code
COPY . .

# Set default environment variables
ENV MODEL_DIR=/app/model_assets/anneli
ENV TTS_HOST=0.0.0.0
ENV TTS_PORT=5000

# Expose the port the TTS server will run on
EXPOSE 5000

# Command to run the Style-Bert-VITS2 FastAPI server (without extra args)
CMD ["python", "server_fastapi.py"]
# Note: You might need to adjust the CMD based on how server_fastapi.py expects arguments,
# especially the model path (-m or other flags). Check server_fastapi.py documentation/code.
# Also, ensure config.yml within the tts directory is configured correctly if needed by the server. 