FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget \
    coreutils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g typescript

# Install Miniforge - detect architecture and download appropriate version
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh; \
    elif [ "$arch" = "aarch64" ] || [ "$arch" = "arm64" ]; then \
        wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O miniforge.sh; \
    else \
        echo "Unsupported architecture: $arch"; \
        exit 1; \
    fi && \
    bash miniforge.sh -b -p /opt/conda && \
    rm miniforge.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH
RUN conda init bash && \
    echo "conda activate emnlp" >> ~/.bashrc

# Configure conda
RUN conda config --set channel_priority strict \
    && conda config --set solver libmamba

WORKDIR /app

# Copy only the environment file
COPY environment.yml requirements.txt config.json ./
# Copy the config file

# Create conda environment
RUN conda env create -f environment.yml && \
    conda run -n emnlp --no-capture-output python -m pip install --no-cache-dir --progress-bar off -r requirements.txt && \
    echo "conda activate emnlp" >> ~/.bashrc

# Set Python to run unbuffered
ENV PYTHONUNBUFFERED=1
# ENV PYTHONIOENCODING=utf-8
# ENV PYTHONFAULTHANDLER=1

# Ensure we have write permissions in /app
RUN chmod 777 /app

# Run the Python script directly in the conda environment
ENTRYPOINT ["conda", "run", "-n", "emnlp", "--no-capture-output", "python"]
CMD ["agent_manager.py"]
