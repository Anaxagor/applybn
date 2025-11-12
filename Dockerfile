# syntax=docker/dockerfile:1.7

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG POETRY_VERSION=1.8.3

# System setup and dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        git \
        build-essential \
        locales \
        # Java runtime for JPype/JVM usage \
        openjdk-21-jdk \
        # OpenCV runtime deps
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        # OpenMP runtime for numpy/scikit/shap
        libgomp1 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    LD_LIBRARY_PATH=/usr/lib/jvm/java-17-openjdk-amd64/lib/server:${LD_LIBRARY_PATH}

# Install Python 3.11 on Ubuntu 24.04 (default Python is 3.12). Use deadsnakes PPA for 3.11 packages.
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH=/opt/poetry/bin:${VIRTUAL_ENV}/bin:${PATH}

RUN curl -sSL https://install.python-poetry.org | python - --version ${POETRY_VERSION}

WORKDIR /app

# Copy only dependency metadata first for layer caching
COPY pyproject.toml /app/

# Pre-install dependencies (no project code yet) for better caching
RUN poetry env use /usr/bin/python3.11 \
    && poetry install --no-root --no-interaction --no-ansi

# Now copy the rest of the source
COPY . /app

# Install the project itself
RUN poetry install --no-interaction --no-ansi

# Default command (override as needed)
CMD ["bash"]


