# syntax=docker/dockerfile:1
# MV-AFA SzCORE submission
# Based on the official SzCORE template: https://github.com/esl-epfl/szcore/blob/main/config/template.Dockerfile

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# System dependencies (needed by MNE / scipy / ripser)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        gcc g++ \
        libhdf5-dev \
        git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY algo/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy algorithm package (includes model weights under mvafa_szcore/weights/)
COPY algo/ /app/

# Switch to non-privileged user
USER appuser

VOLUME ["/data"]
VOLUME ["/output"]

ENV INPUT=""
ENV OUTPUT=""

# SzCORE entrypoint: reads /data/${INPUT}, writes /output/${OUTPUT}
CMD python3 -m mvafa_szcore "/data/${INPUT}" "/output/${OUTPUT}"
