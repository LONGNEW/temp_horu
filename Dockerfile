FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

LABEL org.opencontainers.image.title="HoRU Reproduction Artifact" \
      org.opencontainers.image.description="Reviewer environment for Tables I-III and the six-dataset HoRU/HyperFeel/FedHDC benchmark." \
      org.opencontainers.image.source="https://github.com/LONGNEW/horu-artifact" \
      org.opencontainers.image.licenses="See repository terms"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    git \
    ncompress \
    unzip \
    wget \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /artifact

COPY . /artifact

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
      PyYAML==6.0.2 \
      scipy==1.14.1 \
      pytest==8.3.3 && \
    python -m pip install -e .

RUN chmod +x scripts/*.sh artifact/scripts/*.py

VOLUME ["/artifact/data", "/artifact/results"]

CMD ["bash"]
