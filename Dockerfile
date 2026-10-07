# CUDA 12.4 runtime matches torch 2.4+ wheels; host driver 580/CUDA 13.0 is forward-compatible.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY requirements.txt /work/requirements.txt
RUN pip3 install --no-cache-dir -r /work/requirements.txt
COPY . /work

CMD ["bash"]
