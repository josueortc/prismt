# PRISMT runtime image: Python and PRISMT's dependencies only.
# The PRISMT code itself comes from your checkout or a job folder's prismt_src/ through
# PYTHONPATH, so the image only needs rebuilding when dependencies change.
#
#   docker build -f docker/Dockerfile -t prismt:cpu  --build-arg VARIANT=cpu  .
#   docker build -f docker/Dockerfile -t prismt:cuda --build-arg VARIANT=cuda .
# Published images: docker.io/josueortc/prismt:cpu and :cuda (built by .github/workflows/docker.yml).
# On a cluster:  apptainer pull prismt.sif docker://docker.io/josueortc/prismt:cuda
#
# On a Mac, prefer the native environment (environment.yml): containers cannot use the Apple GPU.
ARG VARIANT=cpu
FROM python:3.11-slim AS base-cpu
ENV TORCH_INDEX=https://download.pytorch.org/whl/cpu
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base-cuda
RUN apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && rm -rf /var/lib/apt/lists/*
ENV TORCH_INDEX=https://download.pytorch.org/whl/cu121
FROM base-${VARIANT}
LABEL org.opencontainers.image.title="PRISMT" \
      org.opencontainers.image.description="Python runtime for PRISMT (transformers for trial-based recordings); code comes from PYTHONPATH" \
      org.opencontainers.image.source="https://github.com/josueortc/prismt"
RUN python -m pip install --no-cache-dir "torch==2.5.1" --index-url ${TORCH_INDEX} && \
    python -m pip install --no-cache-dir numpy==2.2.5 scipy==1.15.3 h5py==3.16.0 scikit-learn==1.7.2 optuna==5.0.0
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg HDF5_USE_FILE_LOCKING=FALSE PYTHONPATH=/opt/prismt/src
COPY src /opt/prismt/src
WORKDIR /work
ENTRYPOINT ["python", "-m", "prismt"]
CMD ["doctor"]
