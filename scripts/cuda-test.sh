#!/usr/bin/env bash
set -euo pipefail

readonly source_dir="${VRNA_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly image="viennarna-cuda-dev:13.1"
readonly gpu="${VRNA_GPU_DEVICE:-0}"

exec docker run --rm \
  --device="nvidia.com/gpu=${gpu}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e VRNA_CUDA_SPARSE_M2="${VRNA_CUDA_SPARSE_M2:-}" \
  -e VRNA_CUDA_CANDIDATE_CAPACITY="${VRNA_CUDA_CANDIDATE_CAPACITY:-}" \
  -e VRNA_CUDA_VALIDATE_SPARSE_M2="${VRNA_CUDA_VALIDATE_SPARSE_M2:-}" \
  -v "${source_dir}:/src" \
  -w /src \
  "${image}" \
  bash -lc '
    set -euo pipefail
    gcc -O2 -I/src/src -I/src/install-cuda/include \
      /src/tests/cuda/test_mfe_batch.c \
      -L/src/install-cuda/lib -Wl,-rpath,/src/install-cuda/lib \
      -lRNA_cuda /src/install-cuda/lib/libRNA.a \
      -lm -lgomp -lpthread -lstdc++ -ldl \
      -o /tmp/test_mfe_batch
    gcc -O2 -I/src/install-cuda/include \
      /src/tests/cuda/test_batch_api.c \
      /src/install-cuda/lib/libRNA.a \
      -lm -lgomp -lpthread -lstdc++ -ldl \
      -o /tmp/test_batch_api
    gcc -O2 -I/src/install-cuda/include \
      /src/tests/cuda/test_sparse_multibranch.c \
      /src/install-cuda/lib/libRNA.a \
      -lm -lgomp -lpthread -lstdc++ -ldl \
      -o /tmp/test_sparse_multibranch
    /tmp/test_sparse_multibranch
    /tmp/test_mfe_batch "$@"
    VRNA_MFE_BACKEND=cuda \
      VRNA_CUDA_LIBRARY=/src/install-cuda/lib/libRNA_cuda.so \
      /tmp/test_batch_api
  ' -- "$@"
