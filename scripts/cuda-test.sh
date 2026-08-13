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
  -e VRNA_CUDA_M2_RING="${VRNA_CUDA_M2_RING:-}" \
  -e VRNA_CUDA_TRACEBACK="${VRNA_CUDA_TRACEBACK:-}" \
  -e VRNA_CUDA_ASYNC_ALLOC="${VRNA_CUDA_ASYNC_ALLOC:-}" \
  -e VRNA_CUDA_PAIR_BITS="${VRNA_CUDA_PAIR_BITS:-}" \
  -e VRNA_CUDA_PRECOMPUTE_HAIRPIN="${VRNA_CUDA_PRECOMPUTE_HAIRPIN:-}" \
  -e VRNA_CUDA_DERIVE_PAIR_TYPES="${VRNA_CUDA_DERIVE_PAIR_TYPES:-}" \
  -e VRNA_CUDA_SKIP_DP_INIT="${VRNA_CUDA_SKIP_DP_INIT:-}" \
  -e VRNA_CUDA_PACKED_DP="${VRNA_CUDA_PACKED_DP:-}" \
  -e VRNA_CUDA_PRECOMPUTE_OUTER_CONTEXT="${VRNA_CUDA_PRECOMPUTE_OUTER_CONTEXT:-}" \
  -e VRNA_CUDA_CANDIDATE_LOWER_BOUND="${VRNA_CUDA_CANDIDATE_LOWER_BOUND:-}" \
  -e VRNA_CUDA_REFINE_CANDIDATE_LOWER_BOUND="${VRNA_CUDA_REFINE_CANDIDATE_LOWER_BOUND:-}" \
  -e VRNA_CUDA_PROFILE_COUNTERS="${VRNA_CUDA_PROFILE_COUNTERS:-}" \
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
    gcc -O2 -I/src/install-cuda/include \
      /src/tests/cuda/test_internal_loop_bounds.c \
      /src/install-cuda/lib/libRNA.a \
      -lm -lgomp -lpthread -lstdc++ -ldl \
      -o /tmp/test_internal_loop_bounds
    /tmp/test_sparse_multibranch
    /tmp/test_internal_loop_bounds
    /tmp/test_mfe_batch "$@"
    VRNA_MFE_BACKEND=cuda \
      VRNA_CUDA_LIBRARY=/src/install-cuda/lib/libRNA_cuda.so \
      /tmp/test_batch_api
  ' -- "$@"
