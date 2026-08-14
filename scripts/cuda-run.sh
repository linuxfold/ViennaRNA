#!/usr/bin/env bash
set -euo pipefail

readonly image="viennarna-cuda-dev:13.1"
readonly source_dir="${VRNA_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly gpu="${VRNA_GPU_DEVICE:-0}"

source "${source_dir}/scripts/cuda-docker-runtime.sh"

docker build -t "${image}" -f "${source_dir}/containers/cuda.Dockerfile" "${source_dir}"
vrna_cuda_docker_prepare "${image}" "${gpu}"

exec docker run --rm \
  "${VRNA_CUDA_DOCKER_GPU_ARGS[@]}" \
  "${VRNA_CUDA_DOCKER_USER_ARGS[@]}" \
  -e HOME=/tmp \
  -v "${source_dir}:/src" \
  -w /src \
  "${image}" \
  bash /src/scripts/cuda-build.sh "$@"
