#!/usr/bin/env bash
set -euo pipefail

readonly image="viennarna-cuda-dev:13.1"
readonly source_dir="${VRNA_SOURCE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly gpu="${VRNA_GPU_DEVICE:-0}"

docker build -t "${image}" -f "${source_dir}/containers/cuda.Dockerfile" "${source_dir}"

exec docker run --rm \
  --device="nvidia.com/gpu=${gpu}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "${source_dir}:/src" \
  -w /src \
  "${image}" \
  bash /src/scripts/cuda-build.sh "$@"
