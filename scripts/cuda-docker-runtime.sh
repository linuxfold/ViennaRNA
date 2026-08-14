#!/usr/bin/env bash

# Shared Docker GPU selection for the CUDA build and test helpers.
# This file is sourced by other scripts and intentionally does not enable
# shell options on its own.

vrna_cuda_docker_probe_gpu() {
  local vrna_docker_image="$1"
  local vrna_gpu_index="$2"
  local vrna_gpu_mode="$3"
  local -a vrna_candidate

  case "${vrna_gpu_mode}" in
    cdi)
      vrna_candidate=(--device="nvidia.com/gpu=${vrna_gpu_index}")
      ;;
    gpus)
      vrna_candidate=(--gpus "device=${vrna_gpu_index}")
      ;;
    *)
      printf 'Internal error: unsupported Docker GPU mode: %s\n' "${vrna_gpu_mode}" >&2
      return 2
      ;;
  esac

  if docker run --rm "${vrna_candidate[@]}" "${vrna_docker_image}" nvidia-smi -L >/dev/null 2>&1; then
    VRNA_CUDA_DOCKER_GPU_ARGS=("${vrna_candidate[@]}")
    VRNA_CUDA_DOCKER_GPU_MODE="${vrna_gpu_mode}"
    return 0
  fi

  return 1
}

vrna_cuda_docker_prepare() {
  local vrna_docker_image="$1"
  local vrna_gpu_index="$2"
  local vrna_requested_mode="${VRNA_DOCKER_GPU_MODE:-auto}"
  local vrna_security_options

  VRNA_CUDA_DOCKER_GPU_ARGS=()
  VRNA_CUDA_DOCKER_USER_ARGS=()
  VRNA_CUDA_DOCKER_GPU_MODE=""

  case "${vrna_requested_mode}" in
    auto|cdi|gpus)
      ;;
    *)
      printf 'Invalid VRNA_DOCKER_GPU_MODE=%s; expected auto, cdi, or gpus.\n' \
        "${vrna_requested_mode}" >&2
      return 2
      ;;
  esac

  if ! vrna_security_options="$(docker info --format '{{json .SecurityOptions}}' 2>/dev/null)"; then
    printf 'Unable to query the Docker daemon. Is the selected Docker context running?\n' >&2
    return 1
  fi

  if ! docker image inspect "${vrna_docker_image}" >/dev/null 2>&1; then
    printf 'Docker image %s is unavailable. Run scripts/cuda-run.sh first.\n' \
      "${vrna_docker_image}" >&2
    return 1
  fi

  if [[ "${vrna_security_options}" != *rootless* ]]; then
    VRNA_CUDA_DOCKER_USER_ARGS=(--user "$(id -u):$(id -g)")
  fi

  if [[ "${vrna_requested_mode}" == "auto" ]]; then
    if vrna_cuda_docker_probe_gpu "${vrna_docker_image}" "${vrna_gpu_index}" cdi; then
      return 0
    fi

    if vrna_cuda_docker_probe_gpu "${vrna_docker_image}" "${vrna_gpu_index}" gpus; then
      printf 'CUDA Docker: CDI device selection is unavailable; using --gpus for device %s.\n' \
        "${vrna_gpu_index}" >&2
      return 0
    fi
  elif vrna_cuda_docker_probe_gpu "${vrna_docker_image}" "${vrna_gpu_index}" "${vrna_requested_mode}"; then
    return 0
  fi

  printf 'Unable to expose GPU %s to Docker image %s using mode %s.\n' \
    "${vrna_gpu_index}" "${vrna_docker_image}" "${vrna_requested_mode}" >&2
  printf 'Check the NVIDIA Container Toolkit setup or set VRNA_DOCKER_GPU_MODE=cdi|gpus.\n' >&2
  return 1
}
