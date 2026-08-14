# CUDA backend development

This repository contains experimental CUDA acceleration for batched ViennaRNA
MFE, partition-function, and dense base-pair-probability calculations. The
CUDA backend is optional: the standard implementation remains available and
unsupported inputs fall back automatically.

## Build

The helper scripts build the CUDA development image and compile the optional
shared backend:

```sh
bash scripts/cuda-build.sh
VRNA_GPU_DEVICE=0 bash scripts/cuda-run.sh
```

The default helper configuration selects one GPU. Override the device index
only when a different selected device is intended.

## Runtime selection

The batch APIs dispatch eligible inputs to CUDA and preserve the existing path
for unsupported models, constraints, or unavailable acceleration.

Relevant environment variables:

| Variable | Values | Default | Purpose |
| --- | --- | --- | --- |
| `VRNA_PF_BACKEND` | `cuda`, `cpu` | automatic | Select the PF batch backend. |
| `VRNA_CUDA_PF_PRECISION` | `fp64`, `fp32`, `auto` | `fp64` | Select recurrence precision. |
| `VRNA_CUDA_PF_REFERENCE_DAG` | `0`, `1` | `0` | Enable the retained full CUDA DAG for comparison. |
| `VRNA_CUDA_PF_TRANSIENT_PLAN` | `0`, `1` | `0` | Disable persistent allocations and graph replay for diagnostics. |
| `VRNA_GPU_DEVICE` | device index | `0` in helpers | Select the GPU exposed to a helper script. |

Strict FP64 is the supported exactness-oriented default. FP32 and automatic
precision selection are experimental. In automatic mode, numerical health
checks cause an unhealthy FP32 batch to be recomputed in FP64.

## Reduced PF/BPP recurrence

The default PF implementation stores full triangular B, S, and M matrices.
Unpaired U and two-component multiloop M2 values use two-span rolling buffers.
Exterior q5 and q3 vectors replace the former full Q and E matrices.

The reverse pass uses gather kernels for dM, dS, and dB. Each output cell has a
single writer, so the recurrence-critical reverse pass does not use atomic
updates. The dense-BPP path forms probabilities, validates paired sums, and
writes ViennaRNA's row-wise probability layout on the GPU.

A persistent plan caches device allocations, pinned host staging, and a
captured CUDA graph for repeated calls with the same sequence length and batch
size. Set `VRNA_CUDA_PF_TRANSIENT_PLAN=1` to retain the allocation-per-call
diagnostic path.

## Single-GPU MFE benchmark

The earlier exact MFE benchmark measured one RTX PRO 6000 with a batch of 256
public 900-nt EternaFold sequences. The reference used 32 threads on an AMD
Ryzen Threadripper PRO 9955WX. Values are mean wall times across seven timed
iterations.

| Backend | Energy only | Structures |
| --- | ---: | ---: |
| 32-thread AMD Ryzen Threadripper PRO 9955WX | 5.313528 s | 5.305571 s |
| RTX PRO 6000 | 0.107429 s | 0.162433 s |
| GPU speedup | 49.46× | 32.66× |

Energy and structure checksums matched exactly. These measurements describe
only the named CPU and GPU types, thread count, and workload.

## Single-GPU PF/BPP benchmark

The strict-FP64 benchmark used one RTX PRO 6000, 256 sequences of length 900,
an untimed warm-up, and one timed iteration.

| CUDA implementation | PF only | Dense BPP |
| --- | ---: | ---: |
| Full reference DAG | 1.331000 s | 5.164998 s |
| Reduced persistent recurrence | 0.932028 s | 2.804007 s |
| Runtime reduction | 30.0% | 45.7% |

Energy and BPP checksums matched between the two CUDA implementations. These
numbers describe this GPU type and workload only; they are not a claim about
other hardware or inputs.

Run the same harness with a FASTA input:

```sh
VRNA_GPU_DEVICE=0 bash scripts/cuda-benchmark-pf-fasta.sh cuda-pf INPUT.fasta 256 1 900
VRNA_GPU_DEVICE=0 bash scripts/cuda-benchmark-pf-fasta.sh cuda-bpp INPUT.fasta 256 1 900
```

Use `VRNA_CUDA_PF_REFERENCE_DAG=1` for the reference-DAG comparison.

## Validation

Run the CUDA validation suite with:

```sh
VRNA_GPU_DEVICE=0 bash scripts/cuda-test.sh 32 80
```

The suite checks the CPU adjoint identity, exact PF/BPP agreement, randomized
sequence coverage, model variants, supported hard constraints, and fallback
behavior. The current CUDA comparison covers 1,327 cells with maximum energy
error `9.16e-07` and maximum probability error `7.77e-16`.

## Limitations

- The CUDA PF/BPP batch path currently supports eligible single-strand fold
  compounds and dense probabilities.
- Unsupported soft constraints, callbacks, comparative inputs, and other
  unimplemented model features use the established fallback.
- Strict FP64 is the default. Experimental FP32 modes do not replace the exact
  validation requirement.
- The additional 10× performance target has not been reached.

## Development disclosure

The CUDA work was developed with AI-assisted implementation and review.
Correctness is established by the repository's executable comparison tests,
not by the development method.
