# CUDA backend development

> [!IMPORTANT]
> This is an independent experimental fork, not an official ViennaRNA release.
> The CUDA path currently implements a deliberately narrow subset of MFE
> folding and transparently falls back to the authoritative CPU implementation
> outside that subset. See the eligibility envelope below before using it.

The CUDA backend is developed against CUDA 13.1 and emits native code for both
the RTX 4090 (`sm_89`) and RTX PRO 6000 Blackwell (`sm_120`). The CPU backend
remains the authoritative fallback for unsupported model features and for any
device-side compact-energy range failure.

Build and install from the repository root without requiring root access:

```sh
bash scripts/cuda-run.sh
```

Select exactly one device by its `nvidia-smi` index:

```sh
VRNA_GPU_DEVICE=0 bash scripts/cuda-run.sh
```

The user-local install is stored in `install-cuda/`. Generated build products
are covered by ViennaRNA's existing ignore rules.

The build installs the optional backend as
`install-cuda/lib/libRNA_cuda.so`. The ordinary RNAlib remains CPU-only and
loads this plugin at runtime only when `vrna_mfe_batch()` is called. Select the
backend with `VRNA_MFE_BACKEND=auto|cpu|cuda`; an absolute plugin path may be
provided through `VRNA_CUDA_LIBRARY`.

The initial exact CUDA eligibility envelope is deliberately narrow:

- single, linear RNA strands using the dangles=2 MFE model;
- standard thermodynamic parameter tables with default salt;
- static default hard constraints and no soft constraints or callbacks;
- no G-quadruplex, circular, comparative, multistrand, auxiliary-grammar, or
  unstructured-domain recurrences.

Any input outside that envelope is folded by the unmodified CPU path. CUDA
energies use the same `int` decacal/mol representation as the CPU. The dense
`c` and `fML` matrices are stored on the device as signed 16-bit residuals from
a fixed span-dependent offset to reduce memory traffic. Each finite store is
range-checked. If a value cannot be represented exactly, that input is left
unhandled and the public API recomputes it on the CPU. There is no saturating
or approximate path. When structures are requested, the matrices are expanded
back to the original integers and consumed by the existing CPU backtracker.

Energy-only calls copy only the final `f5[n]` values to the host. Structure
calls copy the exact `c`, `fML`, and `f5` matrices and backtrack on the CPU.

Run exact cell-by-cell, energy, and structure comparisons on the selected GPU:

```sh
VRNA_GPU_DEVICE=0 bash scripts/cuda-test.sh 256 300
```

The test also checks the public batch API, an intentionally unsupported model,
and an explicit hard constraint that must take the CPU fallback. A longer
compact-range validation is:

```sh
VRNA_GPU_DEVICE=0 bash scripts/cuda-test.sh 12 1000
```

## Tuning and benchmarking

`VRNA_CUDA_LANES` controls the multibranch split width. The batch-size
heuristic selects 4 lanes for batches of at least 128 inputs, which was optimal
for the 256 by 1000 benchmark on the RTX 4090. `VRNA_CUDA_PAIRED_LANES` can
override the paired-loop width independently; by default it follows
`VRNA_CUDA_LANES`. `VRNA_CUDA_BATCH_CHUNK` caps the number of same-length,
same-model inputs in a device chunk. `VRNA_CUDA_PROFILE=1` prints opt-in stage
and dispatch timings.

The benchmark accepts mode, count, length, and iteration count:

```sh
VRNA_GPU_DEVICE=0 VRNA_CUDA_LANES=4 \
  bash scripts/cuda-benchmark.sh cuda-energy 256 1000 3
```

No machine-specific benchmark results are distributed in this repository.
Users should record the GPU model, driver and CUDA versions, CPU model and
thread count, batch dimensions, mode, warm-up policy, and raw timings when
reporting their own measurements. Compare CPU and CUDA runs over identical,
deterministically generated inputs and verify their reported checksums match.

## Prior work and development disclosure

ViennaRNA previously published the experimental `ViennaRNA-2.3.0cuda` package,
and Langdon and Lorenz described that work in *CUDA RNAfold* (2018),
<https://doi.org/10.1101/298885>. This fork is not the first GPU implementation
of RNAfold. Its goal is an exact, optional CUDA backend for a current ViennaRNA
codebase, with explicit CPU fallback and support for contemporary NVIDIA GPU
architectures.

The CUDA backend, tests, build tooling, optimization experiments, and this
development report were primarily implemented, optimized, tested, and
documented by OpenAI GPT-5.6 Sol at Extra High reasoning effort using paid
access. The human repository owner supplied the goal, compute, constraints,
and oversight and remains responsible for reviewing the code and any reported
results. This AI assistance should be disclosed in any contribution or
publication derived from this branch.
