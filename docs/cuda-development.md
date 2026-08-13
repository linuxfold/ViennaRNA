# CUDA backend development

> [!IMPORTANT]
> This is an independent experimental fork, not an official ViennaRNA release.
> The CUDA path currently implements a deliberately narrow subset of MFE
> folding and transparently falls back to the authoritative CPU implementation
> outside that subset. See the eligibility envelope below before using it.

The CUDA backend is developed against CUDA 13.1 and emits native code for both
the RTX 4090 (`sm_89`) and RTX PRO 6000 Blackwell (`sm_120`). The CPU backend
remains the authoritative fallback for unsupported model features and for any
device-side compact-energy or candidate-capacity failure.

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
energies use the same `int` decacal/mol representation as the CPU. The `c` and
`fML` matrices are stored on the device as signed 16-bit residuals from a fixed
span-dependent offset to reduce memory traffic. Blackwell and newer devices
use a packed span-major upper triangle; older architectures retain the faster
square indexing measured there. Each finite store is range-checked. If a value
cannot be represented exactly, that input is left unhandled and the public API
recomputes it on the CPU. There is no saturating or approximate path.

The multibranch split uses an exact candidate-sparse recurrence by default.
An interval is recorded only when its paired branch is strictly better than
all split and extension alternatives. Every omitted suffix is therefore
decomposable into a candidate without increasing its energy. Candidates are
stored by right endpoint with a default capacity of 64. Capacity overflow is
reported per input and causes an exact CPU recomputation. The sparse recurrence
can be checked against the original dense recurrence at runtime, and a separate
CPU oracle exhaustively checks short sequences.

Only two spans of the `M2` matrix are retained because paired cells consume
`M2` two spans later. Exact pair bitsets skip illegal internal-loop endpoints,
hairpin size penalties are evaluated once on the CPU and uploaded as a shared
lookup table. Forward kernels explicitly define every upper-triangular DP cell,
avoiding full square `c` and `fML` initialization, and stream-ordered device
allocation reuses CUDA's memory pool. None of these changes alters the energy
recurrences.

Before a finite enclosed `c` cell enters the full thermodynamic table lookup,
the paired kernel adds an exact precomputed lower bound for its `(u1,u2)` loop
shape. The bound minimizes over every pair type and nucleotide context admitted
by the parameter tables. If that optimistic value cannot improve the current
cell minimum, the expensive lookup is skipped. The CPU oracle exhaustively
checks every table context for all 496 legal shapes and seven admitted model
variants.

Energy-only calls copy only the final `f5[n]` values to the host. Structure
calls perform exact traceback on the device and copy only the energy and
dot-bracket result. The traceback follows the CPU decision and tie-breaking
order. If device traceback cannot reproduce a decision, that input is
recomputed by the CPU. Setting `VRNA_CUDA_TRACEBACK=0` retains the diagnostic
path that copies exact matrices and backtracks independent inputs in parallel
on the CPU.

Run exact cell-by-cell, energy, and structure comparisons on the selected GPU:

```sh
VRNA_GPU_DEVICE=0 bash scripts/cuda-test.sh 256 300
```

The test also checks the public batch API, an intentionally unsupported model,
an explicit hard constraint, and forced compact-energy and candidate-capacity
overflows that must take the CPU fallback. A longer validation is:

```sh
VRNA_GPU_DEVICE=0 bash scripts/cuda-test.sh 12 1000
```

## Tuning and benchmarking

`VRNA_CUDA_LANES` controls the multibranch split width. The batch-size
heuristic selects 4 lanes for batches of at least 128 inputs.
`VRNA_CUDA_PAIRED_LANES` can override the paired-loop width independently; by
default it follows `VRNA_CUDA_LANES`. Both accept `1`, `2`, `4`, `8`, `16`, or
`32`. `VRNA_CUDA_BATCH_CHUNK` caps the number of same-length, same-model inputs
in a device chunk. `VRNA_CUDA_PROFILE=1` prints opt-in stage and dispatch
timings.

The exact sparse features are enabled by default. Their diagnostic controls
are:

- `VRNA_CUDA_SPARSE_M2=0`: use the original dense split recurrence;
- `VRNA_CUDA_CANDIDATE_CAPACITY=N`: change the per-column candidate capacity;
- `VRNA_CUDA_VALIDATE_SPARSE_M2=1`: compute sparse and dense splits together
  and fail if any cell differs;
- `VRNA_CUDA_M2_RING=0`: retain the full `M2` matrix;
- `VRNA_CUDA_PAIR_BITS=0`: scan every legal internal-loop coordinate instead
  of using exact pair bitsets;
- `VRNA_CUDA_PRECOMPUTE_HAIRPIN=0`: evaluate long-hairpin logarithms in each
  paired cell instead of using the exact host-precomputed size table;
- `VRNA_CUDA_PRECOMPUTE_OUTER_CONTEXT=0|1`: override register-cached outer
  mismatch terms. The default enables them for GPU-saturating buckets of at
  least 128 inputs;
- `VRNA_CUDA_CANDIDATE_LOWER_BOUND=0`: disable the exact per-shape lower-bound
  test before full internal-loop energy evaluation;
- `VRNA_CUDA_DERIVE_PAIR_TYPES=0|1`: override pair-type storage. The default
  derives pair types from encoded bases on compute capability 12 and newer,
  but retains the dense byte matrix on older devices where lookup is faster;
- `VRNA_CUDA_PACKED_DP=0|1`: override square versus packed span-major `c` and
  `fML` storage. Packed is the default on compute capability 12 and newer;
- `VRNA_CUDA_SKIP_DP_INIT=0`: restore full square `c` and `fML`
  initialization for diagnostics;
- `VRNA_CUDA_ASYNC_ALLOC=0`: use ordinary `cudaMalloc` and `cudaFree`;
- `VRNA_CUDA_TRACEBACK=0`: copy matrices for CPU traceback.

The benchmark accepts mode, count, length, and iteration count:

```sh
VRNA_GPU_DEVICE=0 VRNA_CUDA_LANES=4 \
  bash scripts/cuda-benchmark.sh cuda-energy 256 1000 3
```

For a public FASTA file, select exactly one sequence length so CPU and CUDA
runs receive the same bucket:

```sh
VRNA_GPU_DEVICE=0 \
  bash scripts/cuda-benchmark-fasta.sh cuda-energy data.fasta 256 7 900
VRNA_CPU_THREADS=32 \
  bash scripts/cuda-benchmark-fasta.sh cpu-energy data.fasta 256 7 900
```

The FASTA harness accepts `cpu`, `cuda`, `cpu-energy`, and `cuda-energy`. It
normalizes DNA `T` to RNA `U`, creates fold compounds outside the timed region,
performs a warm-up iteration, and prints energy plus structure checksums.

No machine-specific benchmark results are distributed in this repository.
Users should record the GPU model, driver and CUDA versions, CPU model and
thread count, batch dimensions, mode, warm-up policy, and raw timings when
reporting their own measurements. Compare CPU and CUDA runs over identical,
deterministically generated inputs and verify their reported checksums match.

## Optimization outcome

Candidate sparsification, exact candidate lower bounds, the two-span `M2` ring,
pair bitsets, precomputed
hairpin size penalties, explicit DP-cell writes, architecture-selected pair
types and DP layouts, lane-width specializations, stream-ordered allocation,
and device traceback remain on the development branch because controlled
comparisons improved throughput while preserving exact results. Cooperative
persistent wavefront, alternate lane distribution, batch-SIMD paired-kernel,
exact internal-loop band pruning, and precomputed loop-shape experiments were
implemented and measured but not retained because they did not improve
throughput. Their commits remain on separate rejected-experiment branches so
the results can be reproduced without shipping slower code.

The sparse CPU oracle reports candidate counts and densities, rather than
assuming the proposed recurrence is sparse for a given workload. The benchmark
harness similarly reports raw timings and checksums; it does not claim a fixed
speedup for all sequence distributions, lengths, batch sizes, or devices.

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
