#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <thread>
#include <vector>

extern "C" {
#include "ViennaRNA/constraints/hard.h"
#include "ViennaRNA/datastructures/dp_matrices.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/gpu/backend.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/params/basic.h"
#include "ViennaRNA/params/constants.h"
}


namespace {

constexpr int kInf       = INF;
constexpr int kMaxNinio  = 300;
constexpr int kBlockSize = 256;
constexpr int kExteriorBatchTile = 8;
constexpr int kCompactMSpanSlope = -32;
constexpr unsigned int kDefaultCandidateCapacity = 64;

enum ProfileCounter : unsigned int {
  kProfileOuterCells = 0,
  kProfilePairableOuterCells,
  kProfilePotentialInternalCoordinates,
  kProfilePairBitAcceptedCoordinates,
  kProfileFiniteEnclosedCells,
  kProfileInternalEnergyEvaluations,
  kProfileWinnerHairpin,
  kProfileWinnerStack,
  kProfileWinnerBulge,
  kProfileWinnerInternal,
  kProfileWinnerMultibranch,
  kProfileCandidateComparisons,
  kProfileRightExtensionWins,
  kProfilePairedBranchWins,
  kProfileCandidateCapacityFallbacks,
  kProfileCandidateTotal,
  kProfileCandidateMaximum,
  kProfileWinnerUnpairedBase,
  kProfileCounterCount = kProfileWinnerUnpairedBase + MAXLOOP + 1
};

enum PairedWinner : unsigned int {
  kWinnerHairpin = 0,
  kWinnerStack,
  kWinnerBulge,
  kWinnerInternal,
  kWinnerMultibranch
};

enum TraceKind : unsigned char {
  kTraceF5 = 0,
  kTraceC  = 1,
  kTraceM  = 2
};

enum LoopShapeKind : unsigned char {
  kLoopShapeStack = 0,
  kLoopShapeBulge,
  kLoopShapeInt11,
  kLoopShapeInt21Forward,
  kLoopShapeInt21Reverse,
  kLoopShapeInt22,
  kLoopShapeGeneric1n,
  kLoopShapeInt23,
  kLoopShapeGeneric,
  kLoopShapeInvalid
};

struct LoopShape {
  int           base;
  unsigned char kind;
  unsigned char padding[3];
};

struct TraceSector {
  unsigned short i;
  unsigned short j;
  unsigned char  kind;
  unsigned char  padding[3];
};


template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() : data_(nullptr), asynchronous_(false) {}

  ~DeviceBuffer()
  {
    if (data_) {
      if (asynchronous_)
        cudaFreeAsync(data_, nullptr);
      else
        cudaFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer &)             = delete;
  DeviceBuffer &operator=(const DeviceBuffer &)  = delete;

  bool allocate(size_t count,
                bool   asynchronous = false)
  {
    asynchronous_ = asynchronous;
    return (count == 0) ||
           ((asynchronous ? cudaMallocAsync(reinterpret_cast<void **>(&data_),
                                             sizeof(T) * count,
                                             nullptr) :
                            cudaMalloc(reinterpret_cast<void **>(&data_),
                                       sizeof(T) * count)) == cudaSuccess);
  }

  T *get() const
  {
    return data_;
  }

private:
  T    *data_;
  bool asynchronous_;
};


template <typename T>
T *
arena_take(unsigned char *arena,
           size_t        &offset,
           size_t        count)
{
  const size_t alignment = alignof(T);
  offset = (offset + alignment - 1) & ~(alignment - 1);
  T *result = reinterpret_cast<T *>(arena + offset);
  offset += sizeof(T) * count;
  return result;
}


class EventTimeline {
public:
  EventTimeline() : used_(0) {}

  ~EventTimeline()
  {
    for (cudaEvent_t event : events_)
      if (event)
        cudaEventDestroy(event);
  }

  EventTimeline(const EventTimeline &)             = delete;
  EventTimeline &operator=(const EventTimeline &)  = delete;

  bool initialize(size_t count)
  {
    events_.resize(count, nullptr);
    for (cudaEvent_t &event : events_)
      if (cudaEventCreate(&event) != cudaSuccess)
        return false;

    return true;
  }

  bool record()
  {
    if (used_ >= events_.size())
      return false;

    return cudaEventRecord(events_[used_++]) == cudaSuccess;
  }

  bool synchronize() const
  {
    return (used_ > 0) && (cudaEventSynchronize(events_[used_ - 1]) == cudaSuccess);
  }

  float interval(size_t begin,
                 size_t end) const
  {
    float milliseconds = 0.f;
    if ((begin >= used_) ||
        (end >= used_) ||
        (cudaEventElapsedTime(&milliseconds, events_[begin], events_[end]) != cudaSuccess))
      return 0.f;

    return milliseconds;
  }

private:
  std::vector<cudaEvent_t> events_;
  size_t                   used_;
};


__host__ __device__ inline unsigned int
dense_index(unsigned int i,
            unsigned int j,
            unsigned int batch,
            unsigned int n,
            unsigned int batch_size)
{
  return (i * (n + 1) + j) * batch_size + batch;
}


__host__ __device__ inline size_t
candidate_index(unsigned int j,
                unsigned int entry,
                unsigned int batch,
                unsigned int capacity,
                unsigned int batch_size)
{
  return (static_cast<size_t>(j) * capacity + entry) * batch_size + batch;
}


__host__ __device__ inline size_t
m2_ring_index(unsigned int span,
              unsigned int i,
              unsigned int batch,
              unsigned int n,
              unsigned int batch_size)
{
  return (static_cast<size_t>(span & 1U) * (n + 1) + i) * batch_size + batch;
}


__host__ __device__ inline size_t
pair_bit_index(unsigned int p,
               unsigned int word,
               unsigned int batch,
               unsigned int words,
               unsigned int batch_size)
{
  return (static_cast<size_t>(p) * words + word) * batch_size + batch;
}


__device__ __forceinline__ unsigned int
pair_type_at(const unsigned char *pair_types,
             const short         *sequence2,
             unsigned int        i,
             unsigned int        j,
             unsigned int        batch,
             unsigned int        n,
             unsigned int        batch_size,
             const vrna_param_t  *params)
{
  if (pair_types)
    return pair_types[dense_index(i, j, batch, n, batch_size)];

  if ((i == 0) ||
      (j <= i) ||
      ((j - i) >= static_cast<unsigned int>(params->model_details.max_bp_span)) ||
      ((j - i) <= static_cast<unsigned int>(params->model_details.min_loop_size)))
    return 0;

  const size_t pitch = n + 2;
  const short *s2 = sequence2 + static_cast<size_t>(batch) * pitch;
  unsigned int type = params->model_details.pair[s2[i]][s2[j]];
  if (params->model_details.noGU && ((type == 3) || (type == 4)))
    type = 0;

  return type;
}


__device__ __forceinline__ int
minimum(int a,
        int b)
{
  return (a < b) ? a : b;
}


__device__ __forceinline__ int
add_minimum(int a,
            int b,
            int current)
{
  if ((a >= kInf) || (b >= kInf))
    return current;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  return __viaddmin_s32(a, b, current);
#else
  return minimum(a + b, current);
#endif
}


__device__ __forceinline__ void
profile_add(unsigned long long *counters,
            unsigned int       counter,
            unsigned long long value = 1)
{
  if (counters && value)
    atomicAdd(counters + counter, value);
}


__device__ __forceinline__ int
load_compact_m(const short *m,
               unsigned int index,
               unsigned int span)
{
  const short value = m[index];
  return (value == SHRT_MAX) ? kInf :
         static_cast<int>(value) + static_cast<int>(span) * kCompactMSpanSlope;
}


__device__ __forceinline__ void
store_compact_m(short         *m,
                unsigned int  index,
                unsigned int  span,
                unsigned int  batch,
                int           value,
                unsigned int  *overflow)
{
  if (value >= kInf) {
    m[index] = SHRT_MAX;
    return;
  }

  const int residual = value - static_cast<int>(span) * kCompactMSpanSlope;
  if ((residual < SHRT_MIN) || (residual >= SHRT_MAX)) {
    atomicExch(overflow + batch, 1U);
    m[index] = SHRT_MAX;
  } else {
    m[index] = static_cast<short>(residual);
  }
}


__device__ __forceinline__ int
multibranch_stem_energy(unsigned int       type,
                        int                si1,
                        int                sj1,
                        const vrna_param_t *params)
{
  int energy = params->MLintern[type];

  if ((si1 >= 0) && (sj1 >= 0))
    energy += params->mismatchM[type][si1][sj1];
  else if (si1 >= 0)
    energy += params->dangle5[type][si1];
  else if (sj1 >= 0)
    energy += params->dangle3[type][sj1];

  if (type > 2)
    energy += params->TerminalAU;

  return energy;
}


__device__ __forceinline__ int
exterior_stem_energy(unsigned int       type,
                     int                si1,
                     int                sj1,
                     const vrna_param_t *params)
{
  int energy = 0;

  if ((si1 >= 0) && (sj1 >= 0))
    energy += params->mismatchExt[type][si1][sj1];
  else if (si1 >= 0)
    energy += params->dangle5[type][si1];
  else if (sj1 >= 0)
    energy += params->dangle3[type][sj1];

  if (type > 2)
    energy += params->TerminalAU;

  return energy;
}


__device__ bool
loop_matches(const char *sequence,
             const char *table,
             unsigned int sequence_length,
             unsigned int table_stride)
{
  for (unsigned int k = 0; k < sequence_length; k++)
    if (sequence[k] != table[k])
      return false;

  return table[sequence_length] == '\0' ||
         table[sequence_length] == ' ' ||
         table_stride == sequence_length;
}


__device__ int
special_hairpin_energy(const char         *sequence,
                       unsigned int       size,
                       const vrna_param_t *params)
{
  const char  *table;
  const int   *energies;
  unsigned int stride;
  unsigned int sequence_length;
  unsigned int table_length;

  if (size == 3) {
    table           = params->Triloops;
    energies        = params->Triloop_E;
    stride          = 6;
    sequence_length = 5;
    table_length    = sizeof(params->Triloops);
  } else if (size == 4) {
    table           = params->Tetraloops;
    energies        = params->Tetraloop_E;
    stride          = 7;
    sequence_length = 6;
    table_length    = sizeof(params->Tetraloops);
  } else if (size == 6) {
    table           = params->Hexaloops;
    energies        = params->Hexaloop_E;
    stride          = 9;
    sequence_length = 8;
    table_length    = sizeof(params->Hexaloops);
  } else {
    return kInf;
  }

  for (unsigned int offset = 0;
       (offset + sequence_length) < table_length && table[offset] != '\0';
       offset += stride)
    if (loop_matches(sequence, table + offset, sequence_length, stride))
      return energies[offset / stride];

  return kInf;
}


__device__ int
hairpin_energy(const short        *sequence,
               const char         *sequence_chars,
               const int          *hairpin_size_energies,
               unsigned int       i,
               unsigned int       j,
               unsigned int       batch,
               unsigned int       n,
               unsigned int       batch_size,
               unsigned int       type,
               const vrna_param_t *params)
{
  const unsigned int size        = j - i - 1;
  const size_t       pitch       = n + 2;
  const short        *encoded    = sequence + static_cast<size_t>(batch) * pitch;
  const char         *characters = sequence_chars + static_cast<size_t>(batch) * (n + 1);
  int                energy;

  if (params->model_details.noGUclosure && ((type == 3) || (type == 4)))
    return kInf;

  if (hairpin_size_energies)
    energy = hairpin_size_energies[size];
  else if (size <= 30)
    energy = params->hairpin[size];
  else
    energy = params->hairpin[30] + static_cast<int>(params->lxc * log(static_cast<double>(size) / 30.));

  if (size < 3)
    return energy;

  if (params->model_details.special_hp && ((size == 3) || (size == 4) || (size == 6))) {
    int special = special_hairpin_energy(characters + i - 1, size, params);
    if (special < kInf)
      return special;

    if (size == 3)
      return energy + ((type > 2) ? params->TerminalAU : 0);
  }

  energy += params->mismatchH[type][encoded[i + 1]][encoded[j - 1]];
  return energy;
}


__device__ int
internal_energy(unsigned int       n1,
                unsigned int       n2,
                unsigned int       type,
                unsigned int       type2,
                int                si1,
                int                sj1,
                int                sp1,
                int                sq1,
                const LoopShape   *loop_shapes,
                const vrna_param_t *params)
{
  const bool no_close = params->model_details.noGUclosure &&
                        ((type == 3) || (type == 4) || (type2 == 3) || (type2 == 4));

  if (loop_shapes) {
    const LoopShape shape = loop_shapes[n1 * (MAXLOOP + 1) + n2];
    if (shape.kind == kLoopShapeStack)
      return params->stack[type][type2] + params->SaltStack;

    if (no_close)
      return kInf;

    switch (shape.kind) {
      case kLoopShapeBulge:
        if (n1 + n2 == 1)
          return shape.base + params->stack[type][type2];
        return shape.base + ((type > 2) ? params->TerminalAU : 0) +
               ((type2 > 2) ? params->TerminalAU : 0);

      case kLoopShapeInt11:
        return params->int11[type][type2][si1][sj1];

      case kLoopShapeInt21Forward:
        return params->int21[type][type2][si1][sq1][sj1];

      case kLoopShapeInt21Reverse:
        return params->int21[type2][type][sq1][si1][sp1];

      case kLoopShapeInt22:
        return params->int22[type][type2][si1][sp1][sq1][sj1];

      case kLoopShapeGeneric1n:
        return shape.base + params->mismatch1nI[type][si1][sj1] +
               params->mismatch1nI[type2][sq1][sp1];

      case kLoopShapeInt23:
        return shape.base + params->mismatch23I[type][si1][sj1] +
               params->mismatch23I[type2][sq1][sp1];

      case kLoopShapeGeneric:
        return shape.base + params->mismatchI[type][si1][sj1] +
               params->mismatchI[type2][sq1][sp1];

      default:
        return kInf;
    }
  }

  const unsigned int nl = (n1 > n2) ? n1 : n2;
  const unsigned int ns = (n1 > n2) ? n2 : n1;
  int energy = 0;

  if (nl == 0)
    return params->stack[type][type2] + params->SaltStack;

  if (no_close)
    return kInf;

  switch (ns) {
    case 0:
      energy = params->bulge[nl];
      if (nl == 1) {
        energy += params->stack[type][type2];
      } else {
        if (type > 2)
          energy += params->TerminalAU;
        if (type2 > 2)
          energy += params->TerminalAU;
      }
      break;

    case 1:
      if (nl == 1) {
        energy = params->int11[type][type2][si1][sj1];
      } else if (nl == 2) {
        if (n1 == 1)
          energy = params->int21[type][type2][si1][sq1][sj1];
        else
          energy = params->int21[type2][type][sq1][si1][sp1];
      } else {
        energy  = params->internal_loop[nl + 1];
        energy += minimum(kMaxNinio, static_cast<int>(nl - ns) * params->ninio[2]);
        energy += params->mismatch1nI[type][si1][sj1] +
                  params->mismatch1nI[type2][sq1][sp1];
      }
      break;

    case 2:
      if (nl == 2) {
        energy = params->int22[type][type2][si1][sp1][sq1][sj1];
        break;
      }
      if (nl == 3) {
        energy  = params->internal_loop[5] + params->ninio[2];
        energy += params->mismatch23I[type][si1][sj1] +
                  params->mismatch23I[type2][sq1][sp1];
        break;
      }
      [[fallthrough]];

    default:
      energy  = params->internal_loop[nl + ns];
      energy += minimum(kMaxNinio, static_cast<int>(nl - ns) * params->ninio[2]);
      energy += params->mismatchI[type][si1][sj1] +
                params->mismatchI[type2][sq1][sp1];
      break;
  }

  return energy;
}


__global__ void
initialize_matrices(int    *m2,
                    size_t count)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    m2[index] = kInf;
}


__global__ void
initialize_compact_m(short  *m,
                     size_t count)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    m[index] = SHRT_MAX;
}


__global__ void
initialize_pair_types(unsigned char      *pair_types,
                      const short        *sequence2,
                      size_t             dense_count,
                      unsigned int       n,
                      unsigned int       batch_size,
                      const vrna_param_t *params)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= dense_count)
    return;

  const unsigned int batch = index % batch_size;
  const size_t       cell  = index / batch_size;
  const unsigned int i     = cell / (n + 1);
  const unsigned int j     = cell % (n + 1);
  unsigned char      type  = 0;

  if ((i > 0) &&
      (j > i) &&
      ((j - i) < static_cast<unsigned int>(params->model_details.max_bp_span)) &&
      ((j - i) > static_cast<unsigned int>(params->model_details.min_loop_size))) {
    const size_t pitch = n + 2;
    type = params->model_details.pair[
      sequence2[static_cast<size_t>(batch) * pitch + i]
    ][sequence2[static_cast<size_t>(batch) * pitch + j]];

    if (params->model_details.noGU && ((type == 3) || (type == 4)))
      type = 0;
  }

  pair_types[index] = type;
}


__global__ void
initialize_pair_bits(unsigned int        *pair_bits,
                     const unsigned char *pair_types,
                     const short         *sequence2,
                     size_t              bit_count,
                     unsigned int        n,
                     unsigned int        words,
                     unsigned int        batch_size,
                     const vrna_param_t  *params)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= bit_count)
    return;

  const unsigned int batch = index % batch_size;
  const size_t cell = index / batch_size;
  const unsigned int p = cell / words;
  const unsigned int word = cell % words;
  const unsigned int first_q = word * 32;
  unsigned int bits = 0;

  for (unsigned int bit = 0; bit < 32; bit++) {
    const unsigned int q = first_q + bit;
    if ((q <= n) && pair_type_at(pair_types,
                                 sequence2,
                                 p,
                                 q,
                                 batch,
                                 n,
                                 batch_size,
                                 params))
      bits |= 1U << bit;
  }

  pair_bits[index] = bits;
}


__device__ __forceinline__ int
evaluate_internal_candidate(int                  best,
                            const short          *c,
                            const unsigned char  *pair_types,
                            const short          *sequence2,
                            const short          *s,
                            unsigned int         i,
                            unsigned int         j,
                            unsigned int         p,
                            unsigned int         q,
                            unsigned int         u1,
                            unsigned int         type,
                            unsigned int         batch,
                            unsigned int         n,
                            unsigned int         batch_size,
                            const LoopShape      *loop_shapes,
                            const vrna_param_t   *params,
                            unsigned int         *winner,
                            unsigned int         *winner_unpaired,
                            unsigned long long   *finite_enclosed,
                            unsigned long long   *energy_evaluations)
{
  const unsigned int enclosed_index = dense_index(p, q, batch, n, batch_size);
  const unsigned int inner_type = pair_type_at(pair_types,
                                                sequence2,
                                                p,
                                                q,
                                                batch,
                                                n,
                                                batch_size,
                                                params);
  if (inner_type == 0)
    return best;

  const int enclosed = load_compact_m(c, enclosed_index, q - p);
  if (enclosed >= kInf)
    return best;

  if (finite_enclosed)
    (*finite_enclosed)++;

  const unsigned int reverse_type = params->model_details.rtype[inner_type];
  const unsigned int u2 = j - q - 1;
  if (energy_evaluations)
    (*energy_evaluations)++;
  const int loop = internal_energy(u1,
                                   u2,
                                   type,
                                   reverse_type,
                                   s[i + 1],
                                   s[j - 1],
                                   s[p - 1],
                                   s[q + 1],
                                   loop_shapes,
                                   params);
  const int candidate = add_minimum(enclosed, loop, best);
  if ((candidate < best) && winner && winner_unpaired) {
    const unsigned int total = u1 + u2;
    if (total == 0)
      *winner = kWinnerStack;
    else if ((u1 == 0) || (u2 == 0))
      *winner = kWinnerBulge;
    else
      *winner = kWinnerInternal;
    *winner_unpaired = total;
  }

  return candidate;
}


template <unsigned int LaneWidth>
__global__ void
compute_paired_span(short               *c,
                    const int           *m2,
                    const unsigned char *pair_types,
                    const unsigned int  *pair_bits,
                    const short         *sequence,
                    const short         *sequence2,
                    const char          *sequence_chars,
                    const int           *hairpin_size_energies,
                    const LoopShape     *loop_shapes,
                    unsigned int        n,
                    unsigned int        batch_size,
                    unsigned int        pair_words,
                    unsigned int        span,
                    bool                m2_ring,
                    const vrna_param_t  *params,
                    unsigned int        *overflow,
                    unsigned long long  *profile_counters)
{
  const unsigned int batch_lane = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int lane       = batch_lane % LaneWidth;
  const unsigned int batch      = batch_lane / LaneWidth;

  if (batch >= batch_size)
    return;

  const unsigned int i     = blockIdx.y + 1;
  const unsigned int j     = i + span;
  const unsigned int type  = pair_type_at(pair_types,
                                           sequence2,
                                           i,
                                           j,
                                           batch,
                                           n,
                                           batch_size,
                                           params);

  if ((lane == 0) && profile_counters)
    profile_add(profile_counters, kProfileOuterCells);

  if (type == 0)
    return;

  if ((lane == 0) && profile_counters)
    profile_add(profile_counters, kProfilePairableOuterCells);

  const size_t pitch   = n + 2;
  const short  *s      = sequence + static_cast<size_t>(batch) * pitch;
  const vrna_md_t &md  = params->model_details;
  int          best    = (lane == 0) ? hairpin_energy(sequence,
                                                      sequence_chars,
                                                      hairpin_size_energies,
                                                      i,
                                                      j,
                                                      batch,
                                                      n,
                                                      batch_size,
                                                      type,
                                                      params) : kInf;
  unsigned int winner = kWinnerHairpin;
  unsigned int winner_unpaired = UINT_MAX;
  unsigned long long potential_coordinates = 0;
  unsigned long long pair_bit_accepted = 0;
  unsigned long long finite_enclosed = 0;
  unsigned long long energy_evaluations = 0;

  const unsigned int max_p = minimum(j - 1, i + MAXLOOP + 1);
  for (unsigned int p = i + 1 + lane; p <= max_p; p += LaneWidth) {
    const unsigned int u1 = p - i - 1;
    unsigned int       q_min;

    if (j <= MAXLOOP - u1 + 1)
      q_min = p + 1;
    else
      q_min = j - 1 - (MAXLOOP - u1);

    const unsigned int paired_min = p + params->model_details.min_loop_size + 1;
    if (q_min < paired_min)
      q_min = paired_min;

    if (q_min >= j)
      continue;

    const unsigned int q_max = minimum(j - 1,
                                       p + static_cast<unsigned int>(md.max_bp_span) - 1);
    if (q_min > q_max)
      continue;

    potential_coordinates += q_max - q_min + 1;

    if (pair_bits) {
      unsigned int word = q_max / 32;
      const unsigned int first_word = q_min / 32;
      unsigned int high_bit = q_max % 32;

      while (true) {
        unsigned int bits = pair_bits[pair_bit_index(p,
                                                      word,
                                                      batch,
                                                      pair_words,
                                                      batch_size)];
        bits &= (high_bit == 31) ? UINT_MAX : ((1U << (high_bit + 1)) - 1U);
        if (word == first_word) {
          const unsigned int low_bit = q_min % 32;
          bits &= UINT_MAX << low_bit;
        }

        while (bits) {
          const unsigned int bit = 31U - static_cast<unsigned int>(__clz(bits));
          const unsigned int q = word * 32 + bit;
          pair_bit_accepted++;
          best = evaluate_internal_candidate(best,
                                             c,
                                             pair_types,
                                             sequence2,
                                             s,
                                             i,
                                             j,
                                             p,
                                             q,
                                             u1,
                                             type,
                                             batch,
                                             n,
                                             batch_size,
                                             loop_shapes,
                                             params,
                                             &winner,
                                             &winner_unpaired,
                                             &finite_enclosed,
                                             &energy_evaluations);
          bits &= ~(1U << bit);
        }

        if (word == first_word)
          break;
        word--;
        high_bit = 31;
      }
    } else {
      for (unsigned int q = q_max; q >= q_min; q--) {
        pair_bit_accepted++;
        best = evaluate_internal_candidate(best,
                                           c,
                                           pair_types,
                                           sequence2,
                                           s,
                                           i,
                                           j,
                                           p,
                                           q,
                                           u1,
                                           type,
                                           batch,
                                           n,
                                           batch_size,
                                           loop_shapes,
                                           params,
                                           &winner,
                                           &winner_unpaired,
                                           &finite_enclosed,
                                           &energy_evaluations);
      }
    }
  }

  if ((lane == 0) &&
      (i + 1 < j) &&
      (params->model_details.noGUclosure == 0 || (type != 3 && type != 4))) {
    const int branches = m2_ring ?
                         m2[m2_ring_index(span - 2, i + 1, batch, n, batch_size)] :
                         m2[dense_index(i + 1, j - 1, batch, n, batch_size)];
    if (branches < kInf) {
      const unsigned int reverse_type = params->model_details.pair[
        sequence2[static_cast<size_t>(batch) * pitch + j]
      ][sequence2[static_cast<size_t>(batch) * pitch + i]];
      int closing = multibranch_stem_energy(reverse_type, s[j - 1], s[i + 1], params) +
                    params->MLclosing;
      const int multibranch = add_minimum(branches, closing, best);
      if (multibranch < best) {
        best = multibranch;
        winner = kWinnerMultibranch;
        winner_unpaired = UINT_MAX;
      }
    }
  }

  const unsigned int active = __activemask();
  for (unsigned int offset = LaneWidth / 2; offset > 0; offset /= 2) {
    const int other_best = __shfl_down_sync(active, best, offset, LaneWidth);
    const unsigned int other_winner = __shfl_down_sync(active, winner, offset, LaneWidth);
    const unsigned int other_unpaired = __shfl_down_sync(active,
                                                          winner_unpaired,
                                                          offset,
                                                          LaneWidth);
    if (other_best < best) {
      best = other_best;
      winner = other_winner;
      winner_unpaired = other_unpaired;
    }
  }

  if (profile_counters) {
    profile_add(profile_counters,
                kProfilePotentialInternalCoordinates,
                potential_coordinates);
    profile_add(profile_counters,
                kProfilePairBitAcceptedCoordinates,
                pair_bit_accepted);
    profile_add(profile_counters, kProfileFiniteEnclosedCells, finite_enclosed);
    profile_add(profile_counters, kProfileInternalEnergyEvaluations, energy_evaluations);
  }

  if (lane == 0) {
    if (profile_counters && (best < kInf)) {
      profile_add(profile_counters, kProfileWinnerHairpin + winner);
      if (winner_unpaired <= MAXLOOP)
        profile_add(profile_counters, kProfileWinnerUnpairedBase + winner_unpaired);
    }
    store_compact_m(c,
                    dense_index(i, j, batch, n, batch_size),
                    span,
                    batch,
                    best,
                    overflow);
  }
}


template <unsigned int LaneWidth>
cudaError_t
launch_paired_span(short               *c,
                   const int           *m2,
                   const unsigned char *pair_types,
                   const unsigned int  *pair_bits,
                   const short         *sequence,
                   const short         *sequence2,
                   const char          *sequence_chars,
                   const int           *hairpin_size_energies,
                   const LoopShape     *loop_shapes,
                   unsigned int        n,
                   unsigned int        batch_size,
                   unsigned int        pair_words,
                   unsigned int        span,
                   bool                m2_ring,
                   const vrna_param_t  *params,
                   unsigned int        *overflow,
                   unsigned long long  *profile_counters,
                   dim3                blocks)
{
  compute_paired_span<LaneWidth><<<blocks, kBlockSize>>>(c,
                                                         m2,
                                                         pair_types,
                                                         pair_bits,
                                                         sequence,
                                                         sequence2,
                                                         sequence_chars,
                                                         hairpin_size_energies,
                                                         loop_shapes,
                                                         n,
                                                         batch_size,
                                                         pair_words,
                                                         span,
                                                         m2_ring,
                                                         params,
                                                         overflow,
                                                         profile_counters);
  return cudaGetLastError();
}


__global__ void
compute_multibranch_span(const short         *c,
                         short               *m,
                         int                 *m2,
                         const unsigned char *pair_types,
                         const short         *sequence,
                         const short         *sequence2,
                         unsigned int        n,
                         unsigned int        batch_size,
                         unsigned int        span,
                         unsigned int        lane_width,
                         const vrna_param_t  *params,
                         unsigned int        *overflow)
{
  const unsigned int warp_lane        = threadIdx.x % warpSize;
  const unsigned int batches_per_warp = warpSize / lane_width;
  const unsigned int warp             = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  const unsigned int lane             = warp_lane / batches_per_warp;
  const unsigned int batch            = warp * batches_per_warp + warp_lane % batches_per_warp;

  if (batch >= batch_size)
    return;

  const unsigned int i     = blockIdx.y + 1;
  const unsigned int j     = i + span;
  const size_t       pitch = n + 2;
  const short        *s    = sequence + static_cast<size_t>(batch) * pitch;
  int                split = kInf;

  unsigned int k = i + 1 + lane;
  if (k + 1 < j) {
    unsigned int left_index  = dense_index(i, k, batch, n, batch_size);
    unsigned int right_index = dense_index(k + 1, j, batch, n, batch_size);
    const unsigned int left_step  = lane_width * batch_size;
    const unsigned int right_step = lane_width * (n + 1) * batch_size;

    for (; k + 1 < j; k += lane_width, left_index += left_step, right_index += right_step) {
      const int left  = load_compact_m(m, left_index, k - i);
      const int right = load_compact_m(m, right_index, j - k - 1);
      split = add_minimum(left, right, split);
    }
  }

  const unsigned int active = __activemask();
  for (unsigned int offset = lane_width / 2; offset > 0; offset /= 2)
    split = minimum(split,
                    __shfl_down_sync(active,
                                     split,
                                     offset * batches_per_warp));

  if (lane != 0)
    return;

  const size_t ij = dense_index(i, j, batch, n, batch_size);
  m2[ij] = split;

  int best = split;
  const int paired = load_compact_m(c, static_cast<unsigned int>(ij), span);
  if (paired < kInf) {
    const unsigned int type = pair_type_at(pair_types,
                                           sequence2,
                                           i,
                                           j,
                                           batch,
                                           n,
                                           batch_size,
                                           params);
    const int stem = multibranch_stem_energy(type,
                                              (i == 1) ? s[n] : s[i - 1],
                                              s[j + 1],
                                              params);
    best = add_minimum(paired, stem, best);
  }

  if (j > i + 1) {
    best = add_minimum(load_compact_m(m,
                                      dense_index(i, j - 1, batch, n, batch_size),
                                      span - 1),
                       params->MLbase,
                       best);
    best = add_minimum(load_compact_m(m,
                                      dense_index(i + 1, j, batch, n, batch_size),
                                      span - 1),
                       params->MLbase,
                       best);
  }

  store_compact_m(m, ij, span, batch, best, overflow);
}


__global__ void
compute_multibranch_sparse_span(const short         *c,
                                short               *m,
                                int                 *m2,
                                const unsigned char *pair_types,
                                const short         *sequence,
                                const short         *sequence2,
                                unsigned short      *candidate_count,
                                unsigned short      *candidates,
                                unsigned int        candidate_capacity,
                                unsigned int        n,
                                unsigned int        batch_size,
                                unsigned int        span,
                                unsigned int        lane_width,
                                bool                m2_ring,
                                const vrna_param_t  *params,
                                unsigned int        *overflow,
                                bool                validate,
                                unsigned int        *validation_mismatch,
                                unsigned long long  *profile_counters)
{
  const unsigned int warp_lane        = threadIdx.x % warpSize;
  const unsigned int batches_per_warp = warpSize / lane_width;
  const unsigned int warp             = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  const unsigned int lane             = warp_lane / batches_per_warp;
  const unsigned int batch            = warp * batches_per_warp + warp_lane % batches_per_warp;

  if (batch >= batch_size)
    return;

  const unsigned int i     = blockIdx.y + 1;
  const unsigned int j     = i + span;
  const size_t       pitch = n + 2;
  const short        *s    = sequence + static_cast<size_t>(batch) * pitch;
  const size_t       column = static_cast<size_t>(j) * batch_size + batch;
  const unsigned int count = candidate_count[column];
  int                split = kInf;
  unsigned long long candidate_comparisons = 0;

  if ((lane == 0) && (span > 1))
    split = add_minimum(m2_ring ?
                        m2[m2_ring_index(span - 1, i, batch, n, batch_size)] :
                        m2[dense_index(i, j - 1, batch, n, batch_size)],
                        params->MLbase,
                        split);

  for (unsigned int entry = lane; entry < count; entry += lane_width) {
    const unsigned int a = candidates[candidate_index(j,
                                                       entry,
                                                       batch,
                                                       candidate_capacity,
                                                       batch_size)];
    /* Candidates are appended in descending start-position order. */
    if (a < i + 2)
      break;

    candidate_comparisons++;

    const int left = load_compact_m(m,
                                     dense_index(i, a - 1, batch, n, batch_size),
                                     a - i - 1);
    const int right = load_compact_m(m,
                                      dense_index(a, j, batch, n, batch_size),
                                      j - a);
    split = add_minimum(left, right, split);
  }

  const unsigned int active = __activemask();
  for (unsigned int offset = lane_width / 2; offset > 0; offset /= 2)
    split = minimum(split,
                    __shfl_down_sync(active,
                                     split,
                                     offset * batches_per_warp));

  int dense_split = kInf;
  if (validate) {
    unsigned int k = i + 1 + lane;
    if (k + 1 < j) {
      unsigned int left_index  = dense_index(i, k, batch, n, batch_size);
      unsigned int right_index = dense_index(k + 1, j, batch, n, batch_size);
      const unsigned int left_step  = lane_width * batch_size;
      const unsigned int right_step = lane_width * (n + 1) * batch_size;

      for (; k + 1 < j; k += lane_width, left_index += left_step, right_index += right_step) {
        const int left  = load_compact_m(m, left_index, k - i);
        const int right = load_compact_m(m, right_index, j - k - 1);
        dense_split = add_minimum(left, right, dense_split);
      }
    }

    for (unsigned int offset = lane_width / 2; offset > 0; offset /= 2)
      dense_split = minimum(dense_split,
                            __shfl_down_sync(active,
                                             dense_split,
                                             offset * batches_per_warp));
  }

  if (lane != 0)
  {
    if (profile_counters)
      profile_add(profile_counters, kProfileCandidateComparisons, candidate_comparisons);
    return;
  }

  if (profile_counters)
    profile_add(profile_counters, kProfileCandidateComparisons, candidate_comparisons);

  if (validate && (dense_split != split))
    atomicExch(validation_mismatch, 1U);

  const size_t ij = dense_index(i, j, batch, n, batch_size);
  if (m2_ring)
    m2[m2_ring_index(span, i, batch, n, batch_size)] = split;
  else
    m2[ij] = split;

  int nonclosed = split;
  int extension_best = kInf;
  if (span > 1) {
    extension_best = add_minimum(load_compact_m(m,
                                                dense_index(i, j - 1, batch, n, batch_size),
                                                span - 1),
                                 params->MLbase,
                                 extension_best);
    extension_best = add_minimum(load_compact_m(m,
                                                dense_index(i + 1, j, batch, n, batch_size),
                                                span - 1),
                                 params->MLbase,
                                 extension_best);
    nonclosed = minimum(nonclosed, extension_best);
  }

  if (profile_counters && (extension_best < split))
    profile_add(profile_counters, kProfileRightExtensionWins);

  int branch = kInf;
  const int paired = load_compact_m(c, static_cast<unsigned int>(ij), span);
  if (paired < kInf) {
    const unsigned int type = pair_type_at(pair_types,
                                           sequence2,
                                           i,
                                           j,
                                           batch,
                                           n,
                                           batch_size,
                                           params);
    const int stem = multibranch_stem_energy(type,
                                              (i == 1) ? s[n] : s[i - 1],
                                              s[j + 1],
                                              params);
    branch = add_minimum(paired, stem, branch);
  }

  if (branch < nonclosed) {
    if (profile_counters)
      profile_add(profile_counters, kProfilePairedBranchWins);
    const unsigned int entry = candidate_count[column];
    if (entry < candidate_capacity) {
      candidates[candidate_index(j,
                                  entry,
                                  batch,
                                  candidate_capacity,
                                  batch_size)] = static_cast<unsigned short>(i);
      candidate_count[column] = static_cast<unsigned short>(entry + 1);
      if (profile_counters) {
        profile_add(profile_counters, kProfileCandidateTotal);
        atomicMax(profile_counters + kProfileCandidateMaximum,
                  static_cast<unsigned long long>(entry + 1));
      }
    } else {
      atomicExch(overflow + batch, 1U);
      if (profile_counters)
        profile_add(profile_counters, kProfileCandidateCapacityFallbacks);
    }
  }

  store_compact_m(m,
                  static_cast<unsigned int>(ij),
                  span,
                  batch,
                  minimum(branch, nonclosed),
                  overflow);
}


__global__ void
compute_exterior(const short        *c,
                 int                *f5,
                 const unsigned char *pair_types,
                 const short        *sequence,
                 const short        *sequence2,
                 unsigned int       n,
                 unsigned int       batch_size,
                 const vrna_param_t *params)
{
  __shared__ int candidates[kBlockSize];

  constexpr unsigned int i_lanes = kBlockSize / kExteriorBatchTile;
  const unsigned int batch_lane  = threadIdx.x % kExteriorBatchTile;
  const unsigned int i_lane      = threadIdx.x / kExteriorBatchTile;
  const unsigned int batch       = blockIdx.x * kExteriorBatchTile + batch_lane;
  const bool         valid       = batch < batch_size;
  const size_t       pitch       = n + 2;
  const short        *s          = valid ? sequence + static_cast<size_t>(batch) * pitch : nullptr;

  if (valid && (i_lane == 0))
    f5[batch] = 0;
  __syncthreads();

  for (unsigned int j = 1; j <= n; j++) {
    int best = (valid && (i_lane == 0)) ?
               f5[static_cast<size_t>(j - 1) * batch_size + batch] : kInf;
    const int sj1 = (valid && (j < n)) ? s[j + 1] : -1;

    for (unsigned int i = i_lane + 1; valid && (i < j); i += i_lanes) {
      const int paired = load_compact_m(c,
                                        dense_index(i, j, batch, n, batch_size),
                                        j - i);
      if (paired >= kInf)
        continue;

      const unsigned int type = pair_type_at(pair_types,
                                              sequence2,
                                              i,
                                              j,
                                              batch,
                                              n,
                                              batch_size,
                                              params);
      const int stem = exterior_stem_energy(type, (i > 1) ? s[i - 1] : -1, sj1, params);
      const int prefix = (i > 1) ? f5[static_cast<size_t>(i - 1) * batch_size + batch] : 0;
      int candidate = add_minimum(paired, stem, kInf);
      candidate     = add_minimum(prefix, candidate, kInf);
      best          = minimum(best, candidate);
    }

    candidates[threadIdx.x] = best;
    __syncthreads();

    if (valid && (i_lane == 0)) {
      for (unsigned int lane = 1; lane < i_lanes; lane++)
        best = minimum(best, candidates[lane * kExteriorBatchTile + batch_lane]);
      f5[static_cast<size_t>(j) * batch_size + batch] = best;
    }
    __syncthreads();
  }
}


__device__ __forceinline__ bool
trace_push(TraceSector    *stack,
           unsigned int   capacity,
           unsigned int   &top,
           unsigned int   i,
           unsigned int   j,
           unsigned char  kind)
{
  if ((top >= capacity) || (i > USHRT_MAX) || (j > USHRT_MAX))
    return false;

  stack[top].i       = static_cast<unsigned short>(i);
  stack[top].j       = static_cast<unsigned short>(j);
  stack[top].kind    = kind;
  stack[top].padding[0] = stack[top].padding[1] = stack[top].padding[2] = 0;
  top++;
  return true;
}


__global__ void
compute_traceback(const short         *c,
                  const short         *m,
                  const int           *f5,
                  const unsigned char *pair_types,
                  const short         *sequence,
                  const short         *sequence2,
                  const char          *sequence_chars,
                  const int           *hairpin_size_energies,
                  const LoopShape     *loop_shapes,
                  char                *structures,
                  TraceSector         *stacks,
                  unsigned char       *trace_status,
                  const unsigned int  *overflow,
                  unsigned int        n,
                  unsigned int        batch_size,
                  const vrna_param_t  *params)
{
  const unsigned int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch >= batch_size)
    return;

  const size_t pitch = n + 2;
  const short *s     = sequence + static_cast<size_t>(batch) * pitch;
  const short *s2    = sequence2 + static_cast<size_t>(batch) * pitch;
  char *structure    = structures + static_cast<size_t>(batch) * (n + 1);
  TraceSector *stack = stacks + static_cast<size_t>(batch) * (n + 1);
  unsigned int top   = 0;
  unsigned int steps = 0;
  bool ok            = overflow[batch] == 0;

  for (unsigned int pos = 0; pos < n; pos++)
    structure[pos] = '.';
  structure[n] = '\0';

  if (ok)
    ok = trace_push(stack, n + 1, top, 1, n, kTraceF5);

  while (ok && (top > 0) && (steps++ < 8 * n + 8)) {
    const TraceSector sector = stack[--top];
    unsigned int i = sector.i;
    unsigned int j = sector.j;

    if (sector.kind == kTraceF5) {
      while ((j > 0) &&
             (f5[static_cast<size_t>(j) * batch_size + batch] ==
              f5[static_cast<size_t>(j - 1) * batch_size + batch]))
        j--;

      if (j < 2)
        continue;

      const int target = f5[static_cast<size_t>(j) * batch_size + batch];
      const int sj1 = (j < n) ? s[j + 1] : -1;
      bool found = false;

      for (unsigned int u = j - 1; u >= 1; u--) {
        const unsigned int index = dense_index(u, j, batch, n, batch_size);
        const int paired = load_compact_m(c, index, j - u);
        if (paired < kInf) {
          const unsigned int type = pair_type_at(pair_types,
                                                  sequence2,
                                                  u,
                                                  j,
                                                  batch,
                                                  n,
                                                  batch_size,
                                                  params);
          const int stem = exterior_stem_energy(type,
                                                 (u > 1) ? s[u - 1] : -1,
                                                 sj1,
                                                 params);
          const int prefix = (u > 1) ?
                             f5[static_cast<size_t>(u - 1) * batch_size + batch] : 0;
          if ((prefix < kInf) && (paired + stem + prefix == target)) {
            ok = trace_push(stack, n + 1, top, 1, u - 1, kTraceF5) &&
                 trace_push(stack, n + 1, top, u, j, kTraceC);
            found = ok;
            break;
          }
        }

        if (u == 1)
          break;
      }

      if (!found)
        ok = false;

      continue;
    }

    if ((i == 0) || (j <= i) || (j > n)) {
      ok = false;
      continue;
    }

    if (sector.kind == kTraceM) {
      int target = load_compact_m(m,
                                  dense_index(i, j, batch, n, batch_size),
                                  j - i);

      while (j > i + 1) {
        const int previous = load_compact_m(m,
                                             dense_index(i, j - 1, batch, n, batch_size),
                                             j - i - 1);
        if ((previous >= kInf) || (previous + params->MLbase != target))
          break;
        j--;
        target = previous;
      }

      while (i + 1 < j) {
        const int previous = load_compact_m(m,
                                             dense_index(i + 1, j, batch, n, batch_size),
                                             j - i - 1);
        if ((previous >= kInf) || (previous + params->MLbase != target))
          break;
        i++;
        target = previous;
      }

      const unsigned int index = dense_index(i, j, batch, n, batch_size);
      const int paired = load_compact_m(c, index, j - i);
      if (paired < kInf) {
        const unsigned int type = pair_type_at(pair_types,
                                                sequence2,
                                                i,
                                                j,
                                                batch,
                                                n,
                                                batch_size,
                                                params);
        const int stem = multibranch_stem_energy(type,
                                                  (i == 1) ? s[n] : s[i - 1],
                                                  s[j + 1],
                                                  params);
        if (paired + stem == target) {
          ok = trace_push(stack, n + 1, top, i, j, kTraceC);
          continue;
        }
      }

      bool found = false;
      for (unsigned int u = i + 1; u + 1 < j; u++) {
        const int left = load_compact_m(m,
                                         dense_index(i, u, batch, n, batch_size),
                                         u - i);
        const int right = load_compact_m(m,
                                          dense_index(u + 1, j, batch, n, batch_size),
                                          j - u - 1);
        if ((left < kInf) && (right < kInf) && (left + right == target)) {
          ok = trace_push(stack, n + 1, top, i, u, kTraceM) &&
               trace_push(stack, n + 1, top, u + 1, j, kTraceM);
          found = ok;
          break;
        }
      }

      if (!found)
        ok = false;

      continue;
    }

    if (sector.kind != kTraceC) {
      ok = false;
      continue;
    }

    structure[i - 1] = '(';
    structure[j - 1] = ')';

    const unsigned int index = dense_index(i, j, batch, n, batch_size);
    const unsigned int type  = pair_type_at(pair_types,
                                             sequence2,
                                             i,
                                             j,
                                             batch,
                                             n,
                                             batch_size,
                                             params);
    const int target = load_compact_m(c, index, j - i);
    if ((type == 0) || (target >= kInf)) {
      ok = false;
      continue;
    }

    if (hairpin_energy(sequence,
                       sequence_chars,
                       hairpin_size_energies,
                       i,
                       j,
                       batch,
                       n,
                       batch_size,
                       type,
                       params) == target)
      continue;

    bool found = false;
    const unsigned int max_p = minimum(j - 1, i + MAXLOOP + 1);
    for (unsigned int p = i + 1; (p <= max_p) && !found; p++) {
      const unsigned int u1 = p - i - 1;
      unsigned int q_min;

      if (j <= MAXLOOP - u1 + 1)
        q_min = p + 1;
      else
        q_min = j - 1 - (MAXLOOP - u1);

      const unsigned int paired_min = p + params->model_details.min_loop_size + 1;
      if (q_min < paired_min)
        q_min = paired_min;

      const unsigned int q_max = minimum(j - 1,
                                         p + static_cast<unsigned int>(
                                           params->model_details.max_bp_span) - 1);
      if (q_min > q_max)
        continue;

      for (unsigned int q = q_max; q >= q_min; q--) {
        const unsigned int enclosed_index = dense_index(p, q, batch, n, batch_size);
        const unsigned int inner_type = pair_type_at(pair_types,
                                                      sequence2,
                                                      p,
                                                      q,
                                                      batch,
                                                      n,
                                                      batch_size,
                                                      params);
        const int enclosed = load_compact_m(c, enclosed_index, q - p);
        if ((inner_type != 0) && (enclosed < kInf)) {
          const unsigned int reverse_type = params->model_details.rtype[inner_type];
          const unsigned int u2 = j - q - 1;
          const int loop = internal_energy(u1,
                                           u2,
                                           type,
                                           reverse_type,
                                           s[i + 1],
                                           s[j - 1],
                                           s[p - 1],
                                           s[q + 1],
                                           loop_shapes,
                                           params);
          if (enclosed + loop == target) {
            ok = trace_push(stack, n + 1, top, p, q, kTraceC);
            found = ok;
            break;
          }
        }

        if (q == q_min)
          break;
      }
    }

    if (found)
      continue;

    if ((i + 1 < j) &&
        (params->model_details.noGUclosure == 0 || (type != 3 && type != 4))) {
      const unsigned int reverse_type = params->model_details.pair[s2[j]][s2[i]];
      const int closing = multibranch_stem_energy(reverse_type,
                                                   s[j - 1],
                                                   s[i + 1],
                                                   params) + params->MLclosing;
      const int branches = target - closing;

      for (unsigned int r = i + 2; r < j - 2; r++) {
        const int left = load_compact_m(m,
                                         dense_index(i + 1, r, batch, n, batch_size),
                                         r - i - 1);
        const int right = load_compact_m(m,
                                          dense_index(r + 1, j - 1, batch, n, batch_size),
                                          j - r - 2);
        if ((left < kInf) && (right < kInf) && (left + right == branches)) {
          ok = trace_push(stack, n + 1, top, i + 1, r, kTraceM) &&
               trace_push(stack, n + 1, top, r + 1, j - 1, kTraceM);
          found = ok;
          break;
        }
      }
    }

    if (!found)
      ok = false;
  }

  if (top != 0)
    ok = false;

  trace_status[batch] = ok ? 1 : 0;
}


__global__ void
gather_matrices(const short  *c,
                const short  *m,
                int          *packed_c,
                int          *packed_m,
                unsigned int n,
                unsigned int batch_size)
{
  const size_t total = static_cast<size_t>(n) * n * batch_size;
  const size_t tid   = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid >= total)
    return;

  const unsigned int batch = tid % batch_size;
  const size_t       cell  = tid / batch_size;
  const unsigned int i     = cell / n + 1;
  const unsigned int j     = cell % n + 1;

  if (i > j)
    return;

  const size_t triangle = static_cast<size_t>(j) * (j - 1) / 2 + i;
  const size_t triangular = static_cast<size_t>(n) * (n + 1) / 2 + 1;
  const size_t output   = static_cast<size_t>(batch) * triangular + triangle;
  const size_t input    = dense_index(i, j, batch, n, batch_size);

  packed_c[output] = load_compact_m(c, static_cast<unsigned int>(input), j - i);
  packed_m[output] = load_compact_m(m, static_cast<unsigned int>(input), j - i);
}


bool
default_hard_constraints(const vrna_fold_compound_t *fc)
{
  const vrna_hc_t *hc = fc->hc;
  const vrna_md_t &md = fc->params->model_details;
  const unsigned int n = fc->length;

  if ((!hc) ||
      (hc->type != VRNA_HC_DEFAULT) ||
      (hc->f != nullptr) ||
      (hc->data != nullptr) ||
      (!hc->mx))
    return false;

  for (unsigned int i = 1; i <= n; i++) {
    if (hc->mx[static_cast<size_t>(n) * i + i] != VRNA_CONSTRAINT_CONTEXT_ALL_LOOPS)
      return false;

    for (unsigned int j = i + 1; j <= n; j++) {
      unsigned char expected = VRNA_CONSTRAINT_CONTEXT_NONE;

      if (((j - i) < static_cast<unsigned int>(md.max_bp_span)) &&
          ((j - i) > static_cast<unsigned int>(md.min_loop_size))) {
        const unsigned int type = md.pair[fc->sequence_encoding2[i]][fc->sequence_encoding2[j]];
        if (type != 0) {
          if (((type == 3) || (type == 4)) && md.noGU) {
            expected = VRNA_CONSTRAINT_CONTEXT_NONE;
          } else {
            expected = VRNA_CONSTRAINT_CONTEXT_ALL_LOOPS;
            if (((type == 3) || (type == 4)) && md.noGUclosure)
              expected &= ~(VRNA_CONSTRAINT_CONTEXT_HP_LOOP | VRNA_CONSTRAINT_CONTEXT_MB_LOOP);
          }
        }
      }

      if (hc->mx[static_cast<size_t>(n) * i + j] != expected)
        return false;
    }
  }

  return true;
}


bool
eligible(vrna_fold_compound_t *fc)
{
  if ((!fc) ||
      (fc->type != VRNA_FC_TYPE_SINGLE) ||
      (fc->strands != 1) ||
      (!fc->params) ||
      (!fc->matrices) ||
      (fc->matrices->type != VRNA_MX_DEFAULT) ||
      (!fc->matrices->c) ||
      (!fc->matrices->fML) ||
      (!fc->matrices->f5) ||
      (!fc->sequence) ||
      (!fc->sequence_encoding) ||
      (!fc->sequence_encoding2) ||
      (fc->sc != nullptr) ||
      (fc->domains_up != nullptr) ||
      (fc->aux_grammar != nullptr) ||
      (fc->stat_cb != nullptr))
    return false;

  const vrna_md_t &md = fc->params->model_details;

  if ((md.dangles != 2) ||
      md.noLP ||
      md.logML ||
      md.circ ||
      md.gquad ||
      md.uniq_ML ||
      (md.backtrack_type != VRNA_MODEL_DEFAULT_BACKTRACK_TYPE) ||
      (md.salt != VRNA_MODEL_DEFAULT_SALT) ||
      (fc->params->param_file[0] != '\0'))
    return false;

  return default_hard_constraints(fc);
}


bool
same_bucket(const vrna_fold_compound_t *a,
            const vrna_fold_compound_t *b)
{
  return (a->length == b->length) &&
         (std::memcmp(a->params, b->params, sizeof(vrna_param_t)) == 0);
}


bool
environment_enabled(const char *name,
                    bool       fallback)
{
  const char *setting = std::getenv(name);
  if ((!setting) || (!setting[0]))
    return fallback;

  return std::strcmp(setting, "0") != 0;
}


unsigned int
sparse_candidate_capacity(unsigned int n)
{
  if ((!environment_enabled("VRNA_CUDA_SPARSE_M2", true)) ||
      (n > static_cast<unsigned int>(std::numeric_limits<unsigned short>::max())))
    return 0;

  unsigned long capacity = kDefaultCandidateCapacity;
  const char *setting = std::getenv("VRNA_CUDA_CANDIDATE_CAPACITY");
  if (setting && setting[0]) {
    char *end = nullptr;
    const unsigned long value = std::strtoul(setting, &end, 10);
    if ((end != setting) && (*end == '\0') && (value > 0))
      capacity = value;
  }

  capacity = std::min(capacity, static_cast<unsigned long>(n));
  capacity = std::min(capacity,
                      static_cast<unsigned long>(std::numeric_limits<unsigned short>::max()));
  return static_cast<unsigned int>(capacity);
}


size_t
chunk_limit(unsigned int n,
            size_t       count,
            bool         copy_matrices,
            bool         gpu_traceback)
{
  size_t free_bytes  = 0;
  size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
    return 0;

  const size_t dense_cells = static_cast<size_t>(n + 1) * (n + 1);
  const size_t triangular  = static_cast<size_t>(n) * (n + 1) / 2 + 1;
  const size_t candidate_capacity = sparse_candidate_capacity(n);
  const bool m2_ring = candidate_capacity && environment_enabled("VRNA_CUDA_M2_RING", true);
  const bool pair_bits = environment_enabled("VRNA_CUDA_PAIR_BITS", true);
  const size_t sparse_cells = candidate_capacity ?
                              static_cast<size_t>(n + 1) * (candidate_capacity + 1) : 0;
  const size_t m2_cells = m2_ring ? 2 * static_cast<size_t>(n + 1) : dense_cells;
  const size_t pair_bit_cells = pair_bits ?
                                static_cast<size_t>(n + 1) * ((n + 32) / 32) : 0;
  const size_t per_input   = sizeof(int) * (m2_cells + n + 1 +
                                             pair_bit_cells +
                                             (copy_matrices ? 2 * triangular : 0) + 1) +
                             sizeof(short) * (2 * dense_cells + 2 * (n + 2) + sparse_cells) +
                             sizeof(char) * (dense_cells + n + 1 +
                                             (gpu_traceback ? n + 2 : 0)) +
                             (gpu_traceback ? sizeof(TraceSector) * (n + 1) : 0);
  const size_t usable      = free_bytes * 7 / 10;
  size_t limit             = per_input ? usable / per_input : 0;
  const size_t index_limit = dense_cells ?
                             static_cast<size_t>(std::numeric_limits<unsigned int>::max()) / dense_cells : 0;

  const char *setting = std::getenv("VRNA_CUDA_BATCH_CHUNK");
  size_t configured = 256;
  if (setting && setting[0]) {
    char *end = nullptr;
    const unsigned long long value = std::strtoull(setting, &end, 10);
    if ((end != setting) && (value > 0))
      configured = static_cast<size_t>(value);
  }

  limit = std::min(limit, configured);
  limit = std::min(limit, index_limit);
  limit = std::min(limit, count);
  return limit;
}


unsigned int
lane_width_from_environment(const char   *name,
                            unsigned int fallback)
{
  const char *setting = std::getenv(name);
  if (setting && setting[0]) {
    char *end = nullptr;
    const unsigned long value = std::strtoul(setting, &end, 10);
    if ((end != setting) &&
        ((value == 1) || (value == 2) || (value == 4) || (value == 8) ||
         (value == 16) || (value == 32)))
      return static_cast<unsigned int>(value);
  }

  return fallback;
}


unsigned int
lane_width(size_t batch_size)
{
  unsigned int fallback;

  if (batch_size >= 128)
    fallback = 4;
  else if (batch_size >= 32)
    fallback = 8;
  else if (batch_size >= 8)
    fallback = 16;
  else
    fallback = 32;

  return lane_width_from_environment("VRNA_CUDA_LANES", fallback);
}


template <typename Function>
void
parallel_for(size_t   count,
             Function function)
{
  if (count == 0)
    return;

  std::atomic<size_t> next(0);
  const size_t worker_count = std::min(count,
                                       static_cast<size_t>(std::max(1U,
                                                                    std::thread::hardware_concurrency())));
  std::vector<std::thread> workers;

  workers.reserve(worker_count);
  try {
    for (size_t worker = 0; worker < worker_count; worker++) {
      workers.emplace_back([&]() {
        while (true) {
          const size_t i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= count)
            break;
          function(i);
        }
      });
    }
  } catch (...) {
    for (auto &worker : workers)
      worker.join();
    throw;
  }

  for (auto &worker : workers)
    worker.join();
}


bool
fold_chunk(vrna_fold_compound_t        **fc,
           const std::vector<size_t>   &bucket,
           size_t                      first,
           size_t                      batch_size,
           unsigned char               *handled,
           unsigned char               *traced,
           int                         *energies,
           char                        **structures,
           bool                        copy_matrices,
           bool                        gpu_traceback)
{
  const char *profile_setting = std::getenv("VRNA_CUDA_PROFILE");
  const bool profile = profile_setting && profile_setting[0] && (std::strcmp(profile_setting, "0") != 0);
  const bool detailed_profile = environment_enabled("VRNA_CUDA_PROFILE_COUNTERS", false);
  const auto wall_start = std::chrono::steady_clock::now();
  const unsigned int n             = fc[bucket[first]]->length;
  const size_t       dense_cells   = static_cast<size_t>(n + 1) * (n + 1);
  const size_t       dense_count   = dense_cells * batch_size;
  const size_t       triangular    = static_cast<size_t>(n) * (n + 1) / 2 + 1;
  const size_t       packed_count  = triangular * batch_size;
  const size_t       encoded_count = static_cast<size_t>(n + 2) * batch_size;
  const size_t       char_count    = static_cast<size_t>(n + 1) * batch_size;
  const size_t       f5_count      = static_cast<size_t>(n + 1) * batch_size;
  const size_t       traceback_count = gpu_traceback ? char_count : 0;
  const size_t       trace_stack_count = gpu_traceback ?
                                               static_cast<size_t>(n + 1) * batch_size : 0;
  const unsigned int lanes         = lane_width(batch_size);
  const unsigned int paired_lanes  = lane_width_from_environment("VRNA_CUDA_PAIRED_LANES", lanes);
  const unsigned int candidate_capacity = sparse_candidate_capacity(n);
  const bool sparse_m2 = candidate_capacity > 0;
  const bool m2_ring = sparse_m2 && environment_enabled("VRNA_CUDA_M2_RING", true);
  const bool validate_sparse = sparse_m2 &&
                               environment_enabled("VRNA_CUDA_VALIDATE_SPARSE_M2", false);
  const bool use_pair_bits = environment_enabled("VRNA_CUDA_PAIR_BITS", true);
  const bool precompute_hairpin = environment_enabled("VRNA_CUDA_PRECOMPUTE_HAIRPIN", true);
  const bool precompute_loop_shapes = environment_enabled("VRNA_CUDA_PRECOMPUTE_LOOP_SHAPES", false);
  int current_device = 0;
  int compute_capability_major = 0;
  const bool prefer_derived_pair_types =
    (cudaGetDevice(&current_device) == cudaSuccess) &&
    (cudaDeviceGetAttribute(&compute_capability_major,
                            cudaDevAttrComputeCapabilityMajor,
                            current_device) == cudaSuccess) &&
    (compute_capability_major >= 12);
  const bool derive_pair_types = environment_enabled("VRNA_CUDA_DERIVE_PAIR_TYPES",
                                                      prefer_derived_pair_types);
  const unsigned int pair_words = (n + 32) / 32;

  std::vector<short> host_sequence(encoded_count);
  std::vector<short> host_sequence2(encoded_count);
  std::vector<char>  host_chars(char_count);
  std::vector<int>   host_hairpin_size_energies(precompute_hairpin ? n + 1 : 0);
  std::vector<LoopShape> host_loop_shapes(precompute_loop_shapes ?
                                            static_cast<size_t>(MAXLOOP + 1) * (MAXLOOP + 1) : 0);
  std::unique_ptr<int[]> host_c(copy_matrices ? new int[packed_count] : nullptr);
  std::unique_ptr<int[]> host_m(copy_matrices ? new int[packed_count] : nullptr);
  std::unique_ptr<int[]> host_f5(new int[copy_matrices ?
                                         static_cast<size_t>(n + 1) * batch_size : batch_size]);
  std::unique_ptr<char[]> host_traceback(gpu_traceback ? new char[traceback_count] : nullptr);
  std::vector<unsigned char> host_trace_status(gpu_traceback ? batch_size : 0, 0);

  /* Populate pageable receive buffers concurrently.  Otherwise the first
   * device-to-host copy takes the page-fault cost serially in the CUDA
   * runtime, while value-initializing large std::vectors pays the same cost
   * on just one host thread. */
  if (copy_matrices) {
    parallel_for(batch_size, [&](size_t b) {
      std::memset(host_c.get() + b * triangular, 0, sizeof(int) * triangular);
      std::memset(host_m.get() + b * triangular, 0, sizeof(int) * triangular);
    });
  }
  std::memset(host_f5.get(), 0, sizeof(int) * (copy_matrices ? f5_count : batch_size));

  for (size_t b = 0; b < batch_size; b++) {
    vrna_fold_compound_t *item = fc[bucket[first + b]];
    std::memcpy(host_sequence.data() + b * (n + 2), item->sequence_encoding, sizeof(short) * (n + 2));
    std::memcpy(host_sequence2.data() + b * (n + 2), item->sequence_encoding2, sizeof(short) * (n + 2));
    std::memcpy(host_chars.data() + b * (n + 1), item->sequence, sizeof(char) * (n + 1));
  }

  if (precompute_hairpin) {
    const vrna_param_t *params = fc[bucket[first]]->params;
    for (unsigned int size = 0; size <= n; size++) {
      host_hairpin_size_energies[size] = (size <= 30) ?
                                           params->hairpin[size] :
                                           params->hairpin[30] +
                                           static_cast<int>(params->lxc *
                                                            std::log(static_cast<double>(size) / 30.));
    }
  }

  if (precompute_loop_shapes) {
    const vrna_param_t *params = fc[bucket[first]]->params;
    for (unsigned int n1 = 0; n1 <= MAXLOOP; n1++) {
      for (unsigned int n2 = 0; n2 <= MAXLOOP; n2++) {
        LoopShape &shape = host_loop_shapes[n1 * (MAXLOOP + 1) + n2];
        const unsigned int nl = std::max(n1, n2);
        const unsigned int ns = std::min(n1, n2);
        shape.base = 0;
        shape.kind = kLoopShapeInvalid;
        shape.padding[0] = shape.padding[1] = shape.padding[2] = 0;

        if (n1 + n2 > MAXLOOP)
          continue;
        if (nl == 0) {
          shape.kind = kLoopShapeStack;
        } else if (ns == 0) {
          shape.kind = kLoopShapeBulge;
          shape.base = params->bulge[nl];
        } else if (ns == 1) {
          if (nl == 1) {
            shape.kind = kLoopShapeInt11;
          } else if (nl == 2) {
            shape.kind = (n1 == 1) ? kLoopShapeInt21Forward : kLoopShapeInt21Reverse;
          } else {
            shape.kind = kLoopShapeGeneric1n;
            shape.base = params->internal_loop[nl + 1] +
                         std::min(kMaxNinio,
                                  static_cast<int>(nl - ns) * params->ninio[2]);
          }
        } else if ((ns == 2) && (nl == 2)) {
          shape.kind = kLoopShapeInt22;
        } else if ((ns == 2) && (nl == 3)) {
          shape.kind = kLoopShapeInt23;
          shape.base = params->internal_loop[5] + params->ninio[2];
        } else {
          shape.kind = kLoopShapeGeneric;
          shape.base = params->internal_loop[nl + ns] +
                       std::min(kMaxNinio,
                                static_cast<int>(nl - ns) * params->ninio[2]);
        }
      }
    }
  }

  const size_t candidate_columns = sparse_m2 ? static_cast<size_t>(n + 1) * batch_size : 0;
  const size_t candidate_entries = candidate_columns * candidate_capacity;
  const size_t pair_bit_count = use_pair_bits ?
                                static_cast<size_t>(n + 1) * pair_words * batch_size : 0;
  const size_t m2_count = m2_ring ? 2 * static_cast<size_t>(n + 1) * batch_size : dense_count;
  const size_t hairpin_size_count = precompute_hairpin ? n + 1 : 0;
  const size_t loop_shape_count = host_loop_shapes.size();
  const size_t pair_type_count = derive_pair_types ? 0 : dense_count;
  const size_t int_count = m2_count + f5_count + batch_size + 1 + pair_bit_count +
                           hairpin_size_count +
                           (copy_matrices ? 2 * packed_count : 0);
  const size_t short_count = 2 * dense_count + 2 * encoded_count +
                             candidate_columns + candidate_entries;
  const size_t profile_counter_count = detailed_profile ? kProfileCounterCount : 0;
  const size_t arena_bytes = sizeof(vrna_param_t) +
                             (detailed_profile ? alignof(unsigned long long) - 1 +
                              sizeof(unsigned long long) * profile_counter_count : 0) +
                             alignof(int) - 1 +
                             sizeof(int) * int_count +
                             (precompute_loop_shapes ? alignof(LoopShape) - 1 : 0) +
                             sizeof(LoopShape) * loop_shape_count +
                             alignof(short) - 1 + sizeof(short) * short_count +
                             sizeof(char) * (char_count + pair_type_count + batch_size +
                                             traceback_count) +
                             (gpu_traceback ? alignof(TraceSector) - 1 : 0) +
                             sizeof(TraceSector) * trace_stack_count;
  DeviceBuffer<unsigned char> device_arena;

  if (!device_arena.allocate(arena_bytes,
                             environment_enabled("VRNA_CUDA_ASYNC_ALLOC", true)))
    return false;

  size_t offset = 0;
  vrna_param_t *device_params = arena_take<vrna_param_t>(device_arena.get(), offset, 1);
  unsigned long long *device_profile_counters = detailed_profile ?
                                                        arena_take<unsigned long long>(device_arena.get(),
                                                                                       offset,
                                                                                       profile_counter_count) : nullptr;
  int *device_m2 = arena_take<int>(device_arena.get(), offset, m2_count);
  int *device_f5 = arena_take<int>(device_arena.get(), offset, f5_count);
  int *device_packed_c = copy_matrices ? arena_take<int>(device_arena.get(), offset, packed_count) : nullptr;
  int *device_packed_m = copy_matrices ? arena_take<int>(device_arena.get(), offset, packed_count) : nullptr;
  unsigned int *device_overflow = arena_take<unsigned int>(device_arena.get(), offset, batch_size);
  unsigned int *device_sparse_mismatch = arena_take<unsigned int>(device_arena.get(), offset, 1);
  unsigned int *device_pair_bits = use_pair_bits ?
                                   arena_take<unsigned int>(device_arena.get(),
                                                            offset,
                                                            pair_bit_count) : nullptr;
  int *device_hairpin_size_energies = precompute_hairpin ?
                                        arena_take<int>(device_arena.get(),
                                                        offset,
                                                        hairpin_size_count) : nullptr;
  LoopShape *device_loop_shapes = precompute_loop_shapes ?
                                   arena_take<LoopShape>(device_arena.get(),
                                                         offset,
                                                         loop_shape_count) : nullptr;
  short *device_c = arena_take<short>(device_arena.get(), offset, dense_count);
  short *device_m = arena_take<short>(device_arena.get(), offset, dense_count);
  short *device_sequence = arena_take<short>(device_arena.get(), offset, encoded_count);
  short *device_sequence2 = arena_take<short>(device_arena.get(), offset, encoded_count);
  unsigned short *device_candidate_count = sparse_m2 ?
                                             arena_take<unsigned short>(device_arena.get(),
                                                                         offset,
                                                                         candidate_columns) : nullptr;
  unsigned short *device_candidates = sparse_m2 ?
                                        arena_take<unsigned short>(device_arena.get(),
                                                                    offset,
                                                                    candidate_entries) : nullptr;
  char *device_chars = arena_take<char>(device_arena.get(), offset, char_count);
  unsigned char *device_pair_types = derive_pair_types ? nullptr :
                                           arena_take<unsigned char>(device_arena.get(),
                                                                     offset,
                                                                     pair_type_count);
  char *device_traceback = gpu_traceback ?
                           arena_take<char>(device_arena.get(), offset, traceback_count) : nullptr;
  unsigned char *device_trace_status = gpu_traceback ?
                                       arena_take<unsigned char>(device_arena.get(), offset, batch_size) : nullptr;
  TraceSector *device_trace_stack = gpu_traceback ?
                                    arena_take<TraceSector>(device_arena.get(), offset, trace_stack_count) : nullptr;
  std::vector<unsigned int> host_overflow(batch_size, 0);
  std::vector<unsigned long long> host_profile_counters(profile_counter_count, 0);

  if ((cudaMemcpy(device_sequence, host_sequence.data(), sizeof(short) * encoded_count, cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpy(device_sequence2, host_sequence2.data(), sizeof(short) * encoded_count, cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpy(device_chars, host_chars.data(), sizeof(char) * char_count, cudaMemcpyHostToDevice) != cudaSuccess) ||
      (precompute_hairpin &&
       (cudaMemcpy(device_hairpin_size_energies,
                   host_hairpin_size_energies.data(),
                   sizeof(int) * hairpin_size_count,
                   cudaMemcpyHostToDevice) != cudaSuccess)) ||
      (precompute_loop_shapes &&
       (cudaMemcpy(device_loop_shapes,
                   host_loop_shapes.data(),
                   sizeof(LoopShape) * loop_shape_count,
                   cudaMemcpyHostToDevice) != cudaSuccess)) ||
      (cudaMemcpy(device_params, fc[bucket[first]]->params, sizeof(vrna_param_t), cudaMemcpyHostToDevice) != cudaSuccess))
    return false;

  const auto upload_done = std::chrono::steady_clock::now();
  EventTimeline timeline;
  if (profile &&
      ((!timeline.initialize(static_cast<size_t>(2) * n + 2)) ||
       (!timeline.record())))
    return false;

  const unsigned int initialize_blocks = static_cast<unsigned int>((dense_count + kBlockSize - 1) / kBlockSize);
  const unsigned int m2_initialize_blocks = static_cast<unsigned int>((m2_count + kBlockSize - 1) /
                                                                      kBlockSize);
  initialize_matrices<<<m2_initialize_blocks, kBlockSize>>>(device_m2, m2_count);
  if (cudaGetLastError() != cudaSuccess)
    return false;
  initialize_compact_m<<<initialize_blocks, kBlockSize>>>(device_c, dense_count);
  if (cudaGetLastError() != cudaSuccess)
    return false;
  initialize_compact_m<<<initialize_blocks, kBlockSize>>>(device_m, dense_count);
  if ((cudaGetLastError() != cudaSuccess) ||
      (cudaMemset(device_overflow, 0, sizeof(unsigned int) * batch_size) != cudaSuccess) ||
      (cudaMemset(device_sparse_mismatch, 0, sizeof(unsigned int)) != cudaSuccess) ||
      (detailed_profile &&
       (cudaMemset(device_profile_counters,
                   0,
                   sizeof(unsigned long long) * profile_counter_count) != cudaSuccess)) ||
      (sparse_m2 &&
       (cudaMemset(device_candidate_count,
                   0,
                   sizeof(unsigned short) * candidate_columns) != cudaSuccess)))
    return false;
  if (!derive_pair_types) {
    initialize_pair_types<<<initialize_blocks, kBlockSize>>>(device_pair_types,
                                                             device_sequence2,
                                                             dense_count,
                                                             n,
                                                             static_cast<unsigned int>(batch_size),
                                                             device_params);
    if (cudaGetLastError() != cudaSuccess)
      return false;
  }
  if (use_pair_bits) {
    const unsigned int pair_bit_blocks = static_cast<unsigned int>((pair_bit_count + kBlockSize - 1) /
                                                                   kBlockSize);
    initialize_pair_bits<<<pair_bit_blocks, kBlockSize>>>(device_pair_bits,
                                                          device_pair_types,
                                                          device_sequence2,
                                                          pair_bit_count,
                                                          n,
                                                          pair_words,
                                                          static_cast<unsigned int>(batch_size),
                                                          device_params);
    if (cudaGetLastError() != cudaSuccess)
      return false;
  }
  if (profile && !timeline.record())
    return false;

  for (unsigned int span = 1; span < n; span++) {
    const unsigned int paired_batch_lanes = static_cast<unsigned int>(batch_size) * paired_lanes;
    const dim3 paired_blocks((paired_batch_lanes + kBlockSize - 1) / kBlockSize,
                             n - span);

      cudaError_t paired_error = cudaErrorInvalidValue;
#define LAUNCH_PAIRED(LANES)                                                        \
      paired_error = launch_paired_span<LANES>(device_c,                            \
                                                device_m2,                           \
                                                device_pair_types,                   \
                                                device_pair_bits,                    \
                                                device_sequence,                     \
                                                device_sequence2,                    \
                                                device_chars,                        \
                                                device_hairpin_size_energies,         \
                                                device_loop_shapes,                  \
                                                n,                                   \
                                                static_cast<unsigned int>(batch_size), \
                                                pair_words,                          \
                                                span,                                \
                                                m2_ring,                             \
                                                device_params,                       \
                                                device_overflow,                     \
                                                device_profile_counters,             \
                                                paired_blocks)

      switch (paired_lanes) {
        case 1:
          LAUNCH_PAIRED(1);
          break;
        case 2:
          LAUNCH_PAIRED(2);
          break;
        case 4:
          LAUNCH_PAIRED(4);
          break;
        case 8:
          LAUNCH_PAIRED(8);
          break;
        case 16:
          LAUNCH_PAIRED(16);
          break;
        case 32:
          LAUNCH_PAIRED(32);
          break;
        default:
          break;
      }
#undef LAUNCH_PAIRED

      if (paired_error != cudaSuccess)
        return false;
    if (profile && !timeline.record())
      return false;

    const unsigned int batch_lanes = static_cast<unsigned int>(batch_size) * lanes;
    const dim3 multibranch_blocks((batch_lanes + kBlockSize - 1) / kBlockSize,
                                  n - span);
    if (sparse_m2) {
      compute_multibranch_sparse_span<<<multibranch_blocks, kBlockSize>>>(
        device_c,
        device_m,
        device_m2,
        device_pair_types,
        device_sequence,
        device_sequence2,
        device_candidate_count,
        device_candidates,
        candidate_capacity,
        n,
        static_cast<unsigned int>(batch_size),
        span,
        lanes,
        m2_ring,
        device_params,
        device_overflow,
        validate_sparse,
        device_sparse_mismatch,
        device_profile_counters);
    } else {
      compute_multibranch_span<<<multibranch_blocks, kBlockSize>>>(device_c,
                                                                   device_m,
                                                                   device_m2,
                                                                   device_pair_types,
                                                                   device_sequence,
                                                                   device_sequence2,
                                                                   n,
                                                                   static_cast<unsigned int>(batch_size),
                                                                   span,
                                                                   lanes,
                                                                   device_params,
                                                                   device_overflow);
    }
    if (cudaGetLastError() != cudaSuccess)
      return false;
    if (profile && !timeline.record())
      return false;
  }

  const unsigned int exterior_blocks = static_cast<unsigned int>((batch_size + kExteriorBatchTile - 1) /
                                                                  kExteriorBatchTile);
  compute_exterior<<<exterior_blocks, kBlockSize>>>(device_c,
                                                    device_f5,
                                                    device_pair_types,
                                                    device_sequence,
                                                    device_sequence2,
                                                    n,
                                                    static_cast<unsigned int>(batch_size),
                                                    device_params);
  if (cudaGetLastError() != cudaSuccess)
    return false;
  if (profile && !timeline.record())
    return false;

  if (gpu_traceback) {
    const unsigned int traceback_blocks = static_cast<unsigned int>((batch_size + kBlockSize - 1) /
                                                                    kBlockSize);
    compute_traceback<<<traceback_blocks, kBlockSize>>>(device_c,
                                                        device_m,
                                                        device_f5,
                                                        device_pair_types,
                                                        device_sequence,
                                                        device_sequence2,
                                                        device_chars,
                                                        device_hairpin_size_energies,
                                                        device_loop_shapes,
                                                        device_traceback,
                                                        device_trace_stack,
                                                        device_trace_status,
                                                        device_overflow,
                                                        n,
                                                        static_cast<unsigned int>(batch_size),
                                                        device_params);
    if (cudaGetLastError() != cudaSuccess)
      return false;
  }

  if (cudaMemcpy(host_overflow.data(),
                 device_overflow,
                 sizeof(unsigned int) * batch_size,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;

  if (detailed_profile &&
      (cudaMemcpy(host_profile_counters.data(),
                  device_profile_counters,
                  sizeof(unsigned long long) * profile_counter_count,
                  cudaMemcpyDeviceToHost) != cudaSuccess))
    return false;

  if (validate_sparse) {
    unsigned int sparse_mismatch = 0;
    if (cudaMemcpy(&sparse_mismatch,
                   device_sparse_mismatch,
                   sizeof(sparse_mismatch),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;

    if (sparse_mismatch) {
      std::fprintf(stderr, "vrna-cuda: sparse M2 validation mismatch\n");
      return false;
    }
  }

  if (copy_matrices) {
    const size_t gather_work = static_cast<size_t>(n) * n * batch_size;
    const unsigned int gather_blocks = static_cast<unsigned int>((gather_work + kBlockSize - 1) / kBlockSize);
    gather_matrices<<<gather_blocks, kBlockSize>>>(device_c,
                                                   device_m,
                                                   device_packed_c,
                                                   device_packed_m,
                                                   n,
                                                   static_cast<unsigned int>(batch_size));
    if (cudaGetLastError() != cudaSuccess)
      return false;
  }

  auto kernels_done = upload_done;
  if (profile) {
    if ((!timeline.record()) || (!timeline.synchronize()))
      return false;
    kernels_done = std::chrono::steady_clock::now();
  }

  if (copy_matrices) {
    if ((cudaMemcpy(host_c.get(), device_packed_c, sizeof(int) * packed_count, cudaMemcpyDeviceToHost) != cudaSuccess) ||
        (cudaMemcpy(host_m.get(), device_packed_m, sizeof(int) * packed_count, cudaMemcpyDeviceToHost) != cudaSuccess) ||
        (cudaMemcpy(host_f5.get(), device_f5, sizeof(int) * f5_count, cudaMemcpyDeviceToHost) != cudaSuccess))
      return false;
  } else if ((cudaMemcpy(host_f5.get(),
                         device_f5 + static_cast<size_t>(n) * batch_size,
                         sizeof(int) * batch_size,
                         cudaMemcpyDeviceToHost) != cudaSuccess) ||
             (gpu_traceback &&
              ((cudaMemcpy(host_traceback.get(),
                           device_traceback,
                           sizeof(char) * traceback_count,
                           cudaMemcpyDeviceToHost) != cudaSuccess) ||
               (cudaMemcpy(host_trace_status.data(),
                           device_trace_status,
                           sizeof(unsigned char) * batch_size,
                           cudaMemcpyDeviceToHost) != cudaSuccess)))) {
    return false;
  }

  const auto download_done = std::chrono::steady_clock::now();

  /* The device gather writes one contiguous triangular matrix per input so
   * host delivery is a pair of linear copies rather than a cache-hostile
   * batch transpose. */
  parallel_for(batch_size, [&](size_t b) {
    if (host_overflow[b] || (gpu_traceback && !host_trace_status[b]))
      return;

    vrna_fold_compound_t *item = fc[bucket[first + b]];
    if (copy_matrices) {
      std::memcpy(item->matrices->c,
                  host_c.get() + b * triangular,
                  sizeof(int) * triangular);
      std::memcpy(item->matrices->fML,
                  host_m.get() + b * triangular,
                  sizeof(int) * triangular);

      for (unsigned int j = 0; j <= n; j++)
        item->matrices->f5[j] = host_f5[static_cast<size_t>(j) * batch_size + b];
    }

    const size_t original = bucket[first + b];
    if (gpu_traceback) {
      if ((!structures) || (!structures[original]))
        return;
      std::memcpy(structures[original],
                  host_traceback.get() + b * (n + 1),
                  sizeof(char) * (n + 1));
      traced[original] = 1;
    }
    energies[original] = copy_matrices ? item->matrices->f5[n] : host_f5[b];
    handled[original]  = 1;
  });

  const auto delivery_done = std::chrono::steady_clock::now();
  if (profile) {
    size_t event = 0;
    const float initialize_ms = timeline.interval(event, event + 1);
    event++;
    float paired_ms = 0.f;
    float multibranch_ms = 0.f;
    for (unsigned int span = 1; span < n; span++) {
      paired_ms += timeline.interval(event, event + 1);
      event++;
      multibranch_ms += timeline.interval(event, event + 1);
      event++;
    }
    const float exterior_ms = timeline.interval(event, event + 1);
    event++;
    const float gather_ms = copy_matrices ? timeline.interval(event, event + 1) : 0.f;
    const float kernel_ms = timeline.interval(0, event + 1);
    const double upload_ms = std::chrono::duration<double, std::milli>(upload_done - wall_start).count();
    const double download_ms = std::chrono::duration<double, std::milli>(download_done - kernels_done).count();
    const double delivery_ms = std::chrono::duration<double, std::milli>(delivery_done - download_done).count();
    std::fprintf(stderr,
                 "vrna-cuda profile: n=%u batch=%zu pair-types=%s setup+H2D=%.3f ms kernels=%.3f ms "
                 "(init=%.3f paired=%.3f multibranch=%.3f exterior=%.3f gather=%.3f) "
                 "D2H=%.3f ms host-delivery=%.3f ms\n",
                 n,
                 batch_size,
                 derive_pair_types ? "derived" : "dense",
                 upload_ms,
                 kernel_ms,
                 initialize_ms,
                 paired_ms,
                 multibranch_ms,
                 exterior_ms,
                 gather_ms,
                 download_ms,
                 delivery_ms);
  }

  if (detailed_profile) {
    const unsigned long long potential = host_profile_counters[kProfilePotentialInternalCoordinates];
    const unsigned long long accepted = host_profile_counters[kProfilePairBitAcceptedCoordinates];
    const unsigned long long candidate_columns_count = static_cast<unsigned long long>(n + 1) * batch_size;
    std::fprintf(stderr,
                 "vrna-cuda counters: outer=%llu pairable=%llu internal-potential=%llu "
                 "pairbits-accepted=%llu pairbits-rejected=%llu finite-enclosed=%llu "
                 "energy-evaluations=%llu winners(hairpin=%llu stack=%llu bulge=%llu "
                 "internal=%llu multibranch=%llu)\n",
                 host_profile_counters[kProfileOuterCells],
                 host_profile_counters[kProfilePairableOuterCells],
                 potential,
                 accepted,
                 potential - std::min(potential, accepted),
                 host_profile_counters[kProfileFiniteEnclosedCells],
                 host_profile_counters[kProfileInternalEnergyEvaluations],
                 host_profile_counters[kProfileWinnerHairpin],
                 host_profile_counters[kProfileWinnerStack],
                 host_profile_counters[kProfileWinnerBulge],
                 host_profile_counters[kProfileWinnerInternal],
                 host_profile_counters[kProfileWinnerMultibranch]);
    std::fprintf(stderr,
                 "vrna-cuda sparse counters: candidate-comparisons=%llu "
                 "right-extension-wins=%llu paired-branch-wins=%llu "
                 "capacity-fallbacks=%llu candidate-total=%llu "
                 "mean-candidates-per-column=%.6f max-candidates-per-column=%llu\n",
                 host_profile_counters[kProfileCandidateComparisons],
                 host_profile_counters[kProfileRightExtensionWins],
                 host_profile_counters[kProfilePairedBranchWins],
                 host_profile_counters[kProfileCandidateCapacityFallbacks],
                 host_profile_counters[kProfileCandidateTotal],
                 candidate_columns_count ?
                 static_cast<double>(host_profile_counters[kProfileCandidateTotal]) /
                 static_cast<double>(candidate_columns_count) : 0.,
                 host_profile_counters[kProfileCandidateMaximum]);
    std::fprintf(stderr, "vrna-cuda winner-unpaired:");
    for (unsigned int unpaired = 0; unpaired <= MAXLOOP; unpaired++)
      if (host_profile_counters[kProfileWinnerUnpairedBase + unpaired])
        std::fprintf(stderr,
                     " %u=%llu",
                     unpaired,
                     host_profile_counters[kProfileWinnerUnpairedBase + unpaired]);
    std::fprintf(stderr, "\n");
  }

  return true;
}


bool
fold_bucket(vrna_fold_compound_t      **fc,
            const std::vector<size_t> &bucket,
            unsigned char             *handled,
            unsigned char             *traced,
            int                       *energies,
            char                      **structures,
            bool                      copy_matrices,
            bool                      gpu_traceback)
{
  const unsigned int n = fc[bucket.front()]->length;
  const size_t limit = chunk_limit(n, bucket.size(), copy_matrices, gpu_traceback);
  if (limit == 0)
    return false;

  for (size_t first = 0; first < bucket.size(); first += limit) {
    const size_t count = std::min(limit, bucket.size() - first);
    if (!fold_chunk(fc,
                    bucket,
                    first,
                    count,
                    handled,
                    traced,
                    energies,
                    structures,
                    copy_matrices,
                    gpu_traceback))
      return false;
  }

  return true;
}

}  // namespace


extern "C" int
vrna_cuda_backend_abi_version(void)
{
  return VRNA_CUDA_BACKEND_ABI_VERSION;
}


extern "C" int
vrna_cuda_mfe_batch(vrna_fold_compound_t **fc,
                    size_t               count,
                    unsigned char        *handled,
                    unsigned char        *traced,
                    int                  *energies,
                    char                 **structures,
                    unsigned int         flags)
{
  const bool gpu_traceback = (flags & VRNA_CUDA_BACKEND_TRACEBACK) != 0;
  const bool copy_matrices = (flags & VRNA_CUDA_BACKEND_COPY_MATRICES) != 0;

  if ((!fc) || (!handled) || (!energies) ||
      (gpu_traceback && ((!traced) || (!structures))) ||
      (gpu_traceback && copy_matrices))
    return 0;

  try {
    const char *profile_setting = std::getenv("VRNA_CUDA_PROFILE");
    const bool profile = profile_setting && profile_setting[0] &&
                         (std::strcmp(profile_setting, "0") != 0);
    const auto dispatch_start = std::chrono::steady_clock::now();
    std::vector<unsigned char> assigned(count, 0);
    std::vector<unsigned char> can_use_cuda(count, 0);
    std::memset(handled, 0, sizeof(*handled) * count);
    if (traced)
      std::memset(traced, 0, sizeof(*traced) * count);
    parallel_for(count, [&](size_t i) {
      can_use_cuda[i] = eligible(fc[i]) ? 1 : 0;
    });
    const auto eligibility_done = std::chrono::steady_clock::now();
    double bucket_ms = 0.;
    double fold_ms   = 0.;

    for (size_t i = 0; i < count; i++) {
      if (assigned[i] || !can_use_cuda[i])
        continue;

      const auto bucket_start = std::chrono::steady_clock::now();
      std::vector<size_t> bucket;
      bucket.push_back(i);
      assigned[i] = 1;

      for (size_t j = i + 1; j < count; j++) {
        if ((!assigned[j]) && can_use_cuda[j] && same_bucket(fc[i], fc[j])) {
          bucket.push_back(j);
          assigned[j] = 1;
        }
      }
      const auto bucket_done = std::chrono::steady_clock::now();

      const auto fold_start = std::chrono::steady_clock::now();
      (void)fold_bucket(fc,
                        bucket,
                        handled,
                        traced,
                        energies,
                        structures,
                        copy_matrices,
                        gpu_traceback);
      const auto fold_done = std::chrono::steady_clock::now();
      bucket_ms += std::chrono::duration<double, std::milli>(bucket_done - bucket_start).count();
      fold_ms   += std::chrono::duration<double, std::milli>(fold_done - fold_start).count();
    }

    if (profile) {
      const auto dispatch_done = std::chrono::steady_clock::now();
      const double eligibility_ms = std::chrono::duration<double, std::milli>(eligibility_done -
                                                                              dispatch_start).count();
      const double total_ms = std::chrono::duration<double, std::milli>(dispatch_done -
                                                                        dispatch_start).count();
      std::fprintf(stderr,
                   "vrna-cuda dispatch: eligibility=%.3f ms bucketing=%.3f ms "
                   "fold=%.3f ms total=%.3f ms\n",
                   eligibility_ms,
                   bucket_ms,
                   fold_ms,
                   total_ms);
    }
  } catch (const std::bad_alloc &) {
    return 0;
  } catch (...) {
    return 0;
  }

  return 1;
}
