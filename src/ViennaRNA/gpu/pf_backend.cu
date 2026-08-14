#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
#include <type_traits>
#include <vector>
#include <thread>

extern "C" {
#include "ViennaRNA/constraints/hard.h"
#include "ViennaRNA/datastructures/dp_matrices.h"
#include "ViennaRNA/fold_compound.h"
#include "ViennaRNA/gpu/backend.h"
#include "ViennaRNA/model.h"
#include "ViennaRNA/params/basic.h"
#include "ViennaRNA/params/constants.h"
#include "ViennaRNA/partfunc/global.h"
}


namespace {

constexpr unsigned int kBlockSize = 256;
constexpr unsigned int kPfTileSize = 32;
constexpr unsigned int kBlockedLocalThreads = 256;
constexpr size_t kCublasWorkspaceBytes = 32U * 1024U * 1024U;
template <typename Real>
struct GpuPfParams {
  int   max_bp_span;
  int   min_loop_size;
  int   noGU;
  int   noGUclosure;
  int   special_hp;
  int   pair[MAXALPHA + 1][MAXALPHA + 1];
  int   rtype[NBPAIRS + 1];
  Real  expstack[NBPAIRS + 1][NBPAIRS + 1];
  Real  exphairpin[31];
  Real  expbulge[MAXLOOP + 1];
  Real  expinternal[MAXLOOP + 1];
  Real  expmismatchExt[NBPAIRS + 1][5][5];
  Real  expmismatchI[NBPAIRS + 1][5][5];
  Real  expmismatch23I[NBPAIRS + 1][5][5];
  Real  expmismatch1nI[NBPAIRS + 1][5][5];
  Real  expmismatchH[NBPAIRS + 1][5][5];
  Real  expmismatchM[NBPAIRS + 1][5][5];
  Real  expdangle5[NBPAIRS + 1][5];
  Real  expdangle3[NBPAIRS + 1][5];
  Real  expint11[NBPAIRS + 1][NBPAIRS + 1][5][5];
  Real  expint21[NBPAIRS + 1][NBPAIRS + 1][5][5][5];
  Real  expint22[NBPAIRS + 1][NBPAIRS + 1][5][5][5][5];
  Real  expninio[5][MAXLOOP + 1];
  Real  lxc;
  Real  expMLbase;
  Real  expMLintern[NBPAIRS + 1];
  Real  expMLclosing;
  Real  expTermAU;
  Real  exptetra[40];
  Real  exptri[40];
  Real  exphex[40];
  char  Tetraloops[1401];
  char  Triloops[241];
  char  Hexaloops[1801];
  Real  kT;
  Real  pf_scale;
  Real  expSaltStack;
};
template <typename Real>
void
convert_values(Real         *destination,
               const double *source,
               size_t       count)
{
  for (size_t i = 0; i < count; i++)
    destination[i] = static_cast<Real>(source[i]);
}
template <typename Real>
GpuPfParams<Real>
make_gpu_params(const vrna_exp_param_t *source)
{
  GpuPfParams<Real> result{};
  result.max_bp_span   = source->model_details.max_bp_span;
  result.min_loop_size = source->model_details.min_loop_size;
  result.noGU          = source->model_details.noGU;
  result.noGUclosure   = source->model_details.noGUclosure;
  result.special_hp    = source->model_details.special_hp;
  std::memcpy(result.pair, source->model_details.pair, sizeof(result.pair));
  std::memcpy(result.rtype, source->model_details.rtype, sizeof(result.rtype));
#define VRNA_PF_CONVERT(name) \
  convert_values(reinterpret_cast<Real *>(result.name), \
                 reinterpret_cast<const double *>(source->name), \
                 sizeof(source->name) / sizeof(double))
  VRNA_PF_CONVERT(expstack);
  VRNA_PF_CONVERT(exphairpin);
  VRNA_PF_CONVERT(expbulge);
  VRNA_PF_CONVERT(expinternal);
  VRNA_PF_CONVERT(expmismatchExt);
  VRNA_PF_CONVERT(expmismatchI);
  VRNA_PF_CONVERT(expmismatch23I);
  VRNA_PF_CONVERT(expmismatch1nI);
  VRNA_PF_CONVERT(expmismatchH);
  VRNA_PF_CONVERT(expmismatchM);
  VRNA_PF_CONVERT(expdangle5);
  VRNA_PF_CONVERT(expdangle3);
  VRNA_PF_CONVERT(expint11);
  VRNA_PF_CONVERT(expint21);
  VRNA_PF_CONVERT(expint22);
  VRNA_PF_CONVERT(expninio);
  VRNA_PF_CONVERT(expMLintern);
  VRNA_PF_CONVERT(exptetra);
  VRNA_PF_CONVERT(exptri);
  VRNA_PF_CONVERT(exphex);
#undef VRNA_PF_CONVERT
  result.lxc          = static_cast<Real>(source->lxc);
  result.expMLbase    = static_cast<Real>(source->expMLbase);
  result.expMLclosing = static_cast<Real>(source->expMLclosing);
  result.expTermAU    = static_cast<Real>(source->expTermAU);
  result.kT           = static_cast<Real>(source->kT);
  result.pf_scale     = static_cast<Real>(source->pf_scale);
  result.expSaltStack = static_cast<Real>(source->expSaltStack);
  std::memcpy(result.Tetraloops, source->Tetraloops, sizeof(result.Tetraloops));
  std::memcpy(result.Triloops, source->Triloops, sizeof(result.Triloops));
  std::memcpy(result.Hexaloops, source->Hexaloops, sizeof(result.Hexaloops));
  return result;
}


template <typename T>
class DeviceBuffer {
public:
  DeviceBuffer() : data_(nullptr) {}
  ~DeviceBuffer()
  {
    if (data_)
      cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  bool allocate(size_t count)
  {
    return (count == 0) ||
           (cudaMalloc(reinterpret_cast<void **>(&data_),
                       sizeof(T) * count) == cudaSuccess);
  }

  T *get() const
  {
    return data_;
  }

private:
  T *data_;
};


__host__ __device__ inline size_t
triangle_cell(unsigned int span,
              unsigned int i,
              unsigned int n)
{
  return static_cast<size_t>(span) * (2U * n - span + 1U) / 2U + i - 1U;
}


__host__ __device__ inline size_t
pf_index(unsigned int span,
         unsigned int i,
         unsigned int batch,
         unsigned int n,
         unsigned int batch_size)
{
  return triangle_cell(span, i, n) * batch_size + batch;
}


template <typename Real>
__device__ __forceinline__ void
atomic_accumulate(Real *target,
                  Real value)
{
  if (value != Real(0))
    atomicAdd(target, value);
}


template <typename Real>
__device__ __forceinline__ Real
as_real(double value)
{
  return static_cast<Real>(value);
}


template <typename Real>
__device__ __forceinline__ unsigned int
pair_type(const short              *sequence2,
          unsigned int             i,
          unsigned int             j,
          unsigned int             batch,
          unsigned int             n,
          unsigned int             batch_size,
          const GpuPfParams<Real> *params)
{
  if ((i == 0) ||
      (j <= i) ||
      ((j - i) >= static_cast<unsigned int>(params->max_bp_span)) ||
      ((j - i) <= static_cast<unsigned int>(params->min_loop_size)))
    return 0;

  const unsigned int type = params->pair[
    sequence2[static_cast<size_t>(i) * batch_size + batch]
  ][
    sequence2[static_cast<size_t>(j) * batch_size + batch]
  ];

  return (params->noGU && ((type == 3) || (type == 4))) ? 0 : type;
}


template <typename Real>
__device__ __forceinline__ Real
exterior_stem_weight(unsigned int            type,
                     int                     s5,
                     int                     s3,
                     const GpuPfParams<Real> *params)
{
  Real weight = Real(1);
  if ((s5 >= 0) && (s3 >= 0))
    weight = as_real<Real>(params->expmismatchExt[type][s5][s3]);
  else if (s5 >= 0)
    weight = as_real<Real>(params->expdangle5[type][s5]);
  else if (s3 >= 0)
    weight = as_real<Real>(params->expdangle3[type][s3]);

  if (type > 2)
    weight *= as_real<Real>(params->expTermAU);

  return weight;
}


template <typename Real>
__device__ __forceinline__ Real
multibranch_stem_weight(unsigned int            type,
                        int                     s5,
                        int                     s3,
                        const GpuPfParams<Real> *params)
{
  Real weight = as_real<Real>(params->expMLintern[type]);
  if ((s5 >= 0) && (s3 >= 0))
    weight *= as_real<Real>(params->expmismatchM[type][s5][s3]);
  else if (s5 >= 0)
    weight *= as_real<Real>(params->expdangle5[type][s5]);
  else if (s3 >= 0)
    weight *= as_real<Real>(params->expdangle3[type][s3]);

  if (type > 2)
    weight *= as_real<Real>(params->expTermAU);

  return weight;
}




__device__ int
special_loop_index(const char *sequence,
                   const char *table,
                   unsigned int sequence_length,
                   unsigned int stride,
                   unsigned int table_length,
                   unsigned int first,
                   unsigned int batch,
                   unsigned int batch_size)
{
  for (unsigned int offset = 0;
       (offset + sequence_length) < table_length && table[offset] != '\0';
       offset += stride) {
    bool match = true;
    for (unsigned int k = 0; k < sequence_length; k++)
      if (sequence[static_cast<size_t>(first + k) * batch_size + batch] != table[offset + k]) {
        match = false;
        break;
      }

    if (match)
      return static_cast<int>(offset / stride);
  }

  return -1;
}


template <typename Real>
__device__ Real
hairpin_weight(const short              *sequence,
               const char               *characters,
               unsigned int             i,
               unsigned int             j,
               unsigned int             batch,
               unsigned int             batch_size,
               unsigned int             type,
               const Real               *scale,
               const GpuPfParams<Real> *params)
{
  const unsigned int u = j - i - 1;
  if (params->noGUclosure && ((type == 3) || (type == 4)))
    return Real(0);

  Real weight = (u <= 30) ?
                as_real<Real>(params->exphairpin[u]) :
                as_real<Real>(params->exphairpin[30] *
                              exp(-(params->lxc * log(static_cast<double>(u) / 30.)) *
                                  10. / params->kT));

  if ((u >= 3) && params->special_hp) {
    int index = -1;
    if (u == 4)
      index = special_loop_index(characters,
                                 params->Tetraloops,
                                 6,
                                 7,
                                 sizeof(params->Tetraloops),
                                 i,
                                 batch,
                                 batch_size);
    else if (u == 6)
      index = special_loop_index(characters,
                                 params->Hexaloops,
                                 8,
                                 9,
                                 sizeof(params->Hexaloops),
                                 i,
                                 batch,
                                 batch_size);
    else if (u == 3)
      index = special_loop_index(characters,
                                 params->Triloops,
                                 5,
                                 6,
                                 sizeof(params->Triloops),
                                 i,
                                 batch,
                                 batch_size);

    if (index >= 0) {
      if ((u == 4) && (type == 7))
        weight *= as_real<Real>(params->exptetra[index]);
      else if (u == 4)
        weight = as_real<Real>(params->exptetra[index]);
      else if (u == 6)
        weight = as_real<Real>(params->exphex[index]);
      else
        weight = as_real<Real>(params->exptri[index]);

      return weight * scale[j - i + 1];
    }

    if (u == 3) {
      if (type > 2)
        weight *= as_real<Real>(params->expTermAU);
      return weight * scale[j - i + 1];
    }
  }

  if (u >= 3) {
    const int si = sequence[static_cast<size_t>(i + 1) * batch_size + batch];
    const int sj = sequence[static_cast<size_t>(j - 1) * batch_size + batch];
    weight *= as_real<Real>(params->expmismatchH[type][si][sj]);
  }

  return weight * scale[j - i + 1];
}


template <typename Real>
__device__ Real
internal_weight(unsigned int            u1,
                unsigned int            u2,
                unsigned int            type,
                unsigned int            type2,
                int                     si1,
                int                     sj1,
                int                     sp1,
                int                     sq1,
                const Real             *scale,
                const GpuPfParams<Real> *params)
{
  const unsigned int ul = (u1 > u2) ? u1 : u2;
  const unsigned int us = (u1 > u2) ? u2 : u1;
  Real weight;

  if (ul == 0)
    return as_real<Real>(params->expstack[type][type2] * params->expSaltStack) * scale[2];

  if (params->noGUclosure &&
      ((type == 3) || (type == 4) || (type2 == 3) || (type2 == 4)))
    return Real(0);

  switch (us) {
    case 0:
      weight = as_real<Real>(params->expbulge[ul]);
      if (ul == 1) {
        weight *= as_real<Real>(params->expstack[type][type2]);
      } else {
        if (type > 2)
          weight *= as_real<Real>(params->expTermAU);
        if (type2 > 2)
          weight *= as_real<Real>(params->expTermAU);
      }
      break;

    case 1:
      if (ul == 1) {
        weight = as_real<Real>(params->expint11[type][type2][si1][sj1]);
      } else if (ul == 2) {
        weight = (u1 == 1) ?
                 as_real<Real>(params->expint21[type][type2][si1][sq1][sj1]) :
                 as_real<Real>(params->expint21[type2][type][sq1][si1][sp1]);
      } else {
        weight = as_real<Real>(params->expinternal[ul + us] *
                               params->expninio[2][ul - us] *
                               params->expmismatch1nI[type][si1][sj1] *
                               params->expmismatch1nI[type2][sq1][sp1]);
      }
      break;

    case 2:
      if (ul == 2) {
        weight = as_real<Real>(params->expint22[type][type2][si1][sp1][sq1][sj1]);
        break;
      }
      if (ul == 3) {
        weight = as_real<Real>(params->expinternal[5] *
                               params->expninio[2][1] *
                               params->expmismatch23I[type][si1][sj1] *
                               params->expmismatch23I[type2][sq1][sp1]);
        break;
      }
      [[fallthrough]];

    default:
      weight = as_real<Real>(params->expinternal[ul + us] *
                             params->expninio[2][ul - us] *
                             params->expmismatchI[type][si1][sj1] *
                             params->expmismatchI[type2][sq1][sp1]);
      break;
  }

  return weight * scale[u1 + u2 + 2];
}


template <typename Real>
__global__ void
compute_paired_span(Real                    *B,
                    const Real              *M2,
                    const short             *sequence,
                    const short             *sequence2,
                    const char              *characters,
                    const Real              *scale,
                    unsigned int            n,
                    unsigned int            batch_size,
                    unsigned int            span,
                    const GpuPfParams<Real> *params)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  const size_t ij = pf_index(span, i, batch, n, batch_size);
  const unsigned int type = pair_type(sequence2, i, j, batch, n, batch_size, params);
  if (!type) {
    B[ij] = Real(0);
    return;
  }

  Real total = hairpin_weight<Real>(sequence,
                                    characters,
                                    i,
                                    j,
                                    batch,
                                    batch_size,
                                    type,
                                    scale,
                                    params);

  const unsigned int turn = params->min_loop_size;
  const unsigned int p_end = min(j - 1, i + MAXLOOP + 1);
  for (unsigned int p = i + 1; p <= p_end; p++) {
    const unsigned int u1 = p - i - 1;
    const unsigned int q_min_loop = p + turn + 1;
    const unsigned int q_min_size = (j > MAXLOOP - u1 + 1) ?
                                    j - (MAXLOOP - u1) - 1 : p + 1;
    const unsigned int q_begin = max(q_min_loop, q_min_size);
    for (unsigned int q = q_begin; q < j; q++) {
      const unsigned int inner_type = pair_type(sequence2, p, q, batch, n, batch_size, params);
      if (!inner_type)
        continue;

      const unsigned int u2 = j - q - 1;
      const Real enclosed = B[pf_index(q - p, p, batch, n, batch_size)];
      if (enclosed == Real(0))
        continue;

      const unsigned int reverse_type = params->rtype[inner_type];
      const int si1 = sequence[static_cast<size_t>(i + 1) * batch_size + batch];
      const int sj1 = sequence[static_cast<size_t>(j - 1) * batch_size + batch];
      const int sp1 = sequence[static_cast<size_t>(p - 1) * batch_size + batch];
      const int sq1 = sequence[static_cast<size_t>(q + 1) * batch_size + batch];
      total += enclosed * internal_weight<Real>(u1,
                                                u2,
                                                type,
                                                reverse_type,
                                                si1,
                                                sj1,
                                                sp1,
                                                sq1,
                                                scale,
                                                params);
    }
  }

  if ((span >= 2) &&
      !(params->noGUclosure && ((type == 3) || (type == 4)))) {
    const unsigned int reverse_type = params->rtype[type];
    const int s5 = sequence[static_cast<size_t>(j - 1) * batch_size + batch];
    const int s3 = sequence[static_cast<size_t>(i + 1) * batch_size + batch];
    const Real closing = as_real<Real>(params->expMLclosing) *
                         multibranch_stem_weight<Real>(reverse_type, s5, s3, params) *
                         scale[2];
    total += M2[pf_index(span - 2, i + 1, batch, n, batch_size)] * closing;
  }

  B[ij] = total;
}


template <typename Real>
__global__ void
compute_aux_span(const Real              *B,
                 Real                    *E,
                 Real                    *S,
                 Real                    *Q,
                 Real                    *M,
                 Real                    *M2,
                 const short             *sequence,
                 const short             *sequence2,
                 const Real              *scale,
                 const Real              *mlbase,
                 unsigned int            n,
                 unsigned int            batch_size,
                 unsigned int            span,
                 const GpuPfParams<Real> *params)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  const size_t ij = pf_index(span, i, batch, n, batch_size);
  const Real paired = B[ij];
  const Real previous_E = span ? E[pf_index(span - 1, i, batch, n, batch_size)] : Real(0);
  const Real previous_S = span ? S[pf_index(span - 1, i, batch, n, batch_size)] : Real(0);
  const unsigned int type = pair_type(sequence2, i, j, batch, n, batch_size, params);

  Real ext = Real(0);
  Real ml = Real(0);
  if (type && paired != Real(0)) {
    const int s5 = (i > 1) ?
                   sequence[static_cast<size_t>(i - 1) * batch_size + batch] : -1;
    const int s3 = (j < n) ?
                   sequence[static_cast<size_t>(j + 1) * batch_size + batch] : -1;
    ext = paired * exterior_stem_weight<Real>(type, s5, s3, params);
    ml  = paired * multibranch_stem_weight<Real>(type, s5, s3, params);
  }

  const Real e_value = previous_E * scale[1] + ext;
  const Real s_value = previous_S * mlbase[1] + ml;
  E[ij] = e_value;
  S[ij] = s_value;

  Real q = e_value + scale[span + 1];
  Real m2 = Real(0);
  Real m = s_value;
  for (unsigned int k = i + 1; k <= j; k++) {
    const unsigned int left_span = k - i - 1;
    const unsigned int right_span = j - k;
    const Real e = E[pf_index(right_span, k, batch, n, batch_size)];
    const Real s = S[pf_index(right_span, k, batch, n, batch_size)];
    const Real q_left = Q[pf_index(left_span, i, batch, n, batch_size)];
    const Real m_left = M[pf_index(left_span, i, batch, n, batch_size)];
    q += q_left * e;
    m2 += m_left * s;
    m += mlbase[k - i] * s;
  }
  Q[ij] = q;
  M2[ij] = m2;
  M[ij] = m2 + m;
}


template <typename Real>
__global__ void
compute_split_span(const Real *E,
                   const Real *S,
                   Real       *Q,
                   Real       *M,
                   Real       *M2,
                   const Real *scale,
                   const Real *mlbase,
                   unsigned int n,
                   unsigned int batch_size,
                   unsigned int span)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  const size_t ij = pf_index(span, i, batch, n, batch_size);

  Real q = E[ij] + scale[span + 1];
  Real m2 = Real(0);
  Real m = S[ij];

  for (unsigned int k = i + 1; k <= j; k++) {
    const unsigned int left_span = k - i - 1;
    const unsigned int right_span = j - k;
    const Real e = E[pf_index(right_span, k, batch, n, batch_size)];
    const Real s = S[pf_index(right_span, k, batch, n, batch_size)];
    const Real q_left = Q[pf_index(left_span, i, batch, n, batch_size)];
    const Real m_left = M[pf_index(left_span, i, batch, n, batch_size)];
    q += q_left * e;
    m2 += m_left * s;
    m += mlbase[k - i] * s;
  }

  Q[ij] = q;
  M2[ij] = m2;
  M[ij] = m2 + m;
}


template <typename Real>
__global__ void
seed_root(Real         *dQ,
          const Real   *Q,
          unsigned int n,
          unsigned int batch_size)
{
  const unsigned int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < batch_size) {
    const size_t root = pf_index(n - 1, 1, batch, n, batch_size);
    dQ[root] = Real(1) / Q[root];
  }
}


template <typename Real>
__global__ void
reverse_span(const Real              *B,
             const Real              *E,
             const Real              *S,
             const Real              *Q,
             const Real              *M,
             const Real              *M2,
             Real                    *dB,
             Real                    *dE,
             Real                    *dS,
             Real                    *dQ,
             Real                    *dM,
             Real                    *dM2,
             const short             *sequence,
             const short             *sequence2,
             const Real              *scale,
             const Real              *mlbase,
             unsigned int            n,
             unsigned int            batch_size,
             unsigned int            span,
             const GpuPfParams<Real> *params)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  const size_t ij = pf_index(span, i, batch, n, batch_size);

  const Real aq = dQ[ij];
  Real ae = dE[ij] + aq;
  for (unsigned int k = i + 1; k <= j; k++) {
    const unsigned int ls = k - i - 1;
    const unsigned int rs = j - k;
    const size_t left = pf_index(ls, i, batch, n, batch_size);
    const size_t right = pf_index(rs, k, batch, n, batch_size);
    atomic_accumulate(dQ + left, aq * E[right]);
    atomic_accumulate(dE + right, aq * Q[left]);
  }

  const Real am = dM[ij];
  Real am2 = dM2[ij] + am;
  Real as = dS[ij] + am;
  for (unsigned int k = i + 1; k <= j; k++) {
    const unsigned int ls = k - i - 1;
    const unsigned int rs = j - k;
    const size_t left = pf_index(ls, i, batch, n, batch_size);
    const size_t right = pf_index(rs, k, batch, n, batch_size);
    atomic_accumulate(dM + left, am2 * S[right]);
    atomic_accumulate(dS + right, am2 * M[left] + am * mlbase[k - i]);
  }

  Real ab = dB[ij];
  const unsigned int type = pair_type(sequence2, i, j, batch, n, batch_size, params);
  if (type) {
    const int s5 = (i > 1) ?
                   sequence[static_cast<size_t>(i - 1) * batch_size + batch] : -1;
    const int s3 = (j < n) ?
                   sequence[static_cast<size_t>(j + 1) * batch_size + batch] : -1;
    ab += ae * exterior_stem_weight<Real>(type, s5, s3, params);
    ab += as * multibranch_stem_weight<Real>(type, s5, s3, params);
  }

  if (span) {
    atomic_accumulate(dE + pf_index(span - 1, i, batch, n, batch_size), ae * scale[1]);
    atomic_accumulate(dS + pf_index(span - 1, i, batch, n, batch_size), as * mlbase[1]);
  }

  dE[ij] = ae;
  dS[ij] = as;
  dM2[ij] = am2;
  dB[ij] = ab;

  if (!type || (ab == Real(0)))
    return;

  const unsigned int turn = params->min_loop_size;
  const unsigned int p_end = min(j - 1, i + MAXLOOP + 1);
  for (unsigned int p = i + 1; p <= p_end; p++) {
    const unsigned int u1 = p - i - 1;
    const unsigned int q_min_loop = p + turn + 1;
    const unsigned int q_min_size = (j > MAXLOOP - u1 + 1) ?
                                    j - (MAXLOOP - u1) - 1 : p + 1;
    const unsigned int q_begin = max(q_min_loop, q_min_size);
    for (unsigned int q = q_begin; q < j; q++) {
      const unsigned int inner_type = pair_type(sequence2, p, q, batch, n, batch_size, params);
      if (!inner_type)
        continue;

      const unsigned int u2 = j - q - 1;
      const unsigned int reverse_type = params->rtype[inner_type];
      const int si1 = sequence[static_cast<size_t>(i + 1) * batch_size + batch];
      const int sj1 = sequence[static_cast<size_t>(j - 1) * batch_size + batch];
      const int sp1 = sequence[static_cast<size_t>(p - 1) * batch_size + batch];
      const int sq1 = sequence[static_cast<size_t>(q + 1) * batch_size + batch];
      const Real weight = internal_weight<Real>(u1,
                                                u2,
                                                type,
                                                reverse_type,
                                                si1,
                                                sj1,
                                                sp1,
                                                sq1,
                                                scale,
                                                params);
      atomic_accumulate(dB + pf_index(q - p, p, batch, n, batch_size), ab * weight);
    }
  }

  if ((span >= 2) &&
      !(params->noGUclosure && ((type == 3) || (type == 4)))) {
    const unsigned int reverse_type = params->rtype[type];
    const int close5 = sequence[static_cast<size_t>(j - 1) * batch_size + batch];
    const int close3 = sequence[static_cast<size_t>(i + 1) * batch_size + batch];
    const Real closing = as_real<Real>(params->expMLclosing) *
                         multibranch_stem_weight<Real>(reverse_type, close5, close3, params) *
                         scale[2];
    atomic_accumulate(dM2 + pf_index(span - 2, i + 1, batch, n, batch_size),
                      ab * closing);
  }
}


template <typename Real>
__global__ void
gather_roots(const Real   *Q,
             Real         *roots,
             unsigned int n,
             unsigned int batch_size)
{
  const unsigned int batch = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch < batch_size)
    roots[batch] = Q[pf_index(n - 1, 1, batch, n, batch_size)];
}


template <typename Real>
__global__ void
finalize_probabilities(const Real *B,
                       const Real *dB,
                       FLT_OR_DBL *probabilities,
                       size_t      count)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    probabilities[index] = static_cast<FLT_OR_DBL>(B[index] * dB[index]);
}



__host__ __device__ inline size_t
pf_ring_index(unsigned int slot,
              unsigned int i,
              unsigned int batch,
              unsigned int n,
              unsigned int batch_size)
{
  return (static_cast<size_t>(slot) * n + i - 1U) * batch_size + batch;
}


template <typename Real>
__global__ void
compute_pair_aux_reduced_span(Real                    *B,
                              Real                    *S,
                              Real                    *U_ring,
                              const Real              *M2_ring,
                              const short             *sequence,
                              const short             *sequence2,
                              const char              *characters,
                              const Real              *scale,
                              const Real              *mlbase,
                              unsigned int            n,
                              unsigned int            batch_size,
                              unsigned int            span,
                              const GpuPfParams<Real> *params)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  const size_t ij = pf_index(span, i, batch, n, batch_size);
  const unsigned int type = pair_type(sequence2, i, j, batch, n, batch_size, params);
  Real total = Real(0);

  if (type) {
    total = hairpin_weight<Real>(sequence,
                                 characters,
                                 i,
                                 j,
                                 batch,
                                 batch_size,
                                 type,
                                 scale,
                                 params);

    const unsigned int turn = params->min_loop_size;
    const unsigned int p_end = min(j - 1, i + MAXLOOP + 1);
    for (unsigned int p = i + 1; p <= p_end; p++) {
      const unsigned int u1 = p - i - 1;
      const unsigned int q_min_loop = p + turn + 1;
      const unsigned int q_min_size = (j > MAXLOOP - u1 + 1) ?
                                      j - (MAXLOOP - u1) - 1 : p + 1;
      const unsigned int q_begin = max(q_min_loop, q_min_size);
      for (unsigned int q = q_begin; q < j; q++) {
        const unsigned int inner_type = pair_type(sequence2,
                                                  p,
                                                  q,
                                                  batch,
                                                  n,
                                                  batch_size,
                                                  params);
        if (!inner_type)
          continue;

        const Real enclosed = B[pf_index(q - p, p, batch, n, batch_size)];
        if (enclosed == Real(0))
          continue;

        const unsigned int u2 = j - q - 1;
        const unsigned int reverse_type = params->rtype[inner_type];
        const int si1 = sequence[static_cast<size_t>(i + 1) * batch_size + batch];
        const int sj1 = sequence[static_cast<size_t>(j - 1) * batch_size + batch];
        const int sp1 = sequence[static_cast<size_t>(p - 1) * batch_size + batch];
        const int sq1 = sequence[static_cast<size_t>(q + 1) * batch_size + batch];
        total += enclosed * internal_weight<Real>(u1,
                                                  u2,
                                                  type,
                                                  reverse_type,
                                                  si1,
                                                  sj1,
                                                  sp1,
                                                  sq1,
                                                  scale,
                                                  params);
      }
    }

    if ((span >= 2) &&
        !(params->noGUclosure && ((type == 3) || (type == 4)))) {
      const unsigned int reverse_type = params->rtype[type];
      const int s5 = sequence[static_cast<size_t>(j - 1) * batch_size + batch];
      const int s3 = sequence[static_cast<size_t>(i + 1) * batch_size + batch];
      const Real closing = as_real<Real>(params->expMLclosing) *
                           multibranch_stem_weight<Real>(reverse_type, s5, s3, params) *
                           scale[2];
      total += M2_ring[pf_ring_index((span - 2) & 1U,
                                     i + 1,
                                     batch,
                                     n,
                                     batch_size)] * closing;
    }
  }

  B[ij] = total;

  const Real previous_s = span ?
                          S[pf_index(span - 1, i, batch, n, batch_size)] :
                          Real(0);
  Real stem = Real(0);
  if (type && (total != Real(0))) {
    const int s5 = (i > 1) ?
                   sequence[static_cast<size_t>(i - 1) * batch_size + batch] : -1;
    const int s3 = (j < n) ?
                   sequence[static_cast<size_t>(j + 1) * batch_size + batch] : -1;
    stem = total * multibranch_stem_weight<Real>(type, s5, s3, params);
  }

  const Real s_value = previous_s * mlbase[1] + stem;
  const Real previous_u = span ?
                          U_ring[pf_ring_index((span - 1) & 1U,
                                               i + 1,
                                               batch,
                                               n,
                                               batch_size)] :
                          Real(0);
  S[ij] = s_value;
  U_ring[pf_ring_index(span & 1U, i, batch, n, batch_size)] =
    s_value + mlbase[1] * previous_u;
}


template <typename Real>
__global__ void
compute_multibranch_reduced_span(const Real *S,
                                 Real       *M,
                                 const Real *U_ring,
                                 Real       *M2_ring,
                                 unsigned int n,
                                 unsigned int batch_size,
                                 unsigned int span)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  Real m2 = Real(0);
  for (unsigned int k = i + 1; k <= j; k++) {
    const unsigned int left_span = k - i - 1;
    const unsigned int right_span = j - k;
    m2 += M[pf_index(left_span, i, batch, n, batch_size)] *
          S[pf_index(right_span, k, batch, n, batch_size)];
  }

  M2_ring[pf_ring_index(span & 1U, i, batch, n, batch_size)] = m2;
  M[pf_index(span, i, batch, n, batch_size)] =
    m2 + U_ring[pf_ring_index(span & 1U, i, batch, n, batch_size)];
}


template <typename Real>
__global__ void
compute_exterior_vectors(const Real              *B,
                         Real                    *q5,
                         Real                    *q3,
                         Real                    *roots,
                         const short             *sequence,
                         const short             *sequence2,
                         const Real              *scale,
                         unsigned int            n,
                         unsigned int            batch_size,
                         const GpuPfParams<Real> *params)
{
  const unsigned int batch = blockIdx.x;
  const unsigned int lane = threadIdx.x;
  __shared__ Real partial[kBlockSize];

  if (batch >= batch_size)
    return;

  if (lane == 0) {
    q5[batch] = Real(1);
    q3[static_cast<size_t>(n + 1) * batch_size + batch] = Real(1);
  }
  __syncthreads();

  for (unsigned int j = 1; j <= n; j++) {
    Real value = Real(0);
    for (unsigned int i = lane + 1; i < j; i += blockDim.x) {
      const unsigned int type = pair_type(sequence2,
                                          i,
                                          j,
                                          batch,
                                          n,
                                          batch_size,
                                          params);
      if (type) {
        const int s5 = (i > 1) ?
                       sequence[static_cast<size_t>(i - 1) * batch_size + batch] : -1;
        const int s3 = (j < n) ?
                       sequence[static_cast<size_t>(j + 1) * batch_size + batch] : -1;
        value += q5[static_cast<size_t>(i - 1) * batch_size + batch] *
                 B[pf_index(j - i, i, batch, n, batch_size)] *
                 exterior_stem_weight<Real>(type, s5, s3, params);
      }
    }
    partial[lane] = value;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (lane < offset)
        partial[lane] += partial[lane + offset];
      __syncthreads();
    }
    if (lane == 0)
      q5[static_cast<size_t>(j) * batch_size + batch] =
        q5[static_cast<size_t>(j - 1) * batch_size + batch] * scale[1] +
        partial[0];
    __syncthreads();
  }

  for (unsigned int reverse_i = n; reverse_i > 0; reverse_i--) {
    const unsigned int i = reverse_i;
    Real value = Real(0);
    for (unsigned int j = i + lane + 1; j <= n; j += blockDim.x) {
      const unsigned int type = pair_type(sequence2,
                                          i,
                                          j,
                                          batch,
                                          n,
                                          batch_size,
                                          params);
      if (type) {
        const int s5 = (i > 1) ?
                       sequence[static_cast<size_t>(i - 1) * batch_size + batch] : -1;
        const int s3 = (j < n) ?
                       sequence[static_cast<size_t>(j + 1) * batch_size + batch] : -1;
        value += B[pf_index(j - i, i, batch, n, batch_size)] *
                 exterior_stem_weight<Real>(type, s5, s3, params) *
                 q3[static_cast<size_t>(j + 1) * batch_size + batch];
      }
    }
    partial[lane] = value;
    __syncthreads();
    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
      if (lane < offset)
        partial[lane] += partial[lane + offset];
      __syncthreads();
    }
    if (lane == 0)
      q3[static_cast<size_t>(i) * batch_size + batch] =
        q3[static_cast<size_t>(i + 1) * batch_size + batch] * scale[1] +
        partial[0];
    __syncthreads();
  }

  if (lane == 0)
    roots[batch] = q5[static_cast<size_t>(n) * batch_size + batch];
}


template <typename Real>
__global__ void
seed_exterior_adjoint(Real                    *dB,
                      const Real              *q5,
                      const Real              *q3,
                      const Real              *roots,
                      const short             *sequence,
                      const short             *sequence2,
                      unsigned int            n,
                      unsigned int            batch_size,
                      unsigned int            span,
                      const GpuPfParams<Real> *params)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  Real seed = Real(0);
  const unsigned int type = pair_type(sequence2, i, j, batch, n, batch_size, params);
  if (type && (roots[batch] > Real(0))) {
    const int s5 = (i > 1) ?
                   sequence[static_cast<size_t>(i - 1) * batch_size + batch] : -1;
    const int s3 = (j < n) ?
                   sequence[static_cast<size_t>(j + 1) * batch_size + batch] : -1;
    seed = q5[static_cast<size_t>(i - 1) * batch_size + batch] *
           q3[static_cast<size_t>(j + 1) * batch_size + batch] *
           exterior_stem_weight<Real>(type, s5, s3, params) /
           roots[batch];
  }
  dB[pf_index(span, i, batch, n, batch_size)] = seed;
}


template <typename Real>
__global__ void
reverse_reduced_span(const Real              *B,
                     const Real              *S,
                     const Real              *M,
                     const Real              *q5,
                     const Real              *q3,
                     const Real              *roots,
                     Real                    *dB,
                     Real                    *dS,
                     Real                    *dM,
                     Real                    *dU_ring,
                     Real                    *dM2_ring,
                     const short             *sequence,
                     const short             *sequence2,
                     const Real              *scale,
                     const Real              *mlbase,
                     unsigned int            n,
                     unsigned int            batch_size,
                     unsigned int            span,
                     const GpuPfParams<Real> *params)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  const size_t ij = pf_index(span, i, batch, n, batch_size);

  const Real am = dM[ij];
  const size_t m2_slot = pf_ring_index(span % 3U, i, batch, n, batch_size);
  const Real am2 = dM2_ring[m2_slot] + am;
  dM2_ring[m2_slot] = Real(0);

  Real au = am;
  if (i > 1)
    au += mlbase[1] *
          dU_ring[pf_ring_index((span + 1) & 1U,
                                i - 1,
                                batch,
                                n,
                                batch_size)];

  for (unsigned int k = i + 1; k <= j; k++) {
    const unsigned int left_span = k - i - 1;
    const unsigned int right_span = j - k;
    const size_t left = pf_index(left_span, i, batch, n, batch_size);
    const size_t right = pf_index(right_span, k, batch, n, batch_size);
    atomic_accumulate(dM + left, am2 * S[right]);
    atomic_accumulate(dS + right, am2 * M[left]);
  }

  const Real as = dS[ij] + au;
  Real ab = dB[ij];
  const unsigned int type = pair_type(sequence2, i, j, batch, n, batch_size, params);
  if (type) {
    const int s5 = (i > 1) ?
                   sequence[static_cast<size_t>(i - 1) * batch_size + batch] : -1;
    const int s3 = (j < n) ?
                   sequence[static_cast<size_t>(j + 1) * batch_size + batch] : -1;
    if (roots[batch] > Real(0))
      ab += q5[static_cast<size_t>(i - 1) * batch_size + batch] *
            q3[static_cast<size_t>(j + 1) * batch_size + batch] *
            exterior_stem_weight<Real>(type, s5, s3, params) / roots[batch];
    ab += as * multibranch_stem_weight<Real>(type, s5, s3, params);
  }

  if (span)
    atomic_accumulate(dS + pf_index(span - 1, i, batch, n, batch_size),
                      as * mlbase[1]);

  dU_ring[pf_ring_index(span & 1U, i, batch, n, batch_size)] = au;
  dS[ij] = as;
  dB[ij] = ab;

  if (!type || (ab == Real(0)))
    return;

  const unsigned int turn = params->min_loop_size;
  const unsigned int p_end = min(j - 1, i + MAXLOOP + 1);
  for (unsigned int p = i + 1; p <= p_end; p++) {
    const unsigned int u1 = p - i - 1;
    const unsigned int q_min_loop = p + turn + 1;
    const unsigned int q_min_size = (j > MAXLOOP - u1 + 1) ?
                                    j - (MAXLOOP - u1) - 1 : p + 1;
    const unsigned int q_begin = max(q_min_loop, q_min_size);
    for (unsigned int q = q_begin; q < j; q++) {
      const unsigned int inner_type = pair_type(sequence2,
                                                p,
                                                q,
                                                batch,
                                                n,
                                                batch_size,
                                                params);
      if (!inner_type)
        continue;

      const unsigned int u2 = j - q - 1;
      const unsigned int reverse_type = params->rtype[inner_type];
      const int si1 = sequence[static_cast<size_t>(i + 1) * batch_size + batch];
      const int sj1 = sequence[static_cast<size_t>(j - 1) * batch_size + batch];
      const int sp1 = sequence[static_cast<size_t>(p - 1) * batch_size + batch];
      const int sq1 = sequence[static_cast<size_t>(q + 1) * batch_size + batch];
      const Real weight = internal_weight<Real>(u1,
                                                u2,
                                                type,
                                                reverse_type,
                                                si1,
                                                sj1,
                                                sp1,
                                                sq1,
                                                scale,
                                                params);
      atomic_accumulate(dB + pf_index(q - p, p, batch, n, batch_size),
                        ab * weight);
    }
  }

  if ((span >= 2) &&
      !(params->noGUclosure && ((type == 3) || (type == 4)))) {
    const unsigned int reverse_type = params->rtype[type];
    const int close5 = sequence[static_cast<size_t>(j - 1) * batch_size + batch];
    const int close3 = sequence[static_cast<size_t>(i + 1) * batch_size + batch];
    const Real closing = as_real<Real>(params->expMLclosing) *
                         multibranch_stem_weight<Real>(reverse_type,
                                                       close5,
                                                       close3,
                                                       params) *
                         scale[2];
    atomic_accumulate(dM2_ring +
                      pf_ring_index((span - 2) % 3U,
                                    i + 1,
                                    batch,
                                    n,
                                    batch_size),
                      ab * closing);
  }
}

template <typename Real>
__device__ __forceinline__ Real
multibranch_source_adjoint(const Real              *dB,
                           const Real              *dM,
                           const short             *sequence,
                           const short             *sequence2,
                           const Real              *scale,
                           unsigned int            i,
                           unsigned int            j,
                           unsigned int            batch,
                           unsigned int            n,
                           unsigned int            batch_size,
                           const GpuPfParams<Real> *params)
{
  Real value = dM[pf_index(j - i, i, batch, n, batch_size)];
  if ((i > 1) && (j < n)) {
    const unsigned int outer_i = i - 1;
    const unsigned int outer_j = j + 1;
    const unsigned int outer_type = pair_type(sequence2,
                                              outer_i,
                                              outer_j,
                                              batch,
                                              n,
                                              batch_size,
                                              params);
    if (outer_type &&
        !(params->noGUclosure &&
          ((outer_type == 3) || (outer_type == 4)))) {
      const unsigned int reverse_type = params->rtype[outer_type];
      const int close5 =
        sequence[static_cast<size_t>(outer_j - 1) * batch_size + batch];
      const int close3 =
        sequence[static_cast<size_t>(outer_i + 1) * batch_size + batch];
      const Real closing =
        as_real<Real>(params->expMLclosing) *
        multibranch_stem_weight<Real>(reverse_type,
                                      close5,
                                      close3,
                                      params) *
        scale[2];
      value += dB[pf_index(outer_j - outer_i,
                           outer_i,
                           batch,
                           n,
                           batch_size)] * closing;
    }
  }

  return value;
}


template <typename Real>
__global__ void
gather_dM_span(const Real              *S,
               const Real              *dB,
               Real                    *dM,
               const short             *sequence,
               const short             *sequence2,
               const Real              *scale,
               unsigned int            n,
               unsigned int            batch_size,
               unsigned int            span,
               const GpuPfParams<Real> *params)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  Real value = Real(0);

  for (unsigned int source_j = j + 1; source_j <= n; source_j++) {
    const Real source =
      multibranch_source_adjoint<Real>(dB,
                                       dM,
                                       sequence,
                                       sequence2,
                                       scale,
                                       i,
                                       source_j,
                                       batch,
                                       n,
                                       batch_size,
                                       params);
    value += source *
             S[pf_index(source_j - j - 1,
                        j + 1,
                        batch,
                        n,
                        batch_size)];
  }

  dM[pf_index(span, i, batch, n, batch_size)] = value;
}


template <typename Real>
__global__ void
gather_dS_span(const Real              *M,
               const Real              *dB,
               const Real              *dM,
               Real                    *dS,
               Real                    *dU_ring,
               const short             *sequence,
               const short             *sequence2,
               const Real              *scale,
               const Real              *mlbase,
               unsigned int            n,
               unsigned int            batch_size,
               unsigned int            span,
               const GpuPfParams<Real> *params)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  const size_t ij = pf_index(span, i, batch, n, batch_size);

  Real au = dM[ij];
  if (i > 1)
    au += mlbase[1] *
          dU_ring[pf_ring_index((span + 1) & 1U,
                                i - 1,
                                batch,
                                n,
                                batch_size)];

  Real value = au;
  if (j < n)
    value += mlbase[1] *
             dS[pf_index(span + 1, i, batch, n, batch_size)];

  for (unsigned int source_i = 1; source_i < i; source_i++) {
    const Real source =
      multibranch_source_adjoint<Real>(dB,
                                       dM,
                                       sequence,
                                       sequence2,
                                       scale,
                                       source_i,
                                       j,
                                       batch,
                                       n,
                                       batch_size,
                                       params);
    value += source *
             M[pf_index(i - source_i - 1,
                        source_i,
                        batch,
                        n,
                        batch_size)];
  }

  dU_ring[pf_ring_index(span & 1U, i, batch, n, batch_size)] = au;
  dS[ij] = value;
}


template <typename Real>
__global__ void
gather_dB_span(const Real              *q5,
               const Real              *q3,
               const Real              *roots,
               const Real              *dS,
               Real                    *dB,
               const short             *sequence,
               const Real              *B,
               FLT_OR_DBL              *probabilities,
               Real                    *paired_sums,
               int                     *valid_flags,
               size_t                  output_stride,
               const short             *sequence2,
               const Real              *scale,
               unsigned int            n,
               unsigned int            batch_size,
               unsigned int            span,
               const GpuPfParams<Real> *params)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int cells = (n - span) * batch_size;
  if (tid >= cells)
    return;

  const unsigned int batch = tid % batch_size;
  const unsigned int i = tid / batch_size + 1;
  const unsigned int j = i + span;
  const size_t ij = pf_index(span, i, batch, n, batch_size);
  const unsigned int type = pair_type(sequence2,
                                      i,
                                      j,
                                      batch,
                                      n,
                                      batch_size,
                                      params);
  Real value = Real(0);

  if (type) {
    const int s5 = (i > 1) ?
                   sequence[static_cast<size_t>(i - 1) * batch_size + batch] : -1;
    const int s3 = (j < n) ?
                   sequence[static_cast<size_t>(j + 1) * batch_size + batch] : -1;
    if (roots[batch] > Real(0))
      value += q5[static_cast<size_t>(i - 1) * batch_size + batch] *
               q3[static_cast<size_t>(j + 1) * batch_size + batch] *
               exterior_stem_weight<Real>(type, s5, s3, params) /
               roots[batch];

    value += dS[ij] *
             multibranch_stem_weight<Real>(type, s5, s3, params);

    for (unsigned int u1 = 0; u1 <= MAXLOOP; u1++) {
      if (i <= u1 + 1)
        break;
      const unsigned int outer_i = i - u1 - 1;
      for (unsigned int u2 = 0; u1 + u2 <= MAXLOOP; u2++) {
        const unsigned int outer_j = j + u2 + 1;
        if (outer_j > n)
          break;

        const unsigned int outer_type = pair_type(sequence2,
                                                  outer_i,
                                                  outer_j,
                                                  batch,
                                                  n,
                                                  batch_size,
                                                  params);
        if (!outer_type)
          continue;

        const unsigned int reverse_type = params->rtype[type];
        const int si1 =
          sequence[static_cast<size_t>(outer_i + 1) * batch_size + batch];
        const int sj1 =
          sequence[static_cast<size_t>(outer_j - 1) * batch_size + batch];
        const int sp1 =
          sequence[static_cast<size_t>(i - 1) * batch_size + batch];
        const int sq1 =
          sequence[static_cast<size_t>(j + 1) * batch_size + batch];
        const Real weight = internal_weight<Real>(u1,
                                                  u2,
                                                  outer_type,
                                                  reverse_type,
                                                  si1,
                                                  sj1,
                                                  sp1,
                                                  sq1,
                                                  scale,
                                                  params);
        value += dB[pf_index(outer_j - outer_i,
                             outer_i,
                             batch,
                             n,
                             batch_size)] * weight;
      }
    }
  }

  dB[ij] = value;
  if (probabilities) {
    const Real probability = B[ij] * value;
    if ((!isfinite(probability)) ||
        (probability < Real(-5.e-5)) ||
        (probability > Real(1.0005)))
      atomicExch(valid_flags + batch, 0);

    if (i != j) {
      atomic_accumulate(paired_sums +
                        static_cast<size_t>(i) * batch_size + batch,
                        probability);
      atomic_accumulate(paired_sums +
                        static_cast<size_t>(j) * batch_size + batch,
                        probability);
      const size_t iindx =
        static_cast<size_t>(n + 1 - i) * (n - i) / 2 + n + 1;
      probabilities[static_cast<size_t>(batch) * output_stride + iindx - j] =
        static_cast<FLT_OR_DBL>(probability);
    }
  }
}


template <typename Real>
__global__ void
validate_paired_sums(const Real   *paired_sums,
                     int          *valid_flags,
                     unsigned int n,
                     unsigned int batch_size)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t count = static_cast<size_t>(n + 1) * batch_size;
  if (index >= count)
    return;

  const unsigned int batch = index % batch_size;
  const Real paired_sum = paired_sums[index];
  if ((!isfinite(paired_sum)) ||
      (paired_sum < Real(-5.e-5)) ||
      (paired_sum > Real(1.0005)))
    atomicExch(valid_flags + batch, 0);
}


__host__ __device__ inline size_t
blocked_cell(unsigned int batch,
             unsigned int i,
             unsigned int j,
             unsigned int matrix_dim)
{
  return (static_cast<size_t>(batch) * matrix_dim + i) * matrix_dim + j;
}


__host__ __device__ inline size_t
blocked_batch_cell(unsigned int i,
                   unsigned int j,
                   unsigned int batch,
                   unsigned int matrix_dim,
                   unsigned int batch_size)
{
  return (static_cast<size_t>(i) * matrix_dim + j) * batch_size + batch;
}


template <typename Real>
__global__ void
build_blocked_pair_metadata(const short             *sequence2,
                            unsigned char           *types,
                            unsigned int            *pair_masks,
                            const uint2             *tile_coordinates,
                            unsigned int            tile_count,
                            unsigned int            matrix_dim,
                            unsigned int            n,
                            unsigned int            batch_size,
                            const GpuPfParams<Real> *params)
{
  const unsigned int batch = blockIdx.x % batch_size;
  const unsigned int tile = blockIdx.x / batch_size;
  if (tile >= tile_count)
    return;

  const uint2 coordinate = tile_coordinates[tile];
  const unsigned int i0 = 1U + coordinate.x * kPfTileSize;
  const unsigned int j0 = 1U + coordinate.y * kPfTileSize;
  __shared__ unsigned int masks[kPfTileSize];

  if (threadIdx.x < kPfTileSize)
    masks[threadIdx.x] = 0;
  __syncthreads();

  for (unsigned int local = threadIdx.x;
       local < kPfTileSize * kPfTileSize;
       local += blockDim.x) {
    const unsigned int row = local / kPfTileSize;
    const unsigned int column = local % kPfTileSize;
    const unsigned int i = i0 + row;
    const unsigned int j = j0 + column;
    unsigned int type = 0;
    if ((i <= n) && (j <= n) && (j >= i))
      type = pair_type(sequence2, i, j, batch, n, batch_size, params);
    types[blocked_cell(batch, i, j, matrix_dim)] =
      static_cast<unsigned char>(type);
    if (type)
      atomicOr(masks + row, 1U << column);
  }
  __syncthreads();

  if (threadIdx.x < kPfTileSize)
    pair_masks[(static_cast<size_t>(batch) * tile_count + tile) *
               kPfTileSize + threadIdx.x] = masks[threadIdx.x];
}


template <typename Real, unsigned int G>
__global__ void
forward_local_blocked_diagonal(Real                    *B,
                               Real                    *S,
                               Real                    *U,
                               Real                    *C,
                               Real                    *M,
                               const unsigned char     *types,
                               const short             *sequence,
                               const char              *characters,
                               const Real              *scale,
                               const Real              *mlbase,
                               unsigned int            n,
                               unsigned int            matrix_dim,
                               unsigned int            batch_size,
                               unsigned int            tile_diagonal,
                               unsigned int            tile_count,
                               bool                    use_far,
                               const GpuPfParams<Real> *params)
{
  static_assert(G == 32, "the blocked PF kernels currently require G=32");
  const unsigned int batch = blockIdx.x % batch_size;
  const unsigned int tile_i = blockIdx.x / batch_size;
  if (tile_i + tile_diagonal >= tile_count)
    return;

  const unsigned int lane = threadIdx.x & 7U;
  const unsigned int group = threadIdx.x >> 3U;
  const unsigned int i0 = 1U + tile_i * G;
  const unsigned int j0 = 1U + (tile_i + tile_diagonal) * G;
  const unsigned int row_end = min(n, i0 + G - 1U);
  const unsigned int column_end = min(n, j0 + G - 1U);
  constexpr unsigned int halo_width = G + MAXLOOP + 1U;
  const int halo_i0 = static_cast<int>(i0) + 1;
  const int halo_j0 = static_cast<int>(j0) - MAXLOOP - 1;
  __shared__ unsigned char halo_types[halo_width * halo_width];
  __shared__ unsigned long long halo_pair_masks[halo_width];
  __shared__ unsigned char compact_rows[G];
  __shared__ unsigned int compact_count;
  if (threadIdx.x < halo_width)
    halo_pair_masks[threadIdx.x] = 0;
  __syncthreads();
  for (unsigned int cell = threadIdx.x;
       cell < halo_width * halo_width;
       cell += blockDim.x) {
    const int p = halo_i0 + static_cast<int>(cell / halo_width);
    const int q = halo_j0 + static_cast<int>(cell % halo_width);
    if ((p >= 1) && (q >= p) && (q <= static_cast<int>(n))) {
      const size_t source = blocked_cell(batch,
                                         static_cast<unsigned int>(p),
                                         static_cast<unsigned int>(q),
                                         matrix_dim);
      halo_types[cell] = types[source];
      if (halo_types[cell])
        atomicOr(halo_pair_masks + cell / halo_width,
                 1ULL << (cell % halo_width));
    } else {
      halo_types[cell] = 0;
    }
  }
  __syncthreads();
  const int minimum_span = (tile_diagonal == 0) ?
                           0 : static_cast<int>(tile_diagonal * G) -
                               static_cast<int>(G - 1U);
  const int maximum_span = min(static_cast<int>(n - 1U),
                               static_cast<int>(tile_diagonal * G + G - 1U));

  for (int span = minimum_span; span <= maximum_span; span++) {
    if (threadIdx.x < G) {
      const unsigned int row = threadIdx.x;
      const unsigned int candidate_i = i0 + row;
      const int candidate_j = static_cast<int>(candidate_i) + span;
      const bool pairable = (candidate_i <= row_end) &&
                            (candidate_j >= static_cast<int>(j0)) &&
                            (candidate_j <= static_cast<int>(column_end)) &&
                            (types[blocked_cell(batch,
                                                candidate_i,
                                                static_cast<unsigned int>(candidate_j),
                                                matrix_dim)] != 0);
      const unsigned int mask = __ballot_sync(0xffffffffU, pairable);
      if (pairable) {
        const unsigned int before = (row == 0) ? 0U : (1U << row) - 1U;
        compact_rows[__popc(mask & before)] = static_cast<unsigned char>(row);
      }
      if (row == 0)
        compact_count = __popc(mask);
    }
    __syncthreads();

    const bool pair_active = group < compact_count;
    const unsigned int pair_i = pair_active ? i0 + compact_rows[group] : 0U;
    const unsigned int pair_j = pair_active ? pair_i + span : 0U;
    const unsigned int pair_type = pair_active ?
      types[blocked_cell(batch, pair_i, pair_j, matrix_dim)] : 0U;
    Real internal = Real(0);

    if (pair_active) {
      for (unsigned int u1 = lane; u1 <= MAXLOOP; u1 += 8U) {
        const unsigned int p = pair_i + u1 + 1U;
        const int q_min = max(static_cast<int>(p + params->min_loop_size + 1U),
                              static_cast<int>(pair_j) -
                                static_cast<int>(MAXLOOP - u1) - 1);
        const int q_max = static_cast<int>(pair_j) - 1;
        if (q_min <= q_max) {
          const int first_column = max(0, q_min - halo_j0);
          const int last_column = min(static_cast<int>(halo_width) - 1,
                                      q_max - halo_j0);
          if (first_column <= last_column) {
            const unsigned long long through_last =
              (1ULL << (last_column + 1)) - 1ULL;
            const unsigned long long before_first =
              (first_column == 0) ? 0ULL : (1ULL << first_column) - 1ULL;
            unsigned long long candidates =
              halo_pair_masks[p - static_cast<unsigned int>(halo_i0)] &
              through_last & ~before_first;
            while (candidates) {
              const unsigned int halo_q =
                static_cast<unsigned int>(__ffsll(candidates) - 1);
              candidates &= candidates - 1ULL;
              const unsigned int q =
                static_cast<unsigned int>(halo_j0 + static_cast<int>(halo_q));
              const unsigned int u2 = pair_j - q - 1U;
              const unsigned int halo_p = p - static_cast<unsigned int>(halo_i0);
              const size_t halo_cell =
                static_cast<size_t>(halo_p) * halo_width + halo_q;
              const unsigned int inner_type = halo_types[halo_cell];
              const Real enclosed = B[blocked_cell(batch, p, q, matrix_dim)];
              if (enclosed == Real(0))
                continue;
              const unsigned int reverse_type = params->rtype[inner_type];
              const int si1 = sequence[static_cast<size_t>(pair_i + 1U) *
                                       batch_size + batch];
              const int sj1 = sequence[static_cast<size_t>(pair_j - 1U) *
                                       batch_size + batch];
              const int sp1 = sequence[static_cast<size_t>(p - 1U) *
                                       batch_size + batch];
              const int sq1 = sequence[static_cast<size_t>(q + 1U) *
                                       batch_size + batch];
              internal += enclosed * internal_weight<Real>(u1,
                                                            u2,
                                                            pair_type,
                                                            reverse_type,
                                                            si1,
                                                            sj1,
                                                            sp1,
                                                            sq1,
                                                            scale,
                                                            params);
            }
          }
        }
      }
    }

    internal += __shfl_down_sync(0xffffffffU, internal, 4, 8);
    internal += __shfl_down_sync(0xffffffffU, internal, 2, 8);
    internal += __shfl_down_sync(0xffffffffU, internal, 1, 8);

    if (pair_active && (lane == 0)) {
      Real total = hairpin_weight<Real>(sequence,
                                        characters,
                                        pair_i,
                                        pair_j,
                                        batch,
                                        batch_size,
                                        pair_type,
                                        scale,
                                        params) + internal;
      if (((pair_j - pair_i) >= 2U) &&
          !(params->noGUclosure &&
            ((pair_type == 3) || (pair_type == 4)))) {
        const unsigned int reverse_type = params->rtype[pair_type];
        const int s5 = sequence[static_cast<size_t>(pair_j - 1U) *
                                batch_size + batch];
        const int s3 = sequence[static_cast<size_t>(pair_i + 1U) *
                                batch_size + batch];
        const Real closing = as_real<Real>(params->expMLclosing) *
                             multibranch_stem_weight<Real>(reverse_type,
                                                           s5,
                                                           s3,
                                                           params) *
                             scale[2];
        total += C[blocked_cell(batch,
                                pair_i + 1U,
                                pair_j - 1U,
                                matrix_dim)] * closing;
      }
      B[blocked_cell(batch, pair_i, pair_j, matrix_dim)] = total;
    }
    __syncthreads();

    const unsigned int i = i0 + group;
    const int signed_j = static_cast<int>(i) + span;
    const bool active = (i <= row_end) &&
                        (signed_j >= static_cast<int>(j0)) &&
                        (signed_j <= static_cast<int>(column_end));
    const unsigned int j = active ? static_cast<unsigned int>(signed_j) : 0U;
    const unsigned int type = active ?
      types[blocked_cell(batch, i, j, matrix_dim)] : 0U;
    if (active && (lane == 0)) {
      const size_t ij = blocked_cell(batch, i, j, matrix_dim);
      const Real previous_s = (j > i) ?
        S[blocked_cell(batch, i, j - 1U, matrix_dim)] : Real(0);
      Real stem = Real(0);
      if (type && (B[ij] != Real(0))) {
        const int s5 = (i > 1U) ?
          sequence[static_cast<size_t>(i - 1U) * batch_size + batch] : -1;
        const int s3 = (j < n) ?
          sequence[static_cast<size_t>(j + 1U) * batch_size + batch] : -1;
        stem = B[ij] * multibranch_stem_weight<Real>(type, s5, s3, params);
      }
      const Real s_value = previous_s * mlbase[1] + stem;
      const Real previous_u = (j > i) ?
        U[blocked_cell(batch, i + 1U, j, matrix_dim)] : Real(0);
      S[ij] = s_value;
      U[ij] = s_value + mlbase[1] * previous_u;
    }
    __syncthreads();

    Real boundary = Real(0);
    if (active) {
      for (unsigned int k = i + 1U + lane; k <= j; k += 8U) {
        if (use_far && (k > row_end) && (k < j0))
          continue;
        boundary += M[blocked_cell(batch, i, k - 1U, matrix_dim)] *
                    S[blocked_cell(batch, k, j, matrix_dim)];
      }
    }
    boundary += __shfl_down_sync(0xffffffffU, boundary, 4, 8);
    boundary += __shfl_down_sync(0xffffffffU, boundary, 2, 8);
    boundary += __shfl_down_sync(0xffffffffU, boundary, 1, 8);

    if (active && (lane == 0)) {
      const size_t ij = blocked_cell(batch, i, j, matrix_dim);
      const Real m2 = (use_far ? C[ij] : Real(0)) + boundary;
      C[ij] = m2;
      M[ij] = U[ij] + m2;
    }
    __syncthreads();
  }
}


template <typename Real, unsigned int G>
__global__ void
forward_local_blocked_span(Real                    *B,
                           Real                    *S,
                           Real                    *U,
                           Real                    *C,
                           Real                    *M,
                           const unsigned char     *types,
                           const short             *sequence,
                           const char              *characters,
                           const Real              *scale,
                           const Real              *mlbase,
                           unsigned int            n,
                           unsigned int            matrix_dim,
                           unsigned int            batch_size,
                           unsigned int            batch_chunks,
                           unsigned int            span,
                           unsigned int            tile_diagonal,
                           unsigned int            tile_count,
                           unsigned int            local_begin,
                           unsigned int            local_count,
                           bool                    use_far,
                           const GpuPfParams<Real> *params)
{
  static_assert(G == 32, "the vectorized blocked PF kernel requires G=32");
  const unsigned int batch_chunk = blockIdx.x % batch_chunks;
  const unsigned int cell = blockIdx.x / batch_chunks;
  const unsigned int local_offset = cell % local_count;
  const unsigned int tile_i = cell / local_count;
  const unsigned int batch_lane = threadIdx.x & 31U;
  const unsigned int candidate_lane = threadIdx.x >> 5U;
  const unsigned int batch = batch_chunk * 32U + batch_lane;
  const unsigned int i0 = 1U + tile_i * G;
  const unsigned int j0 = 1U + (tile_i + tile_diagonal) * G;
  const unsigned int i = i0 + local_begin + local_offset;
  const unsigned int j = i + span;
  const unsigned int row_end = min(n, i0 + G - 1U);
  const bool active = (batch < batch_size) && (i <= row_end) && (j <= n) &&
                      (j >= j0) && (j < j0 + G);
  const unsigned int type = active ?
    types[blocked_batch_cell(i, j, batch, matrix_dim, batch_size)] : 0U;
  __shared__ Real partial[8][32];
  Real internal = Real(0);

  if (active && type) {
    for (unsigned int u1 = 0; u1 <= MAXLOOP; u1++) {
      for (unsigned int u2 = candidate_lane;
           u1 + u2 <= MAXLOOP;
           u2 += 8U) {
        const unsigned int p = i + u1 + 1U;
        if (j <= u2 + 1U)
          continue;
        const unsigned int q = j - u2 - 1U;
        if ((p >= q) ||
            ((q - p) <= static_cast<unsigned int>(params->min_loop_size)))
          continue;
        const unsigned int inner_type =
          types[blocked_batch_cell(p, q, batch, matrix_dim, batch_size)];
        if (!inner_type)
          continue;
        const Real enclosed =
          B[blocked_batch_cell(p, q, batch, matrix_dim, batch_size)];
        if (enclosed == Real(0))
          continue;
        const unsigned int reverse_type = params->rtype[inner_type];
        const int si1 = sequence[static_cast<size_t>(i + 1U) * batch_size + batch];
        const int sj1 = sequence[static_cast<size_t>(j - 1U) * batch_size + batch];
        const int sp1 = sequence[static_cast<size_t>(p - 1U) * batch_size + batch];
        const int sq1 = sequence[static_cast<size_t>(q + 1U) * batch_size + batch];
        internal += enclosed * internal_weight<Real>(u1,
                                                      u2,
                                                      type,
                                                      reverse_type,
                                                      si1,
                                                      sj1,
                                                      sp1,
                                                      sq1,
                                                      scale,
                                                      params);
      }
    }
  }
  partial[candidate_lane][batch_lane] = internal;
  __syncthreads();

  if (candidate_lane == 0) {
    const size_t bij = blocked_batch_cell(i, j, batch, matrix_dim, batch_size);
    const size_t ij = blocked_cell(batch, i, j, matrix_dim);
    Real total = Real(0);
    if (active && type) {
      for (unsigned int source = 0; source < 8U; source++)
        total += partial[source][batch_lane];
      total += hairpin_weight<Real>(sequence,
                                    characters,
                                    i,
                                    j,
                                    batch,
                                    batch_size,
                                    type,
                                    scale,
                                    params);
      if ((span >= 2U) &&
          !(params->noGUclosure && ((type == 3) || (type == 4)))) {
        const unsigned int reverse_type = params->rtype[type];
        const int s5 = sequence[static_cast<size_t>(j - 1U) * batch_size + batch];
        const int s3 = sequence[static_cast<size_t>(i + 1U) * batch_size + batch];
        const Real closing = as_real<Real>(params->expMLclosing) *
                             multibranch_stem_weight<Real>(reverse_type,
                                                           s5,
                                                           s3,
                                                           params) *
                             scale[2];
        total += C[blocked_cell(batch, i + 1U, j - 1U, matrix_dim)] * closing;
      }
    }
    if (active)
      B[bij] = total;
    if (active) {
      const Real previous_s = span ?
        S[blocked_cell(batch, i, j - 1U, matrix_dim)] : Real(0);
      Real stem = Real(0);
      if (type && (total != Real(0))) {
        const int s5 = (i > 1U) ?
          sequence[static_cast<size_t>(i - 1U) * batch_size + batch] : -1;
        const int s3 = (j < n) ?
          sequence[static_cast<size_t>(j + 1U) * batch_size + batch] : -1;
        stem = total * multibranch_stem_weight<Real>(type, s5, s3, params);
      }
      const Real s_value = previous_s * mlbase[1] + stem;
      const Real previous_u = span ?
        U[blocked_cell(batch, i + 1U, j, matrix_dim)] : Real(0);
      S[ij] = s_value;
      U[ij] = s_value + mlbase[1] * previous_u;
    }
  }
  __syncthreads();

  Real boundary = Real(0);
  if (active) {
    for (unsigned int k = i + 1U + candidate_lane; k <= j; k += 8U) {
      if (use_far && (k > row_end) && (k < j0))
        continue;
      boundary += M[blocked_cell(batch, i, k - 1U, matrix_dim)] *
                  S[blocked_cell(batch, k, j, matrix_dim)];
    }
  }
  partial[candidate_lane][batch_lane] = boundary;
  __syncthreads();

  if ((candidate_lane == 0) && active) {
    const size_t ij = blocked_cell(batch, i, j, matrix_dim);
    Real m2 = use_far ? C[ij] : Real(0);
    for (unsigned int source = 0; source < 8U; source++)
      m2 += partial[source][batch_lane];
    C[ij] = m2;
    M[ij] = U[ij] + m2;
  }
}


template <typename Real, unsigned int G>
__global__ void
compute_exterior_blocked(const Real              *B,
                         Real                    *q5,
                         Real                    *q3,
                         Real                    *roots,
                         const unsigned char     *types,
                         const short             *sequence,
                         const Real              *scale,
                         unsigned int            n,
                         unsigned int            matrix_dim,
                         unsigned int            tile_count,
                         unsigned int            batch_size,
                         const GpuPfParams<Real> *params)
{
  static_assert(G == 32, "the exterior blocked kernel uses one warp per panel cell");
  const unsigned int batch = blockIdx.x;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  __shared__ Real far_contribution[G];

  if (batch >= batch_size)
    return;
  if (threadIdx.x == 0) {
    q5[batch] = Real(1);
    q3[static_cast<size_t>(n + 1U) * batch_size + batch] = Real(1);
  }
  __syncthreads();

  for (unsigned int panel_start = 1U; panel_start <= n; panel_start += G) {
    const unsigned int panel_end = min(n, panel_start + G - 1U);
    const unsigned int j = panel_start + warp;
    Real value = Real(0);
    if (j <= panel_end) {
      for (unsigned int i = lane + 1U; i < panel_start; i += 32U) {
        const unsigned int type = types[blocked_cell(batch, i, j, matrix_dim)];
        if (type) {
          const int s5 = (i > 1U) ?
            sequence[static_cast<size_t>(i - 1U) * batch_size + batch] : -1;
          const int s3 = (j < n) ?
            sequence[static_cast<size_t>(j + 1U) * batch_size + batch] : -1;
          value += q5[static_cast<size_t>(i - 1U) * batch_size + batch] *
                   B[blocked_cell(batch, i, j, matrix_dim)] *
                   exterior_stem_weight<Real>(type, s5, s3, params);
        }
      }
    }
    for (unsigned int offset = 16U; offset > 0; offset >>= 1U)
      value += __shfl_down_sync(0xffffffffU, value, offset);
    if ((lane == 0) && (j <= panel_end))
      far_contribution[warp] = value;
    __syncthreads();

    if (warp == 0) {
      for (unsigned int current_j = panel_start;
           current_j <= panel_end;
           current_j++) {
        Real local = Real(0);
        for (unsigned int i = panel_start + lane; i < current_j; i += 32U) {
          const unsigned int type = types[blocked_cell(batch, i, current_j, matrix_dim)];
          if (type) {
            const int s5 = (i > 1U) ?
              sequence[static_cast<size_t>(i - 1U) * batch_size + batch] : -1;
            const int s3 = (current_j < n) ?
              sequence[static_cast<size_t>(current_j + 1U) * batch_size + batch] : -1;
            local += q5[static_cast<size_t>(i - 1U) * batch_size + batch] *
                     B[blocked_cell(batch, i, current_j, matrix_dim)] *
                     exterior_stem_weight<Real>(type, s5, s3, params);
          }
        }
        for (unsigned int offset = 16U; offset > 0; offset >>= 1U)
          local += __shfl_down_sync(0xffffffffU, local, offset);
        if (lane == 0)
          q5[static_cast<size_t>(current_j) * batch_size + batch] =
            q5[static_cast<size_t>(current_j - 1U) * batch_size + batch] * scale[1] +
            far_contribution[current_j - panel_start] + local;
        __syncwarp();
      }
    }
    __syncthreads();
  }

  unsigned int panel_end = n;
  while (panel_end > 0) {
    const unsigned int panel_start = (panel_end >= G) ? panel_end - G + 1U : 1U;
    const unsigned int i = panel_start + warp;
    Real value = Real(0);
    if (i <= panel_end) {
      for (unsigned int j = panel_end + 1U + lane; j <= n; j += 32U) {
        const unsigned int type = types[blocked_cell(batch, i, j, matrix_dim)];
        if (type) {
          const int s5 = (i > 1U) ?
            sequence[static_cast<size_t>(i - 1U) * batch_size + batch] : -1;
          const int s3 = (j < n) ?
            sequence[static_cast<size_t>(j + 1U) * batch_size + batch] : -1;
          value += B[blocked_cell(batch, i, j, matrix_dim)] *
                   exterior_stem_weight<Real>(type, s5, s3, params) *
                   q3[static_cast<size_t>(j + 1U) * batch_size + batch];
        }
      }
    }
    for (unsigned int offset = 16U; offset > 0; offset >>= 1U)
      value += __shfl_down_sync(0xffffffffU, value, offset);
    if ((lane == 0) && (i <= panel_end))
      far_contribution[warp] = value;
    __syncthreads();

    if (warp == 0) {
      for (unsigned int reverse_i = panel_end + 1U; reverse_i-- > panel_start;) {
        const unsigned int current_i = reverse_i;
        Real local = Real(0);
        for (unsigned int j = current_i + 1U + lane; j <= panel_end; j += 32U) {
          const unsigned int type = types[blocked_cell(batch, current_i, j, matrix_dim)];
          if (type) {
            const int s5 = (current_i > 1U) ?
              sequence[static_cast<size_t>(current_i - 1U) * batch_size + batch] : -1;
            const int s3 = (j < n) ?
              sequence[static_cast<size_t>(j + 1U) * batch_size + batch] : -1;
            local += B[blocked_cell(batch, current_i, j, matrix_dim)] *
                     exterior_stem_weight<Real>(type, s5, s3, params) *
                     q3[static_cast<size_t>(j + 1U) * batch_size + batch];
          }
        }
        for (unsigned int offset = 16U; offset > 0; offset >>= 1U)
          local += __shfl_down_sync(0xffffffffU, local, offset);
        if (lane == 0)
          q3[static_cast<size_t>(current_i) * batch_size + batch] =
            q3[static_cast<size_t>(current_i + 1U) * batch_size + batch] * scale[1] +
            far_contribution[current_i - panel_start] + local;
        __syncwarp();
      }
    }
    __syncthreads();
    if (panel_start == 1U)
      break;
    panel_end = panel_start - 1U;
  }

  if (threadIdx.x == 0)
    roots[batch] = q5[static_cast<size_t>(n) * batch_size + batch];
}


template <typename Real, unsigned int G>
__global__ void
reverse_local_blocked_diagonal(const Real              *B,
                               const Real              *S,
                               const Real              *M,
                               Real                    *dB,
                               Real                    *dS,
                               Real                    *dM,
                               Real                    *dU,
                               Real                    *A,
                               const Real              *q5,
                               const Real              *q3,
                               const Real              *roots,
                               FLT_OR_DBL              *probabilities,
                               const unsigned char     *types,
                               const short             *sequence,
                               const Real              *scale,
                               const Real              *mlbase,
                               size_t                  output_stride,
                               unsigned int            n,
                               unsigned int            matrix_dim,
                               unsigned int            batch_size,
                               unsigned int            tile_diagonal,
                               unsigned int            tile_count,
                               const GpuPfParams<Real> *params)
{
  static_assert(G == 32, "the blocked PF kernels currently require G=32");
  const unsigned int batch = blockIdx.x % batch_size;
  const unsigned int tile_i = blockIdx.x / batch_size;
  if (tile_i + tile_diagonal >= tile_count)
    return;

  const unsigned int lane = threadIdx.x & 7U;
  const unsigned int group = threadIdx.x >> 3U;
  const unsigned int i0 = 1U + tile_i * G;
  const unsigned int j0 = 1U + (tile_i + tile_diagonal) * G;
  const unsigned int row_end = min(n, i0 + G - 1U);
  const unsigned int column_end = min(n, j0 + G - 1U);
  constexpr unsigned int halo_width = G + MAXLOOP + 1U;
  const int halo_i0 = static_cast<int>(i0) - MAXLOOP - 1;
  const int halo_j0 = static_cast<int>(j0) + 1;
  __shared__ unsigned char halo_types[halo_width * halo_width];
  __shared__ unsigned long long halo_pair_masks[halo_width];
  __shared__ unsigned char compact_rows[G];
  __shared__ unsigned int compact_count;
  if (threadIdx.x < halo_width)
    halo_pair_masks[threadIdx.x] = 0;
  __syncthreads();
  for (unsigned int cell = threadIdx.x;
       cell < halo_width * halo_width;
       cell += blockDim.x) {
    const int p = halo_i0 + static_cast<int>(cell / halo_width);
    const int q = halo_j0 + static_cast<int>(cell % halo_width);
    if ((p >= 1) && (q >= p) && (q <= static_cast<int>(n))) {
      const size_t source = blocked_cell(batch,
                                         static_cast<unsigned int>(p),
                                         static_cast<unsigned int>(q),
                                         matrix_dim);
      halo_types[cell] = types[source];
      if (halo_types[cell])
        atomicOr(halo_pair_masks + cell / halo_width,
                 1ULL << (cell % halo_width));
    } else {
      halo_types[cell] = 0;
    }
  }
  __syncthreads();
  const int minimum_span = (tile_diagonal == 0) ?
                           0 : static_cast<int>(tile_diagonal * G) -
                               static_cast<int>(G - 1U);
  const int maximum_span = min(static_cast<int>(n - 1U),
                               static_cast<int>(tile_diagonal * G + G - 1U));

  for (int span = maximum_span; span >= minimum_span; span--) {
    const unsigned int i = i0 + group;
    const int signed_j = static_cast<int>(i) + span;
    const bool active = (i <= row_end) &&
                        (signed_j >= static_cast<int>(j0)) &&
                        (signed_j <= static_cast<int>(column_end));
    const unsigned int j = active ? static_cast<unsigned int>(signed_j) : 0U;
    Real local_m = Real(0);
    if (active) {
      for (unsigned int source_j = j + 1U + lane;
           source_j <= column_end;
           source_j += 8U)
        local_m += A[blocked_cell(batch, i, source_j, matrix_dim)] *
                   S[blocked_cell(batch, j + 1U, source_j, matrix_dim)];
    }
    local_m += __shfl_down_sync(0xffffffffU, local_m, 4, 8);
    local_m += __shfl_down_sync(0xffffffffU, local_m, 2, 8);
    local_m += __shfl_down_sync(0xffffffffU, local_m, 1, 8);

    Real am = Real(0);
    Real au = Real(0);
    if (active && (lane == 0)) {
      const size_t ij = blocked_cell(batch, i, j, matrix_dim);
      am = dM[ij] + local_m;
      dM[ij] = am;
      au = am;
      if (i > 1U)
        au += mlbase[1] * dU[blocked_cell(batch, i - 1U, j, matrix_dim)];
      dU[ij] = au;
    }
    am = __shfl_sync(0xffffffffU, am, 0, 8);
    au = __shfl_sync(0xffffffffU, au, 0, 8);

    Real local_s = Real(0);
    if (active) {
      for (unsigned int source_i = i0 + lane; source_i < i; source_i += 8U)
        local_s += A[blocked_cell(batch, source_i, j, matrix_dim)] *
                   M[blocked_cell(batch, source_i, i - 1U, matrix_dim)];
    }
    local_s += __shfl_down_sync(0xffffffffU, local_s, 4, 8);
    local_s += __shfl_down_sync(0xffffffffU, local_s, 2, 8);
    local_s += __shfl_down_sync(0xffffffffU, local_s, 1, 8);

    Real as = Real(0);
    if (active && (lane == 0)) {
      const size_t ij = blocked_cell(batch, i, j, matrix_dim);
      as = dS[ij] + local_s + au;
      if (j < n)
        as += mlbase[1] * dS[blocked_cell(batch, i, j + 1U, matrix_dim)];
      dS[ij] = as;
    }
    __syncthreads();
    if (threadIdx.x < G) {
      const unsigned int row = threadIdx.x;
      const unsigned int candidate_i = i0 + row;
      const int candidate_j = static_cast<int>(candidate_i) + span;
      const bool pairable = (candidate_i <= row_end) &&
                            (candidate_j >= static_cast<int>(j0)) &&
                            (candidate_j <= static_cast<int>(column_end)) &&
                            (types[blocked_cell(batch,
                                                candidate_i,
                                                static_cast<unsigned int>(candidate_j),
                                                matrix_dim)] != 0);
      const unsigned int mask = __ballot_sync(0xffffffffU, pairable);
      if (pairable) {
        const unsigned int before = (row == 0) ? 0U : (1U << row) - 1U;
        compact_rows[__popc(mask & before)] = static_cast<unsigned char>(row);
      }
      if (row == 0)
        compact_count = __popc(mask);
    }
    __syncthreads();

    const bool pair_active = group < compact_count;
    const unsigned int pair_i = pair_active ? i0 + compact_rows[group] : 0U;
    const unsigned int pair_j = pair_active ? pair_i + span : 0U;
    const unsigned int pair_type = pair_active ?
      types[blocked_cell(batch, pair_i, pair_j, matrix_dim)] : 0U;
    Real outer = Real(0);
    if (pair_active) {
      for (unsigned int u1 = lane; u1 <= MAXLOOP; u1 += 8U) {
        if (pair_i > u1 + 1U) {
          const unsigned int outer_i = pair_i - u1 - 1U;
          const unsigned int halo_i =
            static_cast<unsigned int>(static_cast<int>(outer_i) - halo_i0);
          const int first_column = max(0,
                                       static_cast<int>(pair_j + 1U) - halo_j0);
          const int last_column =
            min(static_cast<int>(halo_width) - 1,
                static_cast<int>(min(n,
                                     pair_j + (MAXLOOP - u1) + 1U)) - halo_j0);
          if (first_column <= last_column) {
            const unsigned long long through_last =
              (1ULL << (last_column + 1)) - 1ULL;
            const unsigned long long before_first =
              (first_column == 0) ? 0ULL : (1ULL << first_column) - 1ULL;
            unsigned long long candidates = halo_pair_masks[halo_i] &
                                             through_last & ~before_first;
            while (candidates) {
              const unsigned int halo_j =
                static_cast<unsigned int>(__ffsll(candidates) - 1);
              candidates &= candidates - 1ULL;
              const unsigned int outer_j =
                static_cast<unsigned int>(halo_j0 + static_cast<int>(halo_j));
              const unsigned int u2 = outer_j - pair_j - 1U;
              const size_t halo_cell =
                static_cast<size_t>(halo_i) * halo_width + halo_j;
              const unsigned int outer_type = halo_types[halo_cell];
              const unsigned int reverse_type = params->rtype[pair_type];
              const int si1 = sequence[static_cast<size_t>(outer_i + 1U) *
                                       batch_size + batch];
              const int sj1 = sequence[static_cast<size_t>(outer_j - 1U) *
                                       batch_size + batch];
              const int sp1 = sequence[static_cast<size_t>(pair_i - 1U) *
                                       batch_size + batch];
              const int sq1 = sequence[static_cast<size_t>(pair_j + 1U) *
                                       batch_size + batch];
              outer += dB[blocked_cell(batch,
                                       outer_i,
                                       outer_j,
                                       matrix_dim)] *
                       internal_weight<Real>(u1,
                                             u2,
                                             outer_type,
                                             reverse_type,
                                             si1,
                                             sj1,
                                             sp1,
                                             sq1,
                                             scale,
                                             params);
            }
          }
        }
      }
    }
    outer += __shfl_down_sync(0xffffffffU, outer, 4, 8);
    outer += __shfl_down_sync(0xffffffffU, outer, 2, 8);
    outer += __shfl_down_sync(0xffffffffU, outer, 1, 8);

    if (pair_active && (lane == 0)) {
      const size_t ij = blocked_cell(batch, pair_i, pair_j, matrix_dim);
      const int s5 = (pair_i > 1U) ?
        sequence[static_cast<size_t>(pair_i - 1U) * batch_size + batch] : -1;
      const int s3 = (pair_j < n) ?
        sequence[static_cast<size_t>(pair_j + 1U) * batch_size + batch] : -1;
      Real ab = outer +
                dS[ij] * multibranch_stem_weight<Real>(pair_type,
                                                        s5,
                                                        s3,
                                                        params);
      if (roots[batch] > Real(0))
        ab += q5[static_cast<size_t>(pair_i - 1U) * batch_size + batch] *
              q3[static_cast<size_t>(pair_j + 1U) * batch_size + batch] *
              exterior_stem_weight<Real>(pair_type, s5, s3, params) /
              roots[batch];
      dB[ij] = ab;
      if (pair_i != pair_j) {
        const size_t iindx =
          static_cast<size_t>(n + 1U - pair_i) * (n - pair_i) / 2U + n + 1U;
        probabilities[static_cast<size_t>(batch) * output_stride +
                      iindx - pair_j] =
          static_cast<FLT_OR_DBL>(B[ij] * ab);
      }
    }
    __syncthreads();

    if (active && (lane == 0)) {
      const size_t ij = blocked_cell(batch, i, j, matrix_dim);
      Real source = am;
      if ((i > 1U) && (j < n)) {
        const unsigned int outer_i = i - 1U;
        const unsigned int outer_j = j + 1U;
        const unsigned int outer_type =
          types[blocked_cell(batch, outer_i, outer_j, matrix_dim)];
        if (outer_type &&
            !(params->noGUclosure && ((outer_type == 3) || (outer_type == 4)))) {
          const unsigned int reverse_type = params->rtype[outer_type];
          const int close5 = sequence[static_cast<size_t>(outer_j - 1U) *
                                      batch_size + batch];
          const int close3 = sequence[static_cast<size_t>(outer_i + 1U) *
                                      batch_size + batch];
          const Real closing = as_real<Real>(params->expMLclosing) *
                               multibranch_stem_weight<Real>(reverse_type,
                                                             close5,
                                                             close3,
                                                             params) *
                               scale[2];
          source += dB[blocked_cell(batch, outer_i, outer_j, matrix_dim)] *
                    closing;
        }
      }
      A[ij] = source;
    }
    __syncthreads();
  }
}


template <typename Real, unsigned int G>
__global__ void
reverse_local_blocked_span(const Real              *B,
                           const Real              *S,
                           const Real              *M,
                           Real                    *dB,
                           Real                    *dS,
                           Real                    *dM,
                           Real                    *dU,
                           Real                    *A,
                           const Real              *q5,
                           const Real              *q3,
                           const Real              *roots,
                           FLT_OR_DBL              *probabilities,
                           const unsigned char     *types,
                           const short             *sequence,
                           const Real              *scale,
                           const Real              *mlbase,
                           size_t                  output_stride,
                           unsigned int            n,
                           unsigned int            matrix_dim,
                           unsigned int            batch_size,
                           unsigned int            batch_chunks,
                           unsigned int            span,
                           unsigned int            tile_diagonal,
                           unsigned int            tile_count,
                           unsigned int            local_begin,
                           unsigned int            local_count,
                           const GpuPfParams<Real> *params)
{
  static_assert(G == 32, "the vectorized blocked PF kernel requires G=32");
  const unsigned int batch_chunk = blockIdx.x % batch_chunks;
  const unsigned int cell = blockIdx.x / batch_chunks;
  const unsigned int local_offset = cell % local_count;
  const unsigned int tile_i = cell / local_count;
  const unsigned int batch_lane = threadIdx.x & 31U;
  const unsigned int candidate_lane = threadIdx.x >> 5U;
  const unsigned int batch = batch_chunk * 32U + batch_lane;
  const unsigned int i0 = 1U + tile_i * G;
  const unsigned int j0 = 1U + (tile_i + tile_diagonal) * G;
  const unsigned int i = i0 + local_begin + local_offset;
  const unsigned int j = i + span;
  const unsigned int row_end = min(n, i0 + G - 1U);
  const unsigned int column_end = min(n, j0 + G - 1U);
  const bool active = (batch < batch_size) && (i <= row_end) && (j <= n) &&
                      (j >= j0) && (j < j0 + G);
  __shared__ Real partial[8][32];
  __shared__ Real cell_am[32];
  __shared__ Real cell_au[32];
  __shared__ Real cell_as[32];

  Real local_m = Real(0);
  if (active) {
    for (unsigned int source_j = j + 1U + candidate_lane;
         source_j <= column_end;
         source_j += 8U)
      local_m += A[blocked_cell(batch, i, source_j, matrix_dim)] *
                 S[blocked_cell(batch, j + 1U, source_j, matrix_dim)];
  }
  partial[candidate_lane][batch_lane] = local_m;
  __syncthreads();
  if ((candidate_lane == 0) && active) {
    const size_t ij = blocked_cell(batch, i, j, matrix_dim);
    Real am = dM[ij];
    for (unsigned int source = 0; source < 8U; source++)
      am += partial[source][batch_lane];
    dM[ij] = am;
    Real au = am;
    if (i > 1U)
      au += mlbase[1] * dU[blocked_cell(batch, i - 1U, j, matrix_dim)];
    dU[ij] = au;
    cell_am[batch_lane] = am;
    cell_au[batch_lane] = au;
  }
  __syncthreads();

  Real local_s = Real(0);
  if (active) {
    for (unsigned int source_i = i0 + candidate_lane;
         source_i < i;
         source_i += 8U)
      local_s += A[blocked_cell(batch, source_i, j, matrix_dim)] *
                 M[blocked_cell(batch, source_i, i - 1U, matrix_dim)];
  }
  partial[candidate_lane][batch_lane] = local_s;
  __syncthreads();
  if ((candidate_lane == 0) && active) {
    const size_t ij = blocked_cell(batch, i, j, matrix_dim);
    Real as = dS[ij] + cell_au[batch_lane];
    for (unsigned int source = 0; source < 8U; source++)
      as += partial[source][batch_lane];
    if (j < n)
      as += mlbase[1] * dS[blocked_cell(batch, i, j + 1U, matrix_dim)];
    dS[ij] = as;
    cell_as[batch_lane] = as;
  }
  __syncthreads();

  const unsigned int type = active ?
    types[blocked_batch_cell(i, j, batch, matrix_dim, batch_size)] : 0U;
  Real outer = Real(0);
  if (active && type) {
    for (unsigned int u1 = 0; u1 <= MAXLOOP; u1++) {
      for (unsigned int u2 = candidate_lane;
           u1 + u2 <= MAXLOOP;
           u2 += 8U) {
        if ((i <= u1 + 1U) || (j + u2 + 1U > n))
          continue;
        const unsigned int outer_i = i - u1 - 1U;
        const unsigned int outer_j = j + u2 + 1U;
        const unsigned int outer_type =
          types[blocked_batch_cell(outer_i,
                                   outer_j,
                                   batch,
                                   matrix_dim,
                                   batch_size)];
        if (!outer_type)
          continue;
        const unsigned int reverse_type = params->rtype[type];
        const int si1 = sequence[static_cast<size_t>(outer_i + 1U) * batch_size + batch];
        const int sj1 = sequence[static_cast<size_t>(outer_j - 1U) * batch_size + batch];
        const int sp1 = sequence[static_cast<size_t>(i - 1U) * batch_size + batch];
        const int sq1 = sequence[static_cast<size_t>(j + 1U) * batch_size + batch];
        outer += dB[blocked_batch_cell(outer_i,
                                       outer_j,
                                       batch,
                                       matrix_dim,
                                       batch_size)] *
                 internal_weight<Real>(u1,
                                       u2,
                                       outer_type,
                                       reverse_type,
                                       si1,
                                       sj1,
                                       sp1,
                                       sq1,
                                       scale,
                                       params);
      }
    }
  }
  partial[candidate_lane][batch_lane] = outer;
  __syncthreads();

  if ((candidate_lane == 0) && active) {
    const size_t ij = blocked_cell(batch, i, j, matrix_dim);
    const size_t bij = blocked_batch_cell(i, j, batch, matrix_dim, batch_size);
    Real ab = Real(0);
    if (type) {
      const int s5 = (i > 1U) ?
        sequence[static_cast<size_t>(i - 1U) * batch_size + batch] : -1;
      const int s3 = (j < n) ?
        sequence[static_cast<size_t>(j + 1U) * batch_size + batch] : -1;
      if (roots[batch] > Real(0))
        ab += q5[static_cast<size_t>(i - 1U) * batch_size + batch] *
              q3[static_cast<size_t>(j + 1U) * batch_size + batch] *
              exterior_stem_weight<Real>(type, s5, s3, params) /
              roots[batch];
      ab += cell_as[batch_lane] *
            multibranch_stem_weight<Real>(type, s5, s3, params);
      for (unsigned int source = 0; source < 8U; source++)
        ab += partial[source][batch_lane];
    }
    dB[bij] = ab;
    if (i != j) {
      const size_t iindx =
        static_cast<size_t>(n + 1U - i) * (n - i) / 2U + n + 1U;
      probabilities[static_cast<size_t>(batch) * output_stride + iindx - j] =
        static_cast<FLT_OR_DBL>(B[bij] * ab);
    }

    Real source = cell_am[batch_lane];
    if ((i > 1U) && (j < n)) {
      const unsigned int outer_i = i - 1U;
      const unsigned int outer_j = j + 1U;
      const unsigned int outer_type =
        types[blocked_batch_cell(outer_i,
                                 outer_j,
                                 batch,
                                 matrix_dim,
                                 batch_size)];
      if (outer_type &&
          !(params->noGUclosure && ((outer_type == 3) || (outer_type == 4)))) {
        const unsigned int reverse_type = params->rtype[outer_type];
        const int close5 = sequence[static_cast<size_t>(outer_j - 1U) *
                                    batch_size + batch];
        const int close3 = sequence[static_cast<size_t>(outer_i + 1U) *
                                    batch_size + batch];
        const Real closing = as_real<Real>(params->expMLclosing) *
                             multibranch_stem_weight<Real>(reverse_type,
                                                           close5,
                                                           close3,
                                                           params) *
                             scale[2];
        source += dB[blocked_batch_cell(outer_i,
                                        outer_j,
                                        batch,
                                        matrix_dim,
                                        batch_size)] * closing;
      }
    }
    A[ij] = source;
  }
}


template <typename Real>
__global__ void
validate_blocked_probabilities(const Real   *B,
                               const Real   *dB,
                               int          *valid_flags,
                               unsigned int n,
                               unsigned int matrix_dim,
                               unsigned int batch_size)
{
  const unsigned int base = blockIdx.x % n + 1U;
  const unsigned int batch = blockIdx.x / n;
  if (batch >= batch_size)
    return;

  __shared__ Real partial[kBlockSize];
  __shared__ int invalid[kBlockSize];
  Real sum = Real(0);
  int bad = 0;
  for (unsigned int partner = threadIdx.x + 1U;
       partner <= n;
       partner += blockDim.x) {
    if (partner == base)
      continue;
    const unsigned int i = min(base, partner);
    const unsigned int j = max(base, partner);
    const size_t ij = blocked_cell(batch, i, j, matrix_dim);
    const Real probability = B[ij] * dB[ij];
    if ((!isfinite(probability)) ||
        (probability < Real(-5.e-5)) ||
        (probability > Real(1.0005)))
      bad = 1;
    sum += probability;
  }
  partial[threadIdx.x] = sum;
  invalid[threadIdx.x] = bad;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2U; offset > 0; offset >>= 1U) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
      invalid[threadIdx.x] |= invalid[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if ((threadIdx.x == 0) &&
      (invalid[0] || (!isfinite(partial[0])) ||
       (partial[0] < Real(-5.e-5)) || (partial[0] > Real(1.0005))))
    valid_flags[batch] = 0;
}


enum class BlockedGemmMode {
  native,
  emulated
};


template <typename Real>
struct CublasType;

template <>
struct CublasType<float> {
  static constexpr cudaDataType_t data_type = CUDA_R_32F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
};

template <>
struct CublasType<double> {
  static constexpr cudaDataType_t data_type = CUDA_R_64F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_64F;
};


template <typename Real, unsigned int G>
class PersistentBlockedPlan {
public:
  PersistentBlockedPlan(unsigned int   n,
                        unsigned int   batch_size,
                        bool           with_bpp,
                        int            device,
                        BlockedGemmMode gemm_mode)
    : n_(n),
      batch_size_(batch_size),
      with_bpp_(with_bpp),
      device_(device),
      gemm_mode_(gemm_mode),
      tile_count_((n + G - 1U) / G),
      matrix_dim_(tile_count_ * G + 1U),
      matrix_stride_(static_cast<size_t>(matrix_dim_) * matrix_dim_),
      matrix_count_(matrix_stride_ * batch_size_),
      sequence_count_(static_cast<size_t>(n + 2U) * batch_size_),
      output_stride_(static_cast<size_t>(n + 1U) * (n + 2U) / 2U),
      output_count_(output_stride_ * batch_size_),
      packed_tile_count_(static_cast<size_t>(tile_count_) *
                         (tile_count_ + 1U) / 2U),
      gemm_crossover_(read_gemm_crossover()),
      stream_(nullptr),
      cublas_(nullptr),
      host_sequence_(nullptr),
      host_sequence2_(nullptr),
      host_characters_(nullptr),
      host_scale_(nullptr),
      host_mlbase_(nullptr),
      host_params_(nullptr),
      host_roots_(nullptr),
      host_probabilities_(nullptr),
      host_valid_flags_(nullptr),
      profile_enabled_(profile_requested()),
      ready_(false)
  {
    static_assert(G == kPfTileSize, "metadata and local tiles must have the same size");
    ready_ = initialize();
  }

  ~PersistentBlockedPlan()
  {
    if (stream_)
      (void)cudaStreamSynchronize(stream_);
    for (cudaGraphExec_t executable : graph_execs_)
      if (executable)
        (void)cudaGraphExecDestroy(executable);
    for (cudaGraph_t graph : graphs_)
      if (graph)
        (void)cudaGraphDestroy(graph);
    if (cublas_)
      (void)cublasDestroy(cublas_);
    if (stream_)
      (void)cudaStreamDestroy(stream_);
    if (host_sequence_)
      (void)cudaFreeHost(host_sequence_);
    if (host_sequence2_)
      (void)cudaFreeHost(host_sequence2_);
    if (host_characters_)
      (void)cudaFreeHost(host_characters_);
    if (host_scale_)
      (void)cudaFreeHost(host_scale_);
    if (host_mlbase_)
      (void)cudaFreeHost(host_mlbase_);
    if (host_params_)
      (void)cudaFreeHost(host_params_);
    if (host_roots_)
      (void)cudaFreeHost(host_roots_);
    if (host_probabilities_)
      (void)cudaFreeHost(host_probabilities_);
    if (host_valid_flags_)
      (void)cudaFreeHost(host_valid_flags_);
    for (cudaEvent_t event : phase_events_)
      if (event)
        (void)cudaEventDestroy(event);
    destroy_events(forward_far_begin_);
    destroy_events(forward_far_end_);
    destroy_events(forward_local_begin_);
    destroy_events(forward_local_end_);
    destroy_events(reverse_dM_begin_);
    destroy_events(reverse_dM_end_);
    destroy_events(reverse_dS_begin_);
    destroy_events(reverse_dS_end_);
    destroy_events(reverse_local_begin_);
    destroy_events(reverse_local_end_);
  }

  PersistentBlockedPlan(const PersistentBlockedPlan &) = delete;
  PersistentBlockedPlan &operator=(const PersistentBlockedPlan &) = delete;

  bool matches(unsigned int   n,
               unsigned int   batch_size,
               bool           with_bpp,
               int            device,
               BlockedGemmMode gemm_mode) const
  {
    return (n_ == n) &&
           (batch_size_ == batch_size) &&
           (with_bpp_ == with_bpp) &&
           (device_ == device) &&
           (gemm_mode_ == gemm_mode);
  }

  bool ready() const
  {
    return ready_;
  }

  bool execute(vrna_fold_compound_t      **fc,
               const std::vector<size_t> &bucket)
  {
    if ((!ready_) || (bucket.size() != batch_size_))
      return false;

    int current_device = -1;
    if ((cudaGetDevice(&current_device) != cudaSuccess) ||
        (current_device != device_))
      return false;

    for (unsigned int b = 0; b < batch_size_; b++) {
      const vrna_fold_compound_t *current = fc[bucket[b]];
      for (unsigned int position = 0; position <= n_ + 1U; position++) {
        host_sequence_[static_cast<size_t>(position) * batch_size_ + b] =
          current->sequence_encoding[position];
        host_sequence2_[static_cast<size_t>(position) * batch_size_ + b] =
          current->sequence_encoding2[position];
        host_characters_[static_cast<size_t>(position) * batch_size_ + b] =
          ((position >= 1U) && (position <= n_)) ?
          current->sequence[position - 1U] : 0;
      }
    }
    for (unsigned int i = 0; i <= n_; i++) {
      host_scale_[i] = static_cast<Real>(
        fc[bucket.front()]->exp_matrices->scale[i]);
      host_mlbase_[i] = static_cast<Real>(
        fc[bucket.front()]->exp_matrices->expMLbase[i]);
    }
    *host_params_ = make_gpu_params<Real>(fc[bucket.front()]->exp_params);

    if ((cudaMemcpyAsync(d_sequence_.get(),
                         host_sequence_,
                         sizeof(short) * sequence_count_,
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(d_sequence2_.get(),
                         host_sequence2_,
                         sizeof(short) * sequence_count_,
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(d_characters_.get(),
                         host_characters_,
                         sizeof(char) * sequence_count_,
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(d_scale_.get(),
                         host_scale_,
                         sizeof(Real) * (n_ + 1U),
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(d_mlbase_.get(),
                         host_mlbase_,
                         sizeof(Real) * (n_ + 1U),
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(d_params_.get(),
                         host_params_,
                         sizeof(*host_params_),
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess))
      return false;

    const bool synchronize_phases = profile_enabled_;
    auto launch_phase = [&](unsigned int phase, const char *name) {
      if ((cudaEventRecord(phase_events_[phase], stream_) != cudaSuccess) ||
          (cudaGraphLaunch(graph_execs_[phase], stream_) != cudaSuccess))
        return false;
      if (synchronize_phases) {
        const cudaError_t status = cudaStreamSynchronize(stream_);
        if (status != cudaSuccess) {
          std::fprintf(stderr,
                       "CUDA PF blocked phase %s failed: %s\n",
                       name,
                       cudaGetErrorString(status));
          return false;
        }
      }
      return true;
    };

    bool execution_ok;
    if (profile_enabled_) {
      execution_ok = launch_phase(0, "metadata") &&
                     (cudaEventRecord(phase_events_[1], stream_) == cudaSuccess) &&
                     launch_forward_schedule(true) &&
                     (cudaStreamSynchronize(stream_) == cudaSuccess) &&
                     launch_phase(2, "exterior");
      if (execution_ok && with_bpp_)
        execution_ok =
          (cudaEventRecord(phase_events_[3], stream_) == cudaSuccess) &&
          clear_reverse_state() &&
          launch_reverse_schedule(true) &&
          (cudaStreamSynchronize(stream_) == cudaSuccess) &&
          launch_phase(4, "probability-validation");
      else if (execution_ok)
        execution_ok =
          (cudaEventRecord(phase_events_[3], stream_) == cudaSuccess) &&
          (cudaEventRecord(phase_events_[4], stream_) == cudaSuccess);
    } else {
      execution_ok = launch_phase(0, "metadata") &&
                     launch_phase(1, "forward") &&
                     launch_phase(2, "exterior") &&
                     ((!with_bpp_) || launch_phase(3, "reverse")) &&
                     ((!with_bpp_) || launch_phase(4, "probability-validation")) &&
                     (with_bpp_ ||
                      ((cudaEventRecord(phase_events_[3], stream_) == cudaSuccess) &&
                       (cudaEventRecord(phase_events_[4], stream_) == cudaSuccess)));
    }

    if ((!execution_ok) ||
        (cudaEventRecord(phase_events_[5], stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(host_roots_,
                         roots_.get(),
                         sizeof(Real) * batch_size_,
                         cudaMemcpyDeviceToHost,
                         stream_) != cudaSuccess) ||
        (with_bpp_ &&
         ((cudaMemcpyAsync(host_probabilities_,
                           probabilities_.get(),
                           sizeof(FLT_OR_DBL) * output_count_,
                           cudaMemcpyDeviceToHost,
                           stream_) != cudaSuccess) ||
          (cudaMemcpyAsync(host_valid_flags_,
                           valid_flags_.get(),
                           sizeof(int) * batch_size_,
                           cudaMemcpyDeviceToHost,
                           stream_) != cudaSuccess))) ||
        (cudaEventRecord(phase_events_[6], stream_) != cudaSuccess) ||
        (cudaStreamSynchronize(stream_) != cudaSuccess))
      return false;

    if (profile_enabled_) {
      float milliseconds[6]{};
      for (unsigned int phase = 0; phase < 5U; phase++)
        (void)cudaEventElapsedTime(milliseconds + phase,
                                   phase_events_[phase],
                                   phase_events_[phase + 1U]);
      (void)cudaEventElapsedTime(milliseconds + 5U,
                                 phase_events_[5],
                                 phase_events_[6]);
      const float forward_far = elapsed_sum(forward_far_begin_,
                                             forward_far_end_);
      const float forward_local = elapsed_sum(forward_local_begin_,
                                               forward_local_end_);
      const float reverse_dM = elapsed_sum(reverse_dM_begin_, reverse_dM_end_);
      const float reverse_dS = elapsed_sum(reverse_dS_begin_, reverse_dS_end_);
      const float reverse_local = elapsed_sum(reverse_local_begin_,
                                               reverse_local_end_);
      std::fprintf(stderr,
                   "CUDA PF phases device=%d engine=blocked gemm=%s "
                   "metadata=%.3fms forward-far-gemm=%.3fms "
                   "forward-local-internal=%.3fms exterior=%.3fms "
                   "reverse-dM-gemm=%.3fms reverse-dS-gemm=%.3fms "
                   "reverse-local-dB=%.3fms probability-validation=%.3fms "
                   "D2H=%.3fms\n",
                   device_,
                   (gemm_mode_ == BlockedGemmMode::emulated) ? "emulated" : "native",
                   milliseconds[0],
                   forward_far,
                   forward_local,
                   milliseconds[2],
                   reverse_dM,
                   reverse_dS,
                   reverse_local,
                   milliseconds[4],
                   milliseconds[5]);
    }

    return true;
  }

  const Real *roots() const
  {
    return host_roots_;
  }

  const FLT_OR_DBL *probabilities() const
  {
    return host_probabilities_;
  }

  const int *valid_flags() const
  {
    return host_valid_flags_;
  }

  size_t output_stride() const
  {
    return output_stride_;
  }

private:
  static bool profile_requested()
  {
    const char *profile = std::getenv("VRNA_CUDA_PF_PROFILE");
    return profile && (std::strcmp(profile, "1") == 0);
  }

  static void destroy_events(std::vector<cudaEvent_t> &events)
  {
    for (cudaEvent_t event : events)
      if (event)
        (void)cudaEventDestroy(event);
  }

  static float elapsed_sum(const std::vector<cudaEvent_t> &begin,
                           const std::vector<cudaEvent_t> &end)
  {
    float total = 0;
    for (size_t i = 0; i < begin.size(); i++) {
      float milliseconds = 0;
      if (cudaEventElapsedTime(&milliseconds, begin[i], end[i]) == cudaSuccess)
        total += milliseconds;
    }
    return total;
  }

  bool create_events(std::vector<cudaEvent_t> &events)
  {
    events.assign(tile_count_, nullptr);
    for (cudaEvent_t &event : events)
      if (cudaEventCreate(&event) != cudaSuccess)
        return false;
    return true;
  }

  static unsigned int read_gemm_crossover()
  {
    const char *setting = std::getenv("VRNA_CUDA_PF_GEMM_CROSSOVER");
    if (!setting || !setting[0])
      return 2U;
    return static_cast<unsigned int>(std::max(2, std::atoi(setting)));
  }

  cublasComputeType_t compute_type() const
  {
    if constexpr (std::is_same<Real, double>::value) {
      if (gemm_mode_ == BlockedGemmMode::emulated)
        return CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
    }
    return CublasType<Real>::compute_type;
  }

  bool gemm(cublasOperation_t transa,
            cublasOperation_t transb,
            int               m,
            int               n,
            int               k,
            const Real       *a,
            const Real       *b,
            Real             *c)
  {
    const Real alpha = Real(1);
    const Real beta = Real(0);
    return cublasGemmStridedBatchedEx(cublas_,
                                      transa,
                                      transb,
                                      m,
                                      n,
                                      k,
                                      &alpha,
                                      a,
                                      CublasType<Real>::data_type,
                                      matrix_dim_,
                                      static_cast<long long>(matrix_stride_),
                                      b,
                                      CublasType<Real>::data_type,
                                      matrix_dim_,
                                      static_cast<long long>(matrix_stride_),
                                      &beta,
                                      c,
                                      CublasType<Real>::data_type,
                                      matrix_dim_,
                                      static_cast<long long>(matrix_stride_),
                                      static_cast<int>(batch_size_),
                                      compute_type(),
                                      CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS;
  }

  bool forward_far(unsigned int diagonal)
  {
    if (diagonal < gemm_crossover_)
      return true;
    const int k = static_cast<int>((diagonal - 1U) * G);
    for (unsigned int tile_i = 0;
         tile_i + diagonal < tile_count_;
         tile_i++) {
      const unsigned int i0 = 1U + tile_i * G;
      const unsigned int j0 = 1U + (tile_i + diagonal) * G;
      const Real *right = S_.get() + blocked_cell(0, i0 + G, j0, matrix_dim_);
      const Real *left = M_.get() + blocked_cell(0, i0, i0 + G - 1U, matrix_dim_);
      Real *output = C_.get() + blocked_cell(0, i0, j0, matrix_dim_);
      if (!gemm(CUBLAS_OP_N,
                CUBLAS_OP_N,
                static_cast<int>(G),
                static_cast<int>(G),
                k,
                right,
                left,
                output))
        return false;
    }
    return true;
  }

  bool reverse_far_dM(unsigned int diagonal)
  {
    for (unsigned int tile_i = 0;
         tile_i + diagonal < tile_count_;
         tile_i++) {
      const unsigned int i0 = 1U + tile_i * G;
      const unsigned int j0 = 1U + (tile_i + diagonal) * G;
      const unsigned int block_end = j0 + G - 1U;
      if (block_end < n_) {
        const int k = static_cast<int>(n_ - block_end);
        const Real *right = S_.get() +
          blocked_cell(0, j0 + 1U, block_end + 1U, matrix_dim_);
        const Real *source = A_.get() +
          blocked_cell(0, i0, block_end + 1U, matrix_dim_);
        Real *output = dM_.get() + blocked_cell(0, i0, j0, matrix_dim_);
        if (!gemm(CUBLAS_OP_T,
                  CUBLAS_OP_N,
                  static_cast<int>(G),
                  static_cast<int>(G),
                  k,
                  right,
                  source,
                  output))
          return false;
      }

    }
    return true;
  }

  bool reverse_far_dS(unsigned int diagonal)
  {
    for (unsigned int tile_i = 0;
         tile_i + diagonal < tile_count_;
         tile_i++) {
      const unsigned int i0 = 1U + tile_i * G;
      const unsigned int j0 = 1U + (tile_i + diagonal) * G;
      if (i0 > 1U) {
        const int k = static_cast<int>(i0 - 1U);
        const Real *source = A_.get() + blocked_cell(0, 1U, j0, matrix_dim_);
        const Real *left = M_.get() + blocked_cell(0, 1U, i0 - 1U, matrix_dim_);
        Real *output = dS_.get() + blocked_cell(0, i0, j0, matrix_dim_);
        if (!gemm(CUBLAS_OP_N,
                  CUBLAS_OP_T,
                  static_cast<int>(G),
                  static_cast<int>(G),
                  k,
                  source,
                  left,
                  output))
          return false;
      }
    }
    return true;
  }

  bool launch_forward_schedule(bool record_timings)
  {
    for (unsigned int diagonal = 0; diagonal < tile_count_; diagonal++) {
      if (record_timings &&
          (cudaEventRecord(forward_far_begin_[diagonal], stream_) != cudaSuccess))
        return false;
      if (!forward_far(diagonal))
        return false;
      if (record_timings &&
          ((cudaEventRecord(forward_far_end_[diagonal], stream_) != cudaSuccess) ||
           (cudaEventRecord(forward_local_begin_[diagonal], stream_) != cudaSuccess)))
        return false;
      const unsigned int blocks = (tile_count_ - diagonal) * batch_size_;
      forward_local_blocked_diagonal<Real, G>
        <<<blocks, kBlockedLocalThreads, 0, stream_>>>(B_.get(),
                                                       S_.get(),
                                                       U_.get(),
                                                       C_.get(),
                                                       M_.get(),
                                                       pair_types_.get(),
                                                       d_sequence_.get(),
                                                       d_characters_.get(),
                                                       d_scale_.get(),
                                                       d_mlbase_.get(),
                                                       n_,
                                                       matrix_dim_,
                                                       batch_size_,
                                                       diagonal,
                                                       tile_count_,
                                                       diagonal >= gemm_crossover_,
                                                       d_params_.get());
      if (record_timings &&
          (cudaEventRecord(forward_local_end_[diagonal], stream_) != cudaSuccess))
        return false;
    }
    return cudaPeekAtLastError() == cudaSuccess;
  }

  bool clear_reverse_state()
  {
    return (cudaMemsetAsync(dB_.get(),
                            0,
                            sizeof(Real) * matrix_count_,
                            stream_) == cudaSuccess) &&
           (cudaMemsetAsync(dS_.get(),
                            0,
                            sizeof(Real) * matrix_count_,
                            stream_) == cudaSuccess) &&
           (cudaMemsetAsync(dM_.get(),
                            0,
                            sizeof(Real) * matrix_count_,
                            stream_) == cudaSuccess) &&
           (cudaMemsetAsync(dU_.get(),
                            0,
                            sizeof(Real) * matrix_count_,
                            stream_) == cudaSuccess) &&
           (cudaMemsetAsync(A_.get(),
                            0,
                            sizeof(Real) * matrix_count_,
                            stream_) == cudaSuccess) &&
           (cudaMemsetAsync(probabilities_.get(),
                            0,
                            sizeof(FLT_OR_DBL) * output_count_,
                            stream_) == cudaSuccess) &&
           (cudaMemsetAsync(valid_flags_.get(),
                            1,
                            sizeof(int) * batch_size_,
                            stream_) == cudaSuccess);
  }

  bool launch_reverse_schedule(bool record_timings)
  {
    for (unsigned int reverse = tile_count_; reverse-- > 0;) {
      if (record_timings &&
          (cudaEventRecord(reverse_dM_begin_[reverse], stream_) != cudaSuccess))
        return false;
      if (!reverse_far_dM(reverse))
        return false;
      if (record_timings &&
          ((cudaEventRecord(reverse_dM_end_[reverse], stream_) != cudaSuccess) ||
           (cudaEventRecord(reverse_dS_begin_[reverse], stream_) != cudaSuccess)))
        return false;
      if (!reverse_far_dS(reverse))
        return false;
      if (record_timings &&
          ((cudaEventRecord(reverse_dS_end_[reverse], stream_) != cudaSuccess) ||
           (cudaEventRecord(reverse_local_begin_[reverse], stream_) != cudaSuccess)))
        return false;
      const unsigned int blocks = (tile_count_ - reverse) * batch_size_;
      reverse_local_blocked_diagonal<Real, G>
        <<<blocks, kBlockedLocalThreads, 0, stream_>>>(B_.get(),
                                                       S_.get(),
                                                       M_.get(),
                                                       dB_.get(),
                                                       dS_.get(),
                                                       dM_.get(),
                                                       dU_.get(),
                                                       A_.get(),
                                                       q5_.get(),
                                                       q3_.get(),
                                                       roots_.get(),
                                                       probabilities_.get(),
                                                       pair_types_.get(),
                                                       d_sequence_.get(),
                                                       d_scale_.get(),
                                                       d_mlbase_.get(),
                                                       output_stride_,
                                                       n_,
                                                       matrix_dim_,
                                                       batch_size_,
                                                       reverse,
                                                       tile_count_,
                                                       d_params_.get());
      if (record_timings &&
          (cudaEventRecord(reverse_local_end_[reverse], stream_) != cudaSuccess))
        return false;
    }
    return cudaPeekAtLastError() == cudaSuccess;
  }

  bool initialize()
  {
    int current_device = -1;
    if ((cudaGetDevice(&current_device) != cudaSuccess) ||
        (current_device != device_))
      return false;

    const size_t pair_mask_count = packed_tile_count_ * batch_size_ * G;
    if ((!d_sequence_.allocate(sequence_count_)) ||
        (!d_sequence2_.allocate(sequence_count_)) ||
        (!d_characters_.allocate(sequence_count_)) ||
        (!d_scale_.allocate(n_ + 1U)) ||
        (!d_mlbase_.allocate(n_ + 1U)) ||
        (!d_params_.allocate(1)) ||
        (!tile_coordinates_.allocate(packed_tile_count_)) ||
        (!pair_types_.allocate(matrix_count_)) ||
        (!pair_masks_.allocate(pair_mask_count)) ||
        (!B_.allocate(matrix_count_)) ||
        (!S_.allocate(matrix_count_)) ||
        (!U_.allocate(matrix_count_)) ||
        (!C_.allocate(matrix_count_)) ||
        (!M_.allocate(matrix_count_)) ||
        (!q5_.allocate(sequence_count_)) ||
        (!q3_.allocate(sequence_count_)) ||
        (!roots_.allocate(batch_size_)) ||
        (!cublas_workspace_.allocate(kCublasWorkspaceBytes)) ||
        (with_bpp_ &&
         ((!dB_.allocate(matrix_count_)) ||
          (!dS_.allocate(matrix_count_)) ||
          (!dM_.allocate(matrix_count_)) ||
          (!dU_.allocate(matrix_count_)) ||
          (!A_.allocate(matrix_count_)) ||
          (!probabilities_.allocate(output_count_)) ||
          (!valid_flags_.allocate(batch_size_)))))
      return false;

    /* Keep plan construction fully materialized before the non-blocking
     * stream is captured or receives input copies. */
    if (cudaDeviceSynchronize() != cudaSuccess)
      return false;

    if ((cudaHostAlloc(reinterpret_cast<void **>(&host_sequence_),
                       sizeof(short) * sequence_count_,
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_sequence2_),
                       sizeof(short) * sequence_count_,
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_characters_),
                       sizeof(char) * sequence_count_,
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_scale_),
                       sizeof(Real) * (n_ + 1U),
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_mlbase_),
                       sizeof(Real) * (n_ + 1U),
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_params_),
                       sizeof(GpuPfParams<Real>),
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_roots_),
                       sizeof(Real) * batch_size_,
                       cudaHostAllocDefault) != cudaSuccess) ||
        (with_bpp_ &&
         ((cudaHostAlloc(reinterpret_cast<void **>(&host_probabilities_),
                         sizeof(FLT_OR_DBL) * output_count_,
                         cudaHostAllocDefault) != cudaSuccess) ||
          (cudaHostAlloc(reinterpret_cast<void **>(&host_valid_flags_),
                         sizeof(int) * batch_size_,
                         cudaHostAllocDefault) != cudaSuccess))))
      return false;

    std::vector<uint2> coordinates;
    coordinates.reserve(packed_tile_count_);
    for (unsigned int diagonal = 0; diagonal < tile_count_; diagonal++)
      for (unsigned int tile_i = 0; tile_i + diagonal < tile_count_; tile_i++)
        coordinates.push_back(make_uint2(tile_i, tile_i + diagonal));

    if ((cudaMemcpy(tile_coordinates_.get(),
                    coordinates.data(),
                    sizeof(uint2) * coordinates.size(),
                    cudaMemcpyHostToDevice) != cudaSuccess) ||
        (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) ||
        (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) ||
        (cublasSetStream(cublas_, stream_) != CUBLAS_STATUS_SUCCESS) ||
        (cublasSetWorkspace(cublas_,
                            cublas_workspace_.get(),
                            kCublasWorkspaceBytes) != CUBLAS_STATUS_SUCCESS))
      return false;

    for (cudaEvent_t &event : phase_events_)
      if (cudaEventCreate(&event) != cudaSuccess)
        return false;

    if (profile_enabled_ &&
        ((!create_events(forward_far_begin_)) ||
         (!create_events(forward_far_end_)) ||
         (!create_events(forward_local_begin_)) ||
         (!create_events(forward_local_end_)) ||
         (with_bpp_ &&
          ((!create_events(reverse_dM_begin_)) ||
           (!create_events(reverse_dM_end_)) ||
           (!create_events(reverse_dS_begin_)) ||
           (!create_events(reverse_dS_end_)) ||
           (!create_events(reverse_local_begin_)) ||
           (!create_events(reverse_local_end_))))))
      return false;

    if constexpr (std::is_same<Real, double>::value) {
      if ((gemm_mode_ == BlockedGemmMode::emulated) &&
          ((cublasSetEmulationStrategy(cublas_,
                                       CUBLAS_EMULATION_STRATEGY_PERFORMANT) !=
            CUBLAS_STATUS_SUCCESS) ||
           (cublasSetFixedPointEmulationMantissaControl(
              cublas_, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC) !=
            CUBLAS_STATUS_SUCCESS)))
        return false;
    }

    auto begin_graph = [&]() {
      return cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) ==
             cudaSuccess;
    };
    auto end_graph = [&](unsigned int phase) {
      return (cudaStreamEndCapture(stream_, graphs_ + phase) == cudaSuccess) &&
             (cudaGraphInstantiate(graph_execs_ + phase,
                                   graphs_[phase],
                                   nullptr,
                                   nullptr,
                                   0) == cudaSuccess);
    };

    if (!begin_graph())
      return false;

    if ((cudaMemsetAsync(pair_types_.get(),
                         0,
                         sizeof(unsigned char) * matrix_count_,
                         stream_) != cudaSuccess) ||
        (cudaMemsetAsync(B_.get(), 0, sizeof(Real) * matrix_count_, stream_) != cudaSuccess) ||
        (cudaMemsetAsync(S_.get(), 0, sizeof(Real) * matrix_count_, stream_) != cudaSuccess) ||
        (cudaMemsetAsync(U_.get(), 0, sizeof(Real) * matrix_count_, stream_) != cudaSuccess) ||
        (cudaMemsetAsync(C_.get(), 0, sizeof(Real) * matrix_count_, stream_) != cudaSuccess) ||
        (cudaMemsetAsync(M_.get(), 0, sizeof(Real) * matrix_count_, stream_) != cudaSuccess) ||
        (cudaMemsetAsync(q5_.get(), 0, sizeof(Real) * sequence_count_, stream_) != cudaSuccess) ||
        (cudaMemsetAsync(q3_.get(), 0, sizeof(Real) * sequence_count_, stream_) != cudaSuccess))
      return false;

    build_blocked_pair_metadata<Real>
      <<<static_cast<unsigned int>(packed_tile_count_ * batch_size_),
          kBlockSize,
          0,
          stream_>>>(d_sequence2_.get(),
                     pair_types_.get(),
                     pair_masks_.get(),
                     tile_coordinates_.get(),
                     static_cast<unsigned int>(packed_tile_count_),
                     matrix_dim_,
                     n_,
                     batch_size_,
                     d_params_.get());

    if (!end_graph(0) || !begin_graph())
      return false;

    if (!launch_forward_schedule(false))
      return false;

    if (!end_graph(1) || !begin_graph())
      return false;

    compute_exterior_blocked<Real, G>
      <<<batch_size_, G * G, 0, stream_>>>(B_.get(),
                                           q5_.get(),
                                           q3_.get(),
                                           roots_.get(),
                                           pair_types_.get(),
                                           d_sequence_.get(),
                                           d_scale_.get(),
                                           n_,
                                           matrix_dim_,
                                           tile_count_,
                                           batch_size_,
                                           d_params_.get());

    if (!end_graph(2))
      return false;

    if (with_bpp_) {
      if (!begin_graph())
        return false;
      if (!clear_reverse_state())
        return false;

      if (!launch_reverse_schedule(false))
        return false;

      if (!end_graph(3) || !begin_graph())
        return false;

      validate_blocked_probabilities<Real>
        <<<static_cast<unsigned int>(n_) * batch_size_,
            kBlockSize,
            0,
            stream_>>>(B_.get(),
                       dB_.get(),
                       valid_flags_.get(),
                       n_,
                       matrix_dim_,
                       batch_size_);
      if (!end_graph(4))
        return false;
    }

    return true;
  }

  unsigned int n_;
  unsigned int batch_size_;
  bool with_bpp_;
  int device_;
  BlockedGemmMode gemm_mode_;
  unsigned int tile_count_;
  unsigned int matrix_dim_;
  size_t matrix_stride_;
  size_t matrix_count_;
  size_t sequence_count_;
  size_t output_stride_;
  size_t output_count_;
  size_t packed_tile_count_;
  unsigned int gemm_crossover_;
  cudaStream_t stream_;
  cudaGraph_t graphs_[5]{};
  cudaGraphExec_t graph_execs_[5]{};
  cublasHandle_t cublas_;
  short *host_sequence_;
  short *host_sequence2_;
  char *host_characters_;
  Real *host_scale_;
  Real *host_mlbase_;
  GpuPfParams<Real> *host_params_;
  Real *host_roots_;
  FLT_OR_DBL *host_probabilities_;
  int *host_valid_flags_;
  bool profile_enabled_;
  bool ready_;
  DeviceBuffer<short> d_sequence_;
  DeviceBuffer<short> d_sequence2_;
  DeviceBuffer<char> d_characters_;
  DeviceBuffer<Real> d_scale_;
  DeviceBuffer<Real> d_mlbase_;
  DeviceBuffer<GpuPfParams<Real>> d_params_;
  DeviceBuffer<uint2> tile_coordinates_;
  DeviceBuffer<unsigned char> pair_types_;
  DeviceBuffer<unsigned int> pair_masks_;
  DeviceBuffer<Real> B_, S_, U_, C_, M_;
  DeviceBuffer<Real> q5_, q3_, roots_;
  DeviceBuffer<Real> dB_, dS_, dM_, dU_, A_;
  DeviceBuffer<FLT_OR_DBL> probabilities_;
  DeviceBuffer<int> valid_flags_;
  DeviceBuffer<unsigned char> cublas_workspace_;
  cudaEvent_t phase_events_[7]{};
  std::vector<cudaEvent_t> forward_far_begin_, forward_far_end_;
  std::vector<cudaEvent_t> forward_local_begin_, forward_local_end_;
  std::vector<cudaEvent_t> reverse_dM_begin_, reverse_dM_end_;
  std::vector<cudaEvent_t> reverse_dS_begin_, reverse_dS_end_;
  std::vector<cudaEvent_t> reverse_local_begin_, reverse_local_end_;
};


bool
default_hard_constraints(const vrna_fold_compound_t *fc)
{
  const vrna_hc_t *hc = fc->hc;
  const vrna_md_t &md = fc->exp_params->model_details;
  const unsigned int n = fc->length;
  if ((!hc) ||
      (hc->type != VRNA_HC_DEFAULT) ||
      hc->f ||
      hc->data ||
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
        if (type && !(md.noGU && ((type == 3) || (type == 4)))) {
          expected = VRNA_CONSTRAINT_CONTEXT_ALL_LOOPS;
          if (md.noGUclosure && ((type == 3) || (type == 4)))
            expected &= ~(VRNA_CONSTRAINT_CONTEXT_HP_LOOP | VRNA_CONSTRAINT_CONTEXT_MB_LOOP);
        }
      }
      if (hc->mx[static_cast<size_t>(n) * i + j] != expected)
        return false;
    }
  }
  return true;
}


bool
eligible(vrna_fold_compound_t *fc,
         bool                 with_bpp)
{
  if ((!fc) ||
      (fc->type != VRNA_FC_TYPE_SINGLE) ||
      (fc->strands != 1) ||
      (!fc->exp_params) ||
      (!fc->exp_matrices) ||
      (fc->exp_matrices->type != VRNA_MX_DEFAULT) ||
      (!fc->exp_matrices->scale) ||
      (!fc->exp_matrices->expMLbase) ||
      (!fc->sequence) ||
      (!fc->sequence_encoding) ||
      (!fc->sequence_encoding2) ||
      (with_bpp && !fc->exp_matrices->probs) ||
      fc->sc ||
      fc->domains_up ||
      fc->aux_grammar ||
      fc->stat_cb)
    return false;

  const vrna_md_t &md = fc->exp_params->model_details;
  if ((md.dangles != 2) ||
      md.noLP ||
      md.logML ||
      md.circ ||
      md.gquad ||
      md.uniq_ML ||
      (md.backtrack_type != VRNA_MODEL_DEFAULT_BACKTRACK_TYPE) ||
      (md.salt != VRNA_MODEL_DEFAULT_SALT) ||
      (fc->exp_params->param_file[0] != '\0'))
    return false;

  return default_hard_constraints(fc);
}


bool
same_bucket(const vrna_fold_compound_t *a,
            const vrna_fold_compound_t *b)
{
  return (a->length == b->length) &&
         (std::memcmp(a->exp_params, b->exp_params, sizeof(vrna_exp_param_t)) == 0);
}


template <typename Real>
bool
run_bucket_reference(vrna_fold_compound_t       **fc,
           const std::vector<size_t>  &bucket,
           unsigned char              *handled,
           float                      *energies,
           bool                       with_bpp)
{
  const unsigned int n = fc[bucket.front()]->length;
  const unsigned int batch_size = static_cast<unsigned int>(bucket.size());
  const size_t triangular = static_cast<size_t>(n) * (n + 1) / 2;
  const size_t matrix_count = triangular * batch_size;
  const size_t sequence_count = static_cast<size_t>(n + 2) * batch_size;

  std::vector<short> host_sequence(sequence_count);
  std::vector<short> host_sequence2(sequence_count);
  std::vector<char> host_characters(sequence_count, 0);
  std::vector<Real> host_scale(n + 1);
  std::vector<Real> host_mlbase(n + 1);
  const GpuPfParams<Real> host_params = make_gpu_params<Real>(fc[bucket.front()]->exp_params);
  for (unsigned int b = 0; b < batch_size; b++) {
    const vrna_fold_compound_t *current = fc[bucket[b]];
    for (unsigned int position = 0; position <= n + 1; position++) {
      host_sequence[static_cast<size_t>(position) * batch_size + b] =
        current->sequence_encoding[position];
      host_sequence2[static_cast<size_t>(position) * batch_size + b] =
        current->sequence_encoding2[position];
      if ((position >= 1) && (position <= n))
        host_characters[static_cast<size_t>(position) * batch_size + b] =
          current->sequence[position - 1];
    }
  }
  for (unsigned int i = 0; i <= n; i++) {
    host_scale[i] = static_cast<Real>(fc[bucket.front()]->exp_matrices->scale[i]);
    host_mlbase[i] = static_cast<Real>(fc[bucket.front()]->exp_matrices->expMLbase[i]);
  }

  DeviceBuffer<short> d_sequence;
  DeviceBuffer<short> d_sequence2;
  DeviceBuffer<char> d_characters;
  DeviceBuffer<Real> d_scale;
  DeviceBuffer<Real> d_mlbase;
  DeviceBuffer<GpuPfParams<Real>> d_params;
  DeviceBuffer<Real> B, E, S, Q, M, M2;
  DeviceBuffer<Real> dB, dE, dS, dQ, dM, dM2;
  DeviceBuffer<Real> roots;
  DeviceBuffer<FLT_OR_DBL> probabilities;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  if ((!d_sequence.allocate(sequence_count)) ||
      (!d_sequence2.allocate(sequence_count)) ||
      (!d_characters.allocate(sequence_count)) ||
      (!d_scale.allocate(n + 1)) ||
      (!d_mlbase.allocate(n + 1)) ||
      (!d_params.allocate(1)) ||
      (!B.allocate(matrix_count)) ||
      (!E.allocate(matrix_count)) ||
      (!S.allocate(matrix_count)) ||
      (!Q.allocate(matrix_count)) ||
      (!M.allocate(matrix_count)) ||
      (!M2.allocate(matrix_count)) ||
      (!roots.allocate(batch_size)) ||
      (with_bpp &&
       ((!dB.allocate(matrix_count)) ||
        (!dE.allocate(matrix_count)) ||
        (!dS.allocate(matrix_count)) ||
        (!dQ.allocate(matrix_count)) ||
        (!dM.allocate(matrix_count)) ||
        (!dM2.allocate(matrix_count)) ||
        (!probabilities.allocate(matrix_count)))))
    return false;

  if ((cudaMemcpyAsync(d_sequence.get(), host_sequence.data(),
                       sizeof(short) * sequence_count, cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpyAsync(d_sequence2.get(), host_sequence2.data(),
                       sizeof(short) * sequence_count, cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpyAsync(d_characters.get(), host_characters.data(),
                       sizeof(char) * sequence_count, cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpyAsync(d_scale.get(), host_scale.data(),
                       sizeof(Real) * (n + 1), cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpyAsync(d_mlbase.get(), host_mlbase.data(),
                       sizeof(Real) * (n + 1), cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpyAsync(d_params.get(), &host_params,
                       sizeof(host_params), cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemsetAsync(B.get(), 0, sizeof(Real) * matrix_count) != cudaSuccess) ||
      (cudaMemsetAsync(E.get(), 0, sizeof(Real) * matrix_count) != cudaSuccess) ||
      (cudaMemsetAsync(S.get(), 0, sizeof(Real) * matrix_count) != cudaSuccess) ||
      (cudaMemsetAsync(Q.get(), 0, sizeof(Real) * matrix_count) != cudaSuccess) ||
      (cudaMemsetAsync(M.get(), 0, sizeof(Real) * matrix_count) != cudaSuccess) ||
      (cudaMemsetAsync(M2.get(), 0, sizeof(Real) * matrix_count) != cudaSuccess))
    return false;
  if ((cudaStreamSynchronize(nullptr) != cudaSuccess) ||
      (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) ||
      (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess))
    return false;


  for (unsigned int span = 0; span < n; span++) {
    const unsigned int cells = (n - span) * batch_size;
    const unsigned int blocks = (cells + kBlockSize - 1) / kBlockSize;
    compute_paired_span<Real><<<blocks, kBlockSize, 0, stream>>>(B.get(), M2.get(),
                                                       d_sequence.get(), d_sequence2.get(),
                                                       d_characters.get(), d_scale.get(),
                                                       n, batch_size, span, d_params.get());
    compute_aux_span<Real><<<blocks, kBlockSize, 0, stream>>>(B.get(), E.get(), S.get(), Q.get(), M.get(), M2.get(),
                                                    d_sequence.get(), d_sequence2.get(),
                                                    d_scale.get(), d_mlbase.get(),
                                                    n, batch_size, span, d_params.get());
  }


  if (with_bpp) {
    if ((cudaMemsetAsync(dB.get(), 0, sizeof(Real) * matrix_count, stream) != cudaSuccess) ||
        (cudaMemsetAsync(dE.get(), 0, sizeof(Real) * matrix_count, stream) != cudaSuccess) ||
        (cudaMemsetAsync(dS.get(), 0, sizeof(Real) * matrix_count, stream) != cudaSuccess) ||
        (cudaMemsetAsync(dQ.get(), 0, sizeof(Real) * matrix_count, stream) != cudaSuccess) ||
        (cudaMemsetAsync(dM.get(), 0, sizeof(Real) * matrix_count, stream) != cudaSuccess) ||
        (cudaMemsetAsync(dM2.get(), 0, sizeof(Real) * matrix_count, stream) != cudaSuccess))
      return false;

    seed_root<Real><<<(batch_size + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      dQ.get(), Q.get(), n, batch_size);

    for (unsigned int span = n; span-- > 0;) {
      const unsigned int cells = (n - span) * batch_size;
      const unsigned int blocks = (cells + kBlockSize - 1) / kBlockSize;
      reverse_span<Real><<<blocks, kBlockSize, 0, stream>>>(B.get(), E.get(), S.get(), Q.get(), M.get(), M2.get(),
                                                  dB.get(), dE.get(), dS.get(), dQ.get(), dM.get(), dM2.get(),
                                                  d_sequence.get(), d_sequence2.get(),
                                                  d_scale.get(), d_mlbase.get(),
                                                  n, batch_size, span, d_params.get());
    }

    finalize_probabilities<Real><<<(matrix_count + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
      B.get(), dB.get(), probabilities.get(), matrix_count);
  }

  gather_roots<Real><<<(batch_size + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
    Q.get(), roots.get(), n, batch_size);

  if ((cudaStreamEndCapture(stream, &graph) != cudaSuccess) ||
      (cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0) != cudaSuccess) ||
      (cudaGraphLaunch(graph_exec, stream) != cudaSuccess))
    return false;

  std::vector<Real> host_roots(batch_size);
  std::vector<FLT_OR_DBL> host_probabilities(with_bpp ? matrix_count : 0);
  if ((cudaMemcpyAsync(host_roots.data(), roots.get(),
                       sizeof(Real) * batch_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess) ||
      (with_bpp &&
       (cudaMemcpyAsync(host_probabilities.data(), probabilities.get(),
                        sizeof(FLT_OR_DBL) * matrix_count,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess)) ||
      (cudaStreamSynchronize(stream) != cudaSuccess))
    return false;
  (void)cudaGraphExecDestroy(graph_exec);
  (void)cudaGraphDestroy(graph);
  (void)cudaStreamDestroy(stream);


  for (unsigned int b = 0; b < batch_size; b++) {
    const double root = static_cast<double>(host_roots[b]);
    bool valid = std::isfinite(root) && (root > 0.);
    if (with_bpp && valid) {
      std::vector<double> paired_sum(n + 1, 0.);
      for (unsigned int span = 0; span < n && valid; span++)
        for (unsigned int i = 1; i + span <= n; i++) {
          const unsigned int j = i + span;
          const double probability =
            host_probabilities[pf_index(span, i, b, n, batch_size)];
          if ((!std::isfinite(probability)) ||
              (probability < -5.e-5) ||
              (probability > 1.0005)) {
            valid = false;
            break;
          }
          if (i != j) {
            paired_sum[i] += probability;
            paired_sum[j] += probability;
          }
        }

      for (unsigned int i = 1; i <= n && valid; i++)
        if (paired_sum[i] > 1.0005)
          valid = false;
    }

    if (!valid)
      continue;

    vrna_fold_compound_t *current = fc[bucket[b]];
    energies[bucket[b]] = static_cast<float>(
      (-std::log(root) - n * std::log(current->exp_params->pf_scale)) *
      current->exp_params->kT / 1000.);

    if (with_bpp) {
      FLT_OR_DBL *destination = current->exp_matrices->probs;
      for (unsigned int i = 1; i < n; i++)
        for (unsigned int j = i + 1; j <= n; j++)
          destination[current->iindx[i] - j] =
            host_probabilities[pf_index(j - i, i, b, n, batch_size)];
    }
    handled[bucket[b]] = 1;
  }

  return true;
}



template <typename Real>
class PersistentReducedPlan {
public:
  PersistentReducedPlan(unsigned int n,
                        unsigned int batch_size,
                        bool         with_bpp,
                        int          device)
    : n_(n),
      batch_size_(batch_size),
      with_bpp_(with_bpp),
      device_(device),
      matrix_count_(static_cast<size_t>(n) * (n + 1) / 2 * batch_size),
      sequence_count_(static_cast<size_t>(n + 2) * batch_size),
      ring_count_(static_cast<size_t>(2) * n * batch_size),
      output_stride_(static_cast<size_t>(n + 1) * (n + 2) / 2),
      output_count_(output_stride_ * batch_size),
      stream_(nullptr),
      graph_(nullptr),
      graph_exec_(nullptr),
      host_sequence_(nullptr),
      host_sequence2_(nullptr),
      host_characters_(nullptr),
      host_scale_(nullptr),
      host_mlbase_(nullptr),
      host_params_(nullptr),
      host_roots_(nullptr),
      host_probabilities_(nullptr),
      host_valid_flags_(nullptr),
      ready_(false)
  {
    ready_ = initialize();
  }

  ~PersistentReducedPlan()
  {
    (void)cudaSetDevice(device_);
    if (stream_)
      (void)cudaStreamSynchronize(stream_);
    if (graph_exec_)
      (void)cudaGraphExecDestroy(graph_exec_);
    if (graph_)
      (void)cudaGraphDestroy(graph_);
    if (stream_)
      (void)cudaStreamDestroy(stream_);
    if (host_sequence_)
      (void)cudaFreeHost(host_sequence_);
    if (host_sequence2_)
      (void)cudaFreeHost(host_sequence2_);
    if (host_characters_)
      (void)cudaFreeHost(host_characters_);
    if (host_scale_)
      (void)cudaFreeHost(host_scale_);
    if (host_mlbase_)
      (void)cudaFreeHost(host_mlbase_);
    if (host_params_)
      (void)cudaFreeHost(host_params_);
    if (host_roots_)
      (void)cudaFreeHost(host_roots_);
    if (host_probabilities_)
      (void)cudaFreeHost(host_probabilities_);
    if (host_valid_flags_)
      (void)cudaFreeHost(host_valid_flags_);
  }

  PersistentReducedPlan(const PersistentReducedPlan &) = delete;
  PersistentReducedPlan &operator=(const PersistentReducedPlan &) = delete;

  bool matches(unsigned int n,
               unsigned int batch_size,
               bool         with_bpp,
               int          device) const
  {
    return (n_ == n) &&
           (batch_size_ == batch_size) &&
           (with_bpp_ == with_bpp) &&
           (device_ == device);
  }

  bool ready() const
  {
    return ready_;
  }

  bool execute(vrna_fold_compound_t      **fc,
               const std::vector<size_t> &bucket)
  {
    if ((!ready_) || (bucket.size() != batch_size_))
      return false;

    for (unsigned int b = 0; b < batch_size_; b++) {
      const vrna_fold_compound_t *current = fc[bucket[b]];
      for (unsigned int position = 0; position <= n_ + 1; position++) {
        host_sequence_[static_cast<size_t>(position) * batch_size_ + b] =
          current->sequence_encoding[position];
        host_sequence2_[static_cast<size_t>(position) * batch_size_ + b] =
          current->sequence_encoding2[position];
        host_characters_[static_cast<size_t>(position) * batch_size_ + b] =
          ((position >= 1) && (position <= n_)) ?
          current->sequence[position - 1] : 0;
      }
    }

    for (unsigned int i = 0; i <= n_; i++) {
      host_scale_[i] = static_cast<Real>(
        fc[bucket.front()]->exp_matrices->scale[i]);
      host_mlbase_[i] = static_cast<Real>(
        fc[bucket.front()]->exp_matrices->expMLbase[i]);
    }
    *host_params_ =
      make_gpu_params<Real>(fc[bucket.front()]->exp_params);

    if ((cudaMemcpyAsync(d_sequence_.get(),
                         host_sequence_,
                         sizeof(short) * sequence_count_,
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(d_sequence2_.get(),
                         host_sequence2_,
                         sizeof(short) * sequence_count_,
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(d_characters_.get(),
                         host_characters_,
                         sizeof(char) * sequence_count_,
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(d_scale_.get(),
                         host_scale_,
                         sizeof(Real) * (n_ + 1),
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(d_mlbase_.get(),
                         host_mlbase_,
                         sizeof(Real) * (n_ + 1),
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(d_params_.get(),
                         host_params_,
                         sizeof(*host_params_),
                         cudaMemcpyHostToDevice,
                         stream_) != cudaSuccess) ||
        (cudaGraphLaunch(graph_exec_, stream_) != cudaSuccess) ||
        (cudaMemcpyAsync(host_roots_,
                         roots_.get(),
                         sizeof(Real) * batch_size_,
                         cudaMemcpyDeviceToHost,
                         stream_) != cudaSuccess) ||
        (with_bpp_ &&
         ((cudaMemcpyAsync(host_probabilities_,
                           probabilities_.get(),
                           sizeof(FLT_OR_DBL) * output_count_,
                           cudaMemcpyDeviceToHost,
                           stream_) != cudaSuccess) ||
          (cudaMemcpyAsync(host_valid_flags_,
                           valid_flags_.get(),
                           sizeof(int) * batch_size_,
                           cudaMemcpyDeviceToHost,
                           stream_) != cudaSuccess))) ||
        (cudaStreamSynchronize(stream_) != cudaSuccess))
      return false;

    return true;
  }

  const Real *roots() const
  {
    return host_roots_;
  }

  const FLT_OR_DBL *probabilities() const
  {
    return host_probabilities_;
  }

  const int *valid_flags() const
  {
    return host_valid_flags_;
  }

  size_t output_stride() const
  {
    return output_stride_;
  }
private:
  bool initialize()
  {
    if (cudaSetDevice(device_) != cudaSuccess)
      return false;

    if ((!d_sequence_.allocate(sequence_count_)) ||
        (!d_sequence2_.allocate(sequence_count_)) ||
        (!d_characters_.allocate(sequence_count_)) ||
        (!d_scale_.allocate(n_ + 1)) ||
        (!d_mlbase_.allocate(n_ + 1)) ||
        (!d_params_.allocate(1)) ||
        (!B_.allocate(matrix_count_)) ||
        (!S_.allocate(matrix_count_)) ||
        (!M_.allocate(matrix_count_)) ||
        (!U_ring_.allocate(ring_count_)) ||
        (!M2_ring_.allocate(ring_count_)) ||
        (!q5_.allocate(sequence_count_)) ||
        (!q3_.allocate(sequence_count_)) ||
        (!roots_.allocate(batch_size_)) ||
        (with_bpp_ &&
         ((!dB_.allocate(matrix_count_)) ||
          (!dS_.allocate(matrix_count_)) ||
          (!dM_.allocate(matrix_count_)) ||
          (!dU_ring_.allocate(ring_count_)) ||
          (!probabilities_.allocate(output_count_)) ||
          (!paired_sums_.allocate(static_cast<size_t>(n_ + 1) * batch_size_)) ||
          (!valid_flags_.allocate(batch_size_)))))
      return false;

    if ((cudaHostAlloc(reinterpret_cast<void **>(&host_sequence_),
                       sizeof(short) * sequence_count_,
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_sequence2_),
                       sizeof(short) * sequence_count_,
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_characters_),
                       sizeof(char) * sequence_count_,
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_scale_),
                       sizeof(Real) * (n_ + 1),
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_mlbase_),
                       sizeof(Real) * (n_ + 1),
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_params_),
                       sizeof(GpuPfParams<Real>),
                       cudaHostAllocDefault) != cudaSuccess) ||
        (cudaHostAlloc(reinterpret_cast<void **>(&host_roots_),
                       sizeof(Real) * batch_size_,
                       cudaHostAllocDefault) != cudaSuccess) ||
        (with_bpp_ &&
         ((cudaHostAlloc(reinterpret_cast<void **>(&host_probabilities_),
                         sizeof(FLT_OR_DBL) * output_count_,
                         cudaHostAllocDefault) != cudaSuccess) ||
          (cudaHostAlloc(reinterpret_cast<void **>(&host_valid_flags_),
                         sizeof(int) * batch_size_,
                         cudaHostAllocDefault) != cudaSuccess))))
      return false;

    if ((cudaDeviceSynchronize() != cudaSuccess) ||
        (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) ||
        (cudaStreamBeginCapture(stream_,
                                cudaStreamCaptureModeThreadLocal) != cudaSuccess))
      return false;

    for (unsigned int span = 0; span < n_; span++) {
      const unsigned int cells = (n_ - span) * batch_size_;
      const unsigned int blocks = (cells + kBlockSize - 1) / kBlockSize;
      compute_pair_aux_reduced_span<Real><<<blocks, kBlockSize, 0, stream_>>>(
        B_.get(),
        S_.get(),
        U_ring_.get(),
        M2_ring_.get(),
        d_sequence_.get(),
        d_sequence2_.get(),
        d_characters_.get(),
        d_scale_.get(),
        d_mlbase_.get(),
        n_,
        batch_size_,
        span,
        d_params_.get());
      compute_multibranch_reduced_span<Real><<<blocks, kBlockSize, 0, stream_>>>(
        S_.get(),
        M_.get(),
        U_ring_.get(),
        M2_ring_.get(),
        n_,
        batch_size_,
        span);
    }

    compute_exterior_vectors<Real><<<batch_size_, kBlockSize, 0, stream_>>>(
      B_.get(),
      q5_.get(),
      q3_.get(),
      roots_.get(),
      d_sequence_.get(),
      d_sequence2_.get(),
      d_scale_.get(),
      n_,
      batch_size_,
      d_params_.get());

    if (with_bpp_) {
      if ((cudaMemsetAsync(probabilities_.get(),
                           0,
                           sizeof(FLT_OR_DBL) * output_count_,
                           stream_) != cudaSuccess) ||
          (cudaMemsetAsync(paired_sums_.get(),
                           0,
                           sizeof(Real) * (n_ + 1) * batch_size_,
                           stream_) != cudaSuccess) ||
          (cudaMemsetAsync(valid_flags_.get(),
                           1,
                           sizeof(int) * batch_size_,
                           stream_) != cudaSuccess))
        return false;
      for (unsigned int span = n_; span-- > 0;) {
        const unsigned int cells = (n_ - span) * batch_size_;
        const unsigned int blocks = (cells + kBlockSize - 1) / kBlockSize;
        gather_dM_span<Real><<<blocks, kBlockSize, 0, stream_>>>(
          S_.get(),
          dB_.get(),
          dM_.get(),
          d_sequence_.get(),
          d_sequence2_.get(),
          d_scale_.get(),
          n_,
          batch_size_,
          span,
          d_params_.get());
        gather_dS_span<Real><<<blocks, kBlockSize, 0, stream_>>>(
          M_.get(),
          dB_.get(),
          dM_.get(),
          dS_.get(),
          dU_ring_.get(),
          d_sequence_.get(),
          d_sequence2_.get(),
          d_scale_.get(),
          d_mlbase_.get(),
          n_,
          batch_size_,
          span,
          d_params_.get());
        gather_dB_span<Real><<<blocks, kBlockSize, 0, stream_>>>(
          q5_.get(),
          q3_.get(),
          roots_.get(),
          dS_.get(),
          dB_.get(),
          d_sequence_.get(),
          B_.get(),
          probabilities_.get(),
          paired_sums_.get(),
          valid_flags_.get(),
          output_stride_,
          d_sequence2_.get(),
          d_scale_.get(),
          n_,
          batch_size_,
          span,
          d_params_.get());
      }

      const size_t paired_count = static_cast<size_t>(n_ + 1) * batch_size_;
      validate_paired_sums<Real>
        <<<(paired_count + kBlockSize - 1) / kBlockSize,
            kBlockSize,
            0,
            stream_>>>(paired_sums_.get(),
                       valid_flags_.get(),
                       n_,
                       batch_size_);
    }

    if ((cudaStreamEndCapture(stream_, &graph_) != cudaSuccess) ||
        (cudaGraphInstantiate(&graph_exec_,
                              graph_,
                              nullptr,
                              nullptr,
                              0) != cudaSuccess))
      return false;

    return true;
  }

  unsigned int n_;
  unsigned int batch_size_;
  bool with_bpp_;
  int device_;
  size_t matrix_count_;
  size_t sequence_count_;
  size_t ring_count_;
  size_t output_stride_;
  size_t output_count_;
  cudaStream_t stream_;
  cudaGraph_t graph_;
  cudaGraphExec_t graph_exec_;
  short *host_sequence_;
  short *host_sequence2_;
  char *host_characters_;
  Real *host_scale_;
  Real *host_mlbase_;
  GpuPfParams<Real> *host_params_;
  Real *host_roots_;
  FLT_OR_DBL *host_probabilities_;
  int *host_valid_flags_;
  bool ready_;
  DeviceBuffer<short> d_sequence_;
  DeviceBuffer<short> d_sequence2_;
  DeviceBuffer<char> d_characters_;
  DeviceBuffer<Real> d_scale_;
  DeviceBuffer<Real> d_mlbase_;
  DeviceBuffer<GpuPfParams<Real>> d_params_;
  DeviceBuffer<Real> B_, S_, M_;
  DeviceBuffer<Real> U_ring_, M2_ring_;
  DeviceBuffer<Real> q5_, q3_, roots_;
  DeviceBuffer<Real> dB_, dS_, dM_, dU_ring_;
  DeviceBuffer<Real> paired_sums_;
  DeviceBuffer<int> valid_flags_;
  DeviceBuffer<FLT_OR_DBL> probabilities_;
};


template <typename Real>
bool
run_bucket_persistent(vrna_fold_compound_t       **fc,
                      const std::vector<size_t>  &bucket,
                      unsigned char              *handled,
                      float                      *energies,
                      bool                       with_bpp)
{
  const unsigned int n = fc[bucket.front()]->length;
  const unsigned int batch_size = static_cast<unsigned int>(bucket.size());
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return false;

  thread_local std::unique_ptr<PersistentReducedPlan<Real>> plan;
  if ((!plan) || (!plan->matches(n, batch_size, with_bpp, device)))
    plan = std::make_unique<PersistentReducedPlan<Real>>(n,
                                                        batch_size,
                                                        with_bpp,
                                                        device);

  if ((!plan->ready()) || (!plan->execute(fc, bucket)))
    return false;

  const Real *host_roots = plan->roots();
  const FLT_OR_DBL *host_probabilities = plan->probabilities();
  for (unsigned int b = 0; b < batch_size; b++) {
  const int *host_valid_flags = plan->valid_flags();
  const size_t output_stride = plan->output_stride();
    const double root = static_cast<double>(host_roots[b]);
    bool valid = std::isfinite(root) && (root > 0.);
    if (with_bpp)
      valid = valid && (host_valid_flags[b] != 0);

    if (!valid)
      continue;

    vrna_fold_compound_t *current = fc[bucket[b]];
    energies[bucket[b]] = static_cast<float>(
      (-std::log(root) - n * std::log(current->exp_params->pf_scale)) *
      current->exp_params->kT / 1000.);

    if (with_bpp)
      std::memcpy(current->exp_matrices->probs,
                  host_probabilities + static_cast<size_t>(b) * output_stride,
                  sizeof(FLT_OR_DBL) * output_stride);
    handled[bucket[b]] = 1;
  }

  return true;
}


template <typename Real>
bool
run_bucket_blocked_mode(vrna_fold_compound_t       **fc,
                        const std::vector<size_t>  &bucket,
                        unsigned char              *handled,
                        float                      *energies,
                        bool                       with_bpp,
                        BlockedGemmMode            gemm_mode)
{
  const unsigned int n = fc[bucket.front()]->length;
  const unsigned int batch_size = static_cast<unsigned int>(bucket.size());
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess)
    return false;

  using Plan = PersistentBlockedPlan<Real, kPfTileSize>;
  thread_local std::unique_ptr<Plan> native_plan;
  thread_local std::unique_ptr<Plan> emulated_plan;
  std::unique_ptr<Plan> &plan = (gemm_mode == BlockedGemmMode::emulated) ?
                                emulated_plan : native_plan;
  if ((!plan) ||
      (!plan->matches(n, batch_size, with_bpp, device, gemm_mode)))
    plan = std::make_unique<Plan>(n,
                                  batch_size,
                                  with_bpp,
                                  device,
                                  gemm_mode);

  if (!plan->ready()) {
    const char *profile = std::getenv("VRNA_CUDA_PF_PROFILE");
    if (profile && (std::strcmp(profile, "1") == 0))
      std::fprintf(stderr,
                   "CUDA PF blocked plan initialization failed: %s\n",
                   cudaGetErrorString(cudaGetLastError()));
    return false;
  }
  if (!plan->execute(fc, bucket)) {
    const char *profile = std::getenv("VRNA_CUDA_PF_PROFILE");
    if (profile && (std::strcmp(profile, "1") == 0))
      std::fprintf(stderr,
                   "CUDA PF blocked plan execution failed: %s\n",
                   cudaGetErrorString(cudaGetLastError()));
    return false;
  }

  const Real *host_roots = plan->roots();
  const FLT_OR_DBL *host_probabilities = plan->probabilities();
  const int *host_valid_flags = plan->valid_flags();
  const size_t output_stride = plan->output_stride();
  for (unsigned int b = 0; b < batch_size; b++) {
    const double root = static_cast<double>(host_roots[b]);
    bool valid = std::isfinite(root) && (root > 0.);
    if (with_bpp)
      valid = valid && (host_valid_flags[b] != 0);
    if (!valid)
      continue;

    vrna_fold_compound_t *current = fc[bucket[b]];
    energies[bucket[b]] = static_cast<float>(
      (-std::log(root) - n * std::log(current->exp_params->pf_scale)) *
      current->exp_params->kT / 1000.);
    if (with_bpp)
      std::memcpy(current->exp_matrices->probs,
                  host_probabilities + static_cast<size_t>(b) * output_stride,
                  sizeof(FLT_OR_DBL) * output_stride);
    handled[bucket[b]] = 1;
  }

  return true;
}


template <typename Real>
bool
run_bucket_blocked(vrna_fold_compound_t       **fc,
                   const std::vector<size_t>  &bucket,
                   unsigned char              *handled,
                   float                      *energies,
                   bool                       with_bpp)
{
  const char *setting = std::getenv("VRNA_CUDA_PF_GEMM");
  const bool emulated = setting && (std::strcmp(setting, "emulated") == 0);
  const bool automatic = setting && (std::strcmp(setting, "auto") == 0);
  if (emulated)
    return run_bucket_blocked_mode<Real>(fc,
                                         bucket,
                                         handled,
                                         energies,
                                         with_bpp,
                                         BlockedGemmMode::emulated);
  if (!automatic)
    return run_bucket_blocked_mode<Real>(fc,
                                         bucket,
                                         handled,
                                         energies,
                                         with_bpp,
                                         BlockedGemmMode::native);

  const bool launched = run_bucket_blocked_mode<Real>(fc,
                                                       bucket,
                                                       handled,
                                                       energies,
                                                       with_bpp,
                                                       BlockedGemmMode::emulated);
  std::vector<size_t> retry;
  retry.reserve(bucket.size());
  for (size_t input : bucket)
    if ((!launched) || (!handled[input]))
      retry.push_back(input);
  if (retry.empty())
    return true;

  return run_bucket_blocked_mode<Real>(fc,
                                       retry,
                                       handled,
                                       energies,
                                       with_bpp,
                                       BlockedGemmMode::native);
}


template <typename Real>
bool
run_bucket_reduced(vrna_fold_compound_t       **fc,
                   const std::vector<size_t>  &bucket,
                   unsigned char              *handled,
                   float                      *energies,
                   bool                       with_bpp)
{
  const unsigned int n = fc[bucket.front()]->length;
  const unsigned int batch_size = static_cast<unsigned int>(bucket.size());
  const size_t triangular = static_cast<size_t>(n) * (n + 1) / 2;
  const size_t matrix_count = triangular * batch_size;
  const size_t sequence_count = static_cast<size_t>(n + 2) * batch_size;
  const size_t forward_ring_count = static_cast<size_t>(2) * n * batch_size;

  std::vector<short> host_sequence(sequence_count);
  std::vector<short> host_sequence2(sequence_count);
  std::vector<char> host_characters(sequence_count, 0);
  std::vector<Real> host_scale(n + 1);
  std::vector<Real> host_mlbase(n + 1);
  const GpuPfParams<Real> host_params =
    make_gpu_params<Real>(fc[bucket.front()]->exp_params);

  for (unsigned int b = 0; b < batch_size; b++) {
    const vrna_fold_compound_t *current = fc[bucket[b]];
    for (unsigned int position = 0; position <= n + 1; position++) {
      host_sequence[static_cast<size_t>(position) * batch_size + b] =
        current->sequence_encoding[position];
      host_sequence2[static_cast<size_t>(position) * batch_size + b] =
        current->sequence_encoding2[position];
      if ((position >= 1) && (position <= n))
        host_characters[static_cast<size_t>(position) * batch_size + b] =
          current->sequence[position - 1];
    }
  }

  for (unsigned int i = 0; i <= n; i++) {
    host_scale[i] =
      static_cast<Real>(fc[bucket.front()]->exp_matrices->scale[i]);
    host_mlbase[i] =
      static_cast<Real>(fc[bucket.front()]->exp_matrices->expMLbase[i]);
  }

  DeviceBuffer<short> d_sequence;
  DeviceBuffer<short> d_sequence2;
  DeviceBuffer<char> d_characters;
  DeviceBuffer<Real> d_scale;
  DeviceBuffer<Real> d_mlbase;
  DeviceBuffer<GpuPfParams<Real>> d_params;
  DeviceBuffer<Real> B, S, M;
  DeviceBuffer<Real> U_ring, M2_ring;
  DeviceBuffer<Real> q5, q3, roots;
  DeviceBuffer<Real> dB, dS, dM;
  DeviceBuffer<Real> dU_ring;
  DeviceBuffer<FLT_OR_DBL> probabilities;
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  if ((!d_sequence.allocate(sequence_count)) ||
      (!d_sequence2.allocate(sequence_count)) ||
      (!d_characters.allocate(sequence_count)) ||
      (!d_scale.allocate(n + 1)) ||
      (!d_mlbase.allocate(n + 1)) ||
      (!d_params.allocate(1)) ||
      (!B.allocate(matrix_count)) ||
      (!S.allocate(matrix_count)) ||
      (!M.allocate(matrix_count)) ||
      (!U_ring.allocate(forward_ring_count)) ||
      (!M2_ring.allocate(forward_ring_count)) ||
      (!q5.allocate(sequence_count)) ||
      (!q3.allocate(sequence_count)) ||
      (!roots.allocate(batch_size)) ||
      (with_bpp &&
       ((!dB.allocate(matrix_count)) ||
        (!dS.allocate(matrix_count)) ||
        (!dM.allocate(matrix_count)) ||
        (!dU_ring.allocate(forward_ring_count)) ||
        (!probabilities.allocate(matrix_count)))))
    return false;

  if ((cudaMemcpyAsync(d_sequence.get(),
                       host_sequence.data(),
                       sizeof(short) * sequence_count,
                       cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpyAsync(d_sequence2.get(),
                       host_sequence2.data(),
                       sizeof(short) * sequence_count,
                       cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpyAsync(d_characters.get(),
                       host_characters.data(),
                       sizeof(char) * sequence_count,
                       cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpyAsync(d_scale.get(),
                       host_scale.data(),
                       sizeof(Real) * (n + 1),
                       cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpyAsync(d_mlbase.get(),
                       host_mlbase.data(),
                       sizeof(Real) * (n + 1),
                       cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemcpyAsync(d_params.get(),
                       &host_params,
                       sizeof(host_params),
                       cudaMemcpyHostToDevice) != cudaSuccess) ||
      (cudaMemsetAsync(B.get(), 0, sizeof(Real) * matrix_count) != cudaSuccess) ||
      (cudaMemsetAsync(S.get(), 0, sizeof(Real) * matrix_count) != cudaSuccess) ||
      (cudaMemsetAsync(M.get(), 0, sizeof(Real) * matrix_count) != cudaSuccess) ||
      (cudaMemsetAsync(U_ring.get(),
                       0,
                       sizeof(Real) * forward_ring_count) != cudaSuccess) ||
      (cudaMemsetAsync(M2_ring.get(),
                       0,
                       sizeof(Real) * forward_ring_count) != cudaSuccess) ||
      (cudaMemsetAsync(q5.get(), 0, sizeof(Real) * sequence_count) != cudaSuccess) ||
      (cudaMemsetAsync(q3.get(), 0, sizeof(Real) * sequence_count) != cudaSuccess))
    return false;

  if ((cudaStreamSynchronize(nullptr) != cudaSuccess) ||
      (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) ||
      (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess))
    return false;

  for (unsigned int span = 0; span < n; span++) {
    const unsigned int cells = (n - span) * batch_size;
    const unsigned int blocks = (cells + kBlockSize - 1) / kBlockSize;
    compute_pair_aux_reduced_span<Real><<<blocks, kBlockSize, 0, stream>>>(
      B.get(),
      S.get(),
      U_ring.get(),
      M2_ring.get(),
      d_sequence.get(),
      d_sequence2.get(),
      d_characters.get(),
      d_scale.get(),
      d_mlbase.get(),
      n,
      batch_size,
      span,
      d_params.get());
    compute_multibranch_reduced_span<Real><<<blocks, kBlockSize, 0, stream>>>(
      S.get(),
      M.get(),
      U_ring.get(),
      M2_ring.get(),
      n,
      batch_size,
      span);
  }

  compute_exterior_vectors<Real><<<batch_size, kBlockSize, 0, stream>>>(
    B.get(),
    q5.get(),
    q3.get(),
    roots.get(),
    d_sequence.get(),
    d_sequence2.get(),
    d_scale.get(),
    n,
    batch_size,
    d_params.get());

  if (with_bpp) {
    if ((cudaMemsetAsync(dB.get(),
                         0,
                         sizeof(Real) * matrix_count,
                         stream) != cudaSuccess) ||
        (cudaMemsetAsync(dS.get(),
                         0,
                         sizeof(Real) * matrix_count,
                         stream) != cudaSuccess) ||
        (cudaMemsetAsync(dM.get(),
                         0,
                         sizeof(Real) * matrix_count,
                         stream) != cudaSuccess) ||
        (cudaMemsetAsync(dU_ring.get(),
                         0,
                         sizeof(Real) * forward_ring_count,
                         stream) != cudaSuccess))
      return false;

    for (unsigned int span = n; span-- > 0;) {
      const unsigned int cells = (n - span) * batch_size;
      const unsigned int blocks = (cells + kBlockSize - 1) / kBlockSize;
      gather_dM_span<Real><<<blocks, kBlockSize, 0, stream>>>(
        S.get(),
        dB.get(),
        dM.get(),
        d_sequence.get(),
        d_sequence2.get(),
        d_scale.get(),
        n,
        batch_size,
        span,
        d_params.get());
      gather_dS_span<Real><<<blocks, kBlockSize, 0, stream>>>(
        M.get(),
        dB.get(),
        dM.get(),
        dS.get(),
        dU_ring.get(),
        d_sequence.get(),
        d_sequence2.get(),
        d_scale.get(),
        d_mlbase.get(),
        n,
        batch_size,
        span,
        d_params.get());
      gather_dB_span<Real><<<blocks, kBlockSize, 0, stream>>>(
        q5.get(),
        q3.get(),
        roots.get(),
        dS.get(),
        dB.get(),
        d_sequence.get(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0,
        d_sequence2.get(),
        d_scale.get(),
        n,
        batch_size,
        span,
        d_params.get());
    }

    finalize_probabilities<Real>
      <<<(matrix_count + kBlockSize - 1) / kBlockSize,
          kBlockSize,
          0,
          stream>>>(B.get(),
                    dB.get(),
                    probabilities.get(),
                    matrix_count);
  }

  if ((cudaStreamEndCapture(stream, &graph) != cudaSuccess) ||
      (cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0) != cudaSuccess) ||
      (cudaGraphLaunch(graph_exec, stream) != cudaSuccess))
    return false;

  std::vector<Real> host_roots(batch_size);
  std::vector<FLT_OR_DBL> host_probabilities(with_bpp ? matrix_count : 0);
  if ((cudaMemcpyAsync(host_roots.data(),
                       roots.get(),
                       sizeof(Real) * batch_size,
                       cudaMemcpyDeviceToHost,
                       stream) != cudaSuccess) ||
      (with_bpp &&
       (cudaMemcpyAsync(host_probabilities.data(),
                        probabilities.get(),
                        sizeof(FLT_OR_DBL) * matrix_count,
                        cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess)) ||
      (cudaStreamSynchronize(stream) != cudaSuccess))
    return false;

  (void)cudaGraphExecDestroy(graph_exec);
  (void)cudaGraphDestroy(graph);
  (void)cudaStreamDestroy(stream);

  for (unsigned int b = 0; b < batch_size; b++) {
    const double root = static_cast<double>(host_roots[b]);
    bool valid = std::isfinite(root) && (root > 0.);
    if (with_bpp && valid) {
      std::vector<double> paired_sum(n + 1, 0.);
      for (unsigned int span = 0; span < n && valid; span++)
        for (unsigned int i = 1; i + span <= n; i++) {
          const unsigned int j = i + span;
          const double probability =
            host_probabilities[pf_index(span, i, b, n, batch_size)];
          if ((!std::isfinite(probability)) ||
              (probability < -5.e-5) ||
              (probability > 1.0005)) {
            valid = false;
            break;
          }
          if (i != j) {
            paired_sum[i] += probability;
            paired_sum[j] += probability;
          }
        }

      for (unsigned int i = 1; i <= n && valid; i++)
        if (paired_sum[i] > 1.0005)
          valid = false;
    }

    if (!valid)
      continue;

    vrna_fold_compound_t *current = fc[bucket[b]];
    energies[bucket[b]] = static_cast<float>(
      (-std::log(root) - n * std::log(current->exp_params->pf_scale)) *
      current->exp_params->kT / 1000.);

    if (with_bpp) {
      FLT_OR_DBL *destination = current->exp_matrices->probs;
      for (unsigned int i = 1; i < n; i++)
        for (unsigned int j = i + 1; j <= n; j++)
          destination[current->iindx[i] - j] =
            host_probabilities[pf_index(j - i, i, b, n, batch_size)];
    }
    handled[bucket[b]] = 1;
  }

  return true;
}

template <typename Real>
bool
run_bucket(vrna_fold_compound_t       **fc,
           const std::vector<size_t>  &bucket,
           unsigned char              *handled,
           float                      *energies,
           bool                       with_bpp)
{
  const char *engine = std::getenv("VRNA_CUDA_PF_ENGINE");
  if (engine && (std::strcmp(engine, "blocked") == 0))
    return run_bucket_blocked<Real>(fc, bucket, handled, energies, with_bpp);

  const char *reference = std::getenv("VRNA_CUDA_PF_REFERENCE_DAG");
  if ((engine && (std::strcmp(engine, "reference") == 0)) ||
      (reference && (std::strcmp(reference, "1") == 0)))
    return run_bucket_reference<Real>(fc, bucket, handled, energies, with_bpp);

  const char *transient = std::getenv("VRNA_CUDA_PF_TRANSIENT_PLAN");
  if (transient && (std::strcmp(transient, "1") == 0))
    return run_bucket_reduced<Real>(fc, bucket, handled, energies, with_bpp);

  /* Keep the already validated reduced engine as the default until the
   * blocked path passes the full randomized and performance gates. */
  return run_bucket_persistent<Real>(fc, bucket, handled, energies, with_bpp);
}



bool
run_bucket_selected(vrna_fold_compound_t       **fc,
                    const std::vector<size_t>  &bucket,
                    unsigned char              *handled,
                    float                      *energies,
                    bool                       with_bpp)
{
  const char *precision = std::getenv("VRNA_CUDA_PF_PRECISION");
  const bool fp32 = precision && (std::strcmp(precision, "fp32") == 0);
  const bool automatic = precision && (std::strcmp(precision, "auto") == 0);

  if (fp32)
    return run_bucket<float>(fc, bucket, handled, energies, with_bpp);

  if (automatic) {
    (void)run_bucket<float>(fc, bucket, handled, energies, with_bpp);
    std::vector<size_t> retry;
    retry.reserve(bucket.size());
    for (size_t input : bucket)
      if (!handled[input])
        retry.push_back(input);

    if (retry.empty())
      return true;

    return run_bucket<double>(fc, retry, handled, energies, with_bpp);
  }

  return run_bucket<double>(fc, bucket, handled, energies, with_bpp);
}


size_t
single_device_chunk_limit(unsigned int n,
                          size_t       requested,
                          bool         with_bpp)
{
  const char *engine = std::getenv("VRNA_CUDA_PF_ENGINE");
  if ((!engine) || (std::strcmp(engine, "blocked") != 0))
    return requested;

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess)
    return requested;

  const size_t tiles = (n + kPfTileSize - 1U) / kPfTileSize;
  const size_t matrix_dim = tiles * kPfTileSize + 1U;
  const size_t cells = matrix_dim * matrix_dim;
  const size_t real_matrices = with_bpp ? 10U : 5U;
  const size_t probability_cells = with_bpp ?
    static_cast<size_t>(n + 1U) * (n + 2U) / 2U : 0U;
  const size_t per_sequence = real_matrices * cells * sizeof(double) +
                              cells * sizeof(unsigned char) +
                              probability_cells * sizeof(FLT_OR_DBL) +
                              static_cast<size_t>(n + 2U) *
                                (2U * sizeof(short) + sizeof(char) +
                                 2U * sizeof(double));
  const size_t reserve = std::min(free_bytes / 4U,
                                  static_cast<size_t>(4U) * 1024U * 1024U * 1024U);
  if ((per_sequence == 0) || (free_bytes <= reserve + kCublasWorkspaceBytes))
    return 1U;
  const size_t usable = free_bytes - reserve - kCublasWorkspaceBytes;
  return std::max<size_t>(1U, std::min(requested, usable / per_sequence));
}
}  // namespace


extern "C" int
vrna_cuda_pf_batch(vrna_fold_compound_t **fc,
                   size_t               count,
                   unsigned char        *handled,
                   float                *energies,
                   unsigned int         flags)
{
  if ((count > 0) && ((!fc) || (!handled) || (!energies)))
    return 0;

  std::memset(handled, 0, count);
  const bool with_bpp = (flags & VRNA_PF_BATCH_BPP_DENSE) != 0;
  int selected_device = -1;
  if (cudaGetDevice(&selected_device) != cudaSuccess)
    return 0;
  std::vector<unsigned char> assigned(count, 0);
  std::vector<unsigned char> can_use(count, 0);
  const unsigned int validation_threads =
    std::min(static_cast<unsigned int>(count),
             std::max(1U, std::thread::hardware_concurrency()));
  std::atomic<size_t> next_input{0};
  std::vector<std::thread> validation_workers;
  validation_workers.reserve(validation_threads);
  for (unsigned int worker = 0; worker < validation_threads; worker++)
    validation_workers.emplace_back([&]() {
      while (true) {
        const size_t input = next_input.fetch_add(1, std::memory_order_relaxed);
        if (input >= count)
          break;
        can_use[input] = eligible(fc[input], with_bpp) ? 1 : 0;
      }
    });
  for (std::thread &worker : validation_workers)
    worker.join();

  for (size_t i = 0; i < count; i++) {
    if (assigned[i] || !can_use[i])
      continue;

    std::vector<size_t> bucket;
    for (size_t j = i; j < count; j++)
      if ((!assigned[j]) && can_use[j] && same_bucket(fc[i], fc[j])) {
        assigned[j] = 1;
        bucket.push_back(j);
      }

    const size_t chunk_limit = single_device_chunk_limit(fc[bucket.front()]->length,
                                                         bucket.size(),
                                                         with_bpp);
    for (size_t begin = 0; begin < bucket.size(); begin += chunk_limit) {
      const size_t end = std::min(bucket.size(), begin + chunk_limit);
      std::vector<size_t> chunk(bucket.begin() + begin, bucket.begin() + end);
      int current_device = -1;
      if ((cudaGetDevice(&current_device) != cudaSuccess) ||
          (current_device != selected_device) ||
          (!run_bucket_selected(fc, chunk, handled, energies, with_bpp)))
        (void)cudaGetLastError();
    }
  }


  return 1;
}


extern "C" int
vrna_cuda_pf_selected_device(void)
{
  int device = -1;
  return (cudaGetDevice(&device) == cudaSuccess) ? device : -1;
}
