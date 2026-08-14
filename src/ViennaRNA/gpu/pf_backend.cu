#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <limits>
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
      cudaFreeAsync(data_, nullptr);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  bool allocate(size_t count)
  {
    return (count == 0) ||
           (cudaMallocAsync(reinterpret_cast<void **>(&data_),
                            sizeof(T) * count,
                            nullptr) == cudaSuccess);
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
  const char *reference = std::getenv("VRNA_CUDA_PF_REFERENCE_DAG");
  if (reference && (std::strcmp(reference, "1") == 0))
    return run_bucket_reference<Real>(fc, bucket, handled, energies, with_bpp);

  const char *transient = std::getenv("VRNA_CUDA_PF_TRANSIENT_PLAN");
  if (transient && (std::strcmp(transient, "1") == 0))
    return run_bucket_reduced<Real>(fc, bucket, handled, energies, with_bpp);

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
  int visible_devices = 1;
  (void)cudaGetDeviceCount(&visible_devices);
  unsigned int device_limit = 1;
  const char *device_setting = std::getenv("VRNA_CUDA_PF_DEVICES");
  if (device_setting && device_setting[0]) {
    if (std::strcmp(device_setting, "all") == 0)
      device_limit = static_cast<unsigned int>(visible_devices);
    else
      device_limit = static_cast<unsigned int>(std::max(1, std::atoi(device_setting)));
  }
  device_limit = std::min(device_limit,
                          static_cast<unsigned int>(std::max(1, visible_devices)));
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

    const unsigned int active_devices =
      std::min(device_limit, static_cast<unsigned int>(bucket.size()));
    if (active_devices == 1) {
      if (!run_bucket_selected(fc, bucket, handled, energies, with_bpp)) {
        (void)cudaGetLastError();
      }
    } else {
      std::vector<std::vector<size_t>> chunks(active_devices);
      for (size_t entry = 0; entry < bucket.size(); entry++)
        chunks[entry % active_devices].push_back(bucket[entry]);

      std::vector<std::thread> workers;
      workers.reserve(active_devices);
      for (unsigned int device = 0; device < active_devices; device++)
        workers.emplace_back([&, device]() {
          if (cudaSetDevice(static_cast<int>(device)) == cudaSuccess) {
            if (!run_bucket_selected(fc, chunks[device], handled, energies, with_bpp)) {
              (void)cudaGetLastError();
            }
            (void)cudaDeviceSynchronize();
          }
        });

      for (std::thread &worker : workers)
        worker.join();
      (void)cudaSetDevice(0);
    }
  }


  return 1;
}
