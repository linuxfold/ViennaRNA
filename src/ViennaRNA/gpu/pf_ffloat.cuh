#ifndef VIENNARNA_GPU_PF_FFLOAT_CUH
#define VIENNARNA_GPU_PF_FFLOAT_CUH

#include <cuda_runtime.h>

#include <cmath>

/* A two-component FP32 expansion.  The representation deliberately has the
 * same size and alignment as double, but every device arithmetic operation is
 * issued on the FP32 pipelines. */
struct __align__(8) Ffloat {
  float hi;
  float lo;

  __host__ __device__ Ffloat() : hi(0.f), lo(0.f) {}
  __host__ __device__ Ffloat(float value) : hi(value), lo(0.f) {}
  __host__ __device__ Ffloat(int value) : hi(static_cast<float>(value)), lo(0.f) {}
  __host__ __device__ Ffloat(float leading, float trailing)
    : hi(leading), lo(trailing)
  {}
  __host__ __device__ explicit Ffloat(double value)
    : hi(static_cast<float>(value)),
      lo(static_cast<float>(value - static_cast<double>(hi)))
  {}

  __host__ __device__ explicit operator float() const
  {
    return hi + lo;
  }

  __host__ __device__ explicit operator double() const
  {
    return static_cast<double>(hi) + static_cast<double>(lo);
  }
};

__host__ __device__ __forceinline__ Ffloat
ff_quick_two_sum(float a,
                 float b)
{
  const float sum = a + b;
  return Ffloat{sum, b - (sum - a)};
}

__host__ __device__ __forceinline__ Ffloat
ff_add(Ffloat a,
       Ffloat b)
{
  const float sum = a.hi + b.hi;
  const float v = sum - a.hi;
  float error = (a.hi - (sum - v)) + (b.hi - v);
  error += a.lo + b.lo;
  return ff_quick_two_sum(sum, error);
}

__host__ __device__ __forceinline__ Ffloat
ff_mul(Ffloat a,
       Ffloat b)
{
  const float product = a.hi * b.hi;
  float error = fmaf(a.hi, b.hi, -product);
  error = fmaf(a.hi, b.lo, error);
  error = fmaf(a.lo, b.hi, error);
  error = fmaf(a.lo, b.lo, error);
  return ff_quick_two_sum(product, error);
}

__host__ __device__ __forceinline__ Ffloat
operator+(Ffloat a,
          Ffloat b)
{
  return ff_add(a, b);
}

__host__ __device__ __forceinline__ Ffloat
operator-(Ffloat a,
          Ffloat b)
{
  return ff_add(a, Ffloat{-b.hi, -b.lo});
}

__host__ __device__ __forceinline__ Ffloat
operator-(Ffloat value)
{
  return Ffloat{-value.hi, -value.lo};
}

__host__ __device__ __forceinline__ Ffloat
operator*(Ffloat a,
          Ffloat b)
{
  return ff_mul(a, b);
}

__host__ __device__ __forceinline__ Ffloat
operator/(Ffloat a,
          Ffloat b)
{
  const float estimate = static_cast<float>(a) / static_cast<float>(b);
  const Ffloat q(estimate);
  const Ffloat residual = a - b * q;
  return ff_add(q, Ffloat(static_cast<float>(residual) /
                          static_cast<float>(b)));
}

__host__ __device__ __forceinline__ Ffloat &operator+=(Ffloat &a, Ffloat b)
{
  a = a + b;
  return a;
}

__host__ __device__ __forceinline__ Ffloat &operator-=(Ffloat &a, Ffloat b)
{
  a = a - b;
  return a;
}

__host__ __device__ __forceinline__ Ffloat &operator*=(Ffloat &a, Ffloat b)
{
  a = a * b;
  return a;
}

__host__ __device__ __forceinline__ Ffloat &operator/=(Ffloat &a, Ffloat b)
{
  a = a / b;
  return a;
}

__host__ __device__ __forceinline__ bool operator==(Ffloat a, Ffloat b)
{
  return (a.hi == b.hi) && (a.lo == b.lo);
}

__host__ __device__ __forceinline__ bool operator!=(Ffloat a, Ffloat b)
{
  return !(a == b);
}

__host__ __device__ __forceinline__ bool operator<(Ffloat a, Ffloat b)
{
  return (a.hi < b.hi) || ((a.hi == b.hi) && (a.lo < b.lo));
}

__host__ __device__ __forceinline__ bool operator>(Ffloat a, Ffloat b)
{
  return b < a;
}

__host__ __device__ __forceinline__ bool operator<=(Ffloat a, Ffloat b)
{
  return !(b < a);
}

__host__ __device__ __forceinline__ bool operator>=(Ffloat a, Ffloat b)
{
  return !(a < b);
}

__host__ __device__ __forceinline__ bool isfinite(Ffloat value)
{
  return ::isfinite(value.hi) && ::isfinite(value.lo);
}

/* CUDA shuffle intrinsics do not provide aggregate overloads. */
__device__ __forceinline__ Ffloat
__shfl_down_sync(unsigned int mask,
                 Ffloat      value,
                 unsigned int delta,
                 int         width = warpSize)
{
  return Ffloat{::__shfl_down_sync(mask, value.hi, delta, width),
                ::__shfl_down_sync(mask, value.lo, delta, width)};
}

__device__ __forceinline__ Ffloat
__shfl_sync(unsigned int mask,
            Ffloat      value,
            int         source_lane,
            int         width = warpSize)
{
  return Ffloat{::__shfl_sync(mask, value.hi, source_lane, width),
                ::__shfl_sync(mask, value.lo, source_lane, width)};
}

#endif
