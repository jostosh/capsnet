#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_GATHER_COLUMNS_FUNCTOR_GPU_CU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_GATHER_COLUMNS_FUNCTOR_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "gather_columns_functor.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename IndT>
__global__ void CapsMatmulOpKernel(const T* in, const T* weights, T* out,
                                   const int64 batch_size,
                                   const int64 in_caps,
                                   const int64 out_caps,
                                   const int64 in_dim,
                                   const int64 out_dim,
                                   const int64 output_size)
{
  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    // Total matmuls
    const int64 m = in_caps * out_caps;
    // Size first dim
    const int64 w_d0 = m * in_dim;
    const int64 x_d0 = in_caps * in_dim;
    const int64 o_d0 = m * out_dim;
    // Second dim
    const int64 w_d1 = w_d0 / (out_caps * in_dim);
    const int64 x_d1 = x_d0 / in_dim;
    const int64 o_d1 = w_d1 / (out_caps * out_dim);
    // Third dim
    const int64 w_d2 = w_d1 / in_dim;
    const int64 x_d2 = in_dim;
    const int64 o_d2 = o_d2 / out_dim;
    // Fourth dim
    const int64 w_d3 = in_dim;
    const int64 x_d3 = 1;         // or zero?
    const int64 o_d3 = out_dim;

    // So here we have O[b,i,j,e]
    const int64 b = i / o_d0;
    const int64 ci = (i % o_d0) / o_d1;
    const int64 cj = (i % o_d1) / o_d2;
    const int64 e = i % o_d3;

    // Then, we can have a look at computing the array indices for in and W
    const int64 in_idx = b * x_d0 + ci * x_d1;
    const int64 w_idx = b * w_d0 + ci * w_d1 + cj * w_d2;

    // TODO load this in shared memory?
    out[i] = static_cast<T>(0);
    for (int64 v = 0; v < in_dim; ++v)
    {
      out[i] += ldg(in + in_idx + v) * ldg(weights + w_idx + v);
    }
  }
}

namespace functor
{
template <typename T, typename IndT>
struct CapsMatMul<GPUDevice, T, IndT>
{
  int64 operator()(const GPUDevice& d, typename TTypes<T>::ConstTensor x,
                   typename TTypes<IndT>::ConstTensor weights,
                   typename TTypes<IndT>::Tensor out)
  {
    const int64 batch_size  = x.dimension(0);
    const int64 in_caps     = x.dimsension(1);
    const int64 in_dim      = x.dimension(2);
    const int64 out_dim     = weights.dimension(3);
    const int64 out_caps    = weights.dimension(1);

    CudaLaunchConfig config = GetCudaLaunchConfig(output_size, d);
    CapsMatMul<T, IndT>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        x.data(), weights.data(), out.data(), in_caps, out_caps,
        in_dim, out_dim, out.size());

    return -1;
  }
};
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_GATHER_COLUMNS_FUNCTOR_GPU_CU_H_
