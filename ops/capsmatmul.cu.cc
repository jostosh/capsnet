#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

__global__ void CapsMatMulOpKernel(const float* in, const float* weights,
    float* out,
                                   const int64 batch_size,
                                   const int64 in_caps,
                                   const int64 out_caps,
                                   const int64 in_dim,
                                   const int64 out_dim,
                                   const int64 output_size)
{
  CUDA_1D_KERNEL_LOOP(i, output_size)
  {

    // Size first dim
    const int64 w_d0 = out_caps * out_dim * in_dim;
    const int64 x_d0 = in_caps * in_dim;
    const int64 o_d0 = in_caps * out_caps * out_dim;

    // Second dim
    const int64 w_d1 = out_dim * in_dim;
    const int64 x_d1 = in_dim;
    const int64 o_d1 = out_caps * out_dim;

    // Third dim
    const int64 w_d2 = in_dim;
    // const int64 x_d2 = in_dim;
    const int64 o_d2 = out_dim;

    // Fourth dim
    // const int64 w_d3 = in_dim;
    // const int64 x_d3 = 1;         // or zero?
    // const int64 o_d3 = out_dim;

    // So here we have O[b,i,j,e]
    const int64 b = i / o_d0;
    const int64 ci = (i % o_d0) / o_d1;
    const int64 cj = (i % o_d1) / o_d2;
    const int64 e = i % o_d2;

    // Then, we can have a look at computing the array indices for in and W
    const int64 in_idx = b * x_d0 + ci * x_d1;
    const int64 w_idx = ci * w_d0 + cj * w_d1 + e * w_d2;

    // TODO load this in shared memory?
    out[i] = static_cast<float>(0);
    for (int64 v = 0; v < in_dim; ++v)
    {
      out[i] += ldg(in + in_idx + v) * ldg(weights + w_idx + v);
    }
  }
}


void launch(
  const GPUDevice& d,
  typename TTypes<float, 3>::ConstTensor x,
  typename TTypes<float, 4>::ConstTensor weights,
  typename TTypes<float, 4>::Tensor out)
{
  const int64 batch_size  = x.dimension(0);
  const int64 in_caps     = x.dimension(1);
  const int64 in_dim      = x.dimension(2);
  const int64 out_dim     = weights.dimension(2);
  const int64 out_caps    = weights.dimension(1);

  printf("batch_size %d, in_caps %d, in_dim %d, out_dim %d, out_caps %d\n",
    batch_size, in_caps, in_dim, out_dim, out_caps);

  CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
  CapsMatMulOpKernel<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      x.data(), weights.data(), out.data(), batch_size, in_caps, out_caps,
      in_dim, out_dim, out.size());
}


}  // namespa
#endif
