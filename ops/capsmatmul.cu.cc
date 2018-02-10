#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#define BLOCK_SIZE 1024

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

__global__ void CapsMatMulOpKernel(const float* in, const float* weights,
    float* out,
    const int64 o_d0, const int64 o_d1, const int64 o_d2,
    const int64 x_d0, const int64 x_d1,
    const int64 w_d0, const int64 w_d1, const int64 w_d2,
    const int64 in_dim, const int64 output_size)
{
  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    __shared__ float in_shared[BLOCK_SIZE];

    const int64 tid = threadIdx.x;

    // So here we have out[b,ci,cj,e]
    const int64 b = i / o_d0;
    const int64 ci = (i % o_d0) / o_d1;
    const int64 cj = (i % o_d1) / o_d2;
    const int64 e = i % o_d2;

    // Then, we can have a look at computing the array indices for in and W
    int64 in_idx = b * x_d0 + ci * x_d1;
    int64 w_idx = ci * w_d0 + cj * w_d1 + e * w_d2;

    in_shared[tid] = ldg(in + in_idx + tid % in_dim);

    out[i] = static_cast<float>(0);
    int64 shared_offset = tid - (tid % in_dim);
    // Here we must guarantee that other threads have finished their work
    __syncthreads();
    for (int64 v = 0; v < in_dim; ++v)
      out[i] += in_shared[shared_offset++] * ldg(weights + w_idx++);
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
  const int64 o_d2 = out_dim;

  CudaLaunchConfig config = GetCudaLaunchConfig(out.size(), d);
  CapsMatMulOpKernel
    <<<config.block_count, BLOCK_SIZE, 0, d.stream()>>>(
      x.data(), weights.data(), out.data(),
      o_d0, o_d1, o_d2, x_d0, x_d1, w_d0, w_d1, w_d2,
      in_dim, out.size());
}


}  // namespa
#endif
