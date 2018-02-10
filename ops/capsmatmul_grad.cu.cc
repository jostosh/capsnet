#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

__global__ void CapsMatMulGradInputKernel(
  const float* grad, const float* weights, float* grad_input,
  const int64 batch_size,
  const int64 in_caps,
  const int64 out_caps,
  const int64 in_dim,
  const int64 out_dim,
  const int64 output_size)
{
  // Size first dim
  const int64 w_d0 = out_caps * out_dim * in_dim;
  const int64 x_d0 = in_caps * in_dim;
  const int64 o_d0 = in_caps * out_caps * out_dim;

  // Second dim
  const int64 x_d1 = in_dim;
  const int64 o_d1 = out_caps * out_dim;

  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    // So here we have out[b,ci,cj,e]s
    const int64 b = i / x_d0;
    const int64 ci = (i % x_d0) / x_d1;
    const int64 e = i % x_d1;

    // Then, we can have a look at computing the array indices for in and W
    int64 w_idx = ci * w_d0 + e;
    int64 grad_idx = b * o_d0 + ci * o_d1;

    grad_input[i] = static_cast<float>(0);
    for (int cj = 0; cj < out_caps; ++cj)
    {
      for (int e_out = 0; e_out < out_dim; ++e_out)
      {
        grad_input[i] += ldg(grad + grad_idx++) * ldg(weights + w_idx);
        w_idx += in_dim;
      }
    }
  }
}


__global__ void CapsMatMulGradWeightsKernel(
  const float* grad, const float* input, float* grad_weights,
  const int64 batch_size,
  const int64 in_caps,
  const int64 out_caps,
  const int64 in_dim,
  const int64 out_dim,
  const int64 output_size)
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
  const int64 o_d2 = out_dim;

  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    // So here we have out[b,ci,cj,e]s
    const int64 ci = i / w_d0;
    const int64 cj = (i % w_d0) / w_d1;
    const int64 e_out = (i % w_d1) / w_d2;
    const int64 e_in = i % w_d2;

    // Then, we can have a look at computing the array indices for in and W
    int64 input_idx = ci * x_d1 + e_in;
    int64 grad_idx = ci * o_d1 + cj * o_d2 + e_out;

    grad_weights[i] = static_cast<float>(0);
    for (int64 b = 0; b < batch_size; b++)
    {
      grad_weights[i] += ldg(grad + grad_idx) * ldg(input + input_idx);
      input_idx += x_d0;
      grad_idx  += o_d0;
    }
  }
}


void launch_capsmatmul_grad(
  const GPUDevice& d,
  typename TTypes<float, 3>::ConstTensor input,
  typename TTypes<float, 4>::ConstTensor weights,
  typename TTypes<float, 4>::ConstTensor grad,
  typename TTypes<float, 3>::Tensor grad_input,
  typename TTypes<float, 4>::Tensor grad_weights)
{
  const int64 batch_size  = input.dimension(0);
  const int64 in_caps     = input.dimension(1);
  const int64 in_dim      = input.dimension(2);
  const int64 out_dim     = weights.dimension(2);
  const int64 out_caps    = weights.dimension(1);

  CudaLaunchConfig config = GetCudaLaunchConfig(grad_input.size(), d);
  CapsMatMulGradInputKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      grad.data(), weights.data(), grad_input.data(), batch_size, in_caps,
      out_caps, in_dim, out_dim, grad_input.size());

  config = GetCudaLaunchConfig(grad_weights.size(), d);
  CapsMatMulGradWeightsKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      grad.data(), input.data(), grad_weights.data(), batch_size, in_caps,
      out_caps, in_dim, out_dim, grad_weights.size());
}


}  // namespa
#endif
