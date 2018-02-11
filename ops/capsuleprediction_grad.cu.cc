#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

__global__ void capsulePredictionInputGradKernel(
  const float* grad, const float* weights, float* grad_input,
  const int64 w_d0,
  const int64 x_d0, const int64 x_d1,
  const int64 o_d0, const int64 o_d1,
  const int64 out_caps,
  const int64 out_dim,
  const int64 in_dim,
  const int64 output_size)
{
  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    // So here we have in_grad[b,ci,e]
    const int64 b     = i / x_d0;
    const int64 ci    = (i % x_d0) / x_d1;
    const int64 e_in  = i % x_d1;

    // Then, we can have a look at computing the array indices for in and W
    int64 w_idx       = ci * w_d0 + e_in;
    int64 grad_idx    = b * o_d0 + ci * o_d1;

    // Initialize result
    float result      = 0.0;
    // Iterate over cj and e_out, we already have the other indices
    for (int cj = 0; cj < out_caps; ++cj)
    {
      for (int e_out = 0; e_out < out_dim; ++e_out)
      {
        // Next element of grad can be found by incrementing grad_idx
        result  += ldg(grad + grad_idx++) * ldg(weights + w_idx);
        // Next element of weights can be found by going to the next output
        // capsule element, meaning that we add in_dim to w_idx
        w_idx   += in_dim;
      }
    }
    // Write the result
    grad_input[i] = result;
  }
}


__global__ void capsulePredictionWeightsGradKernel(
  const float* grad, const float* input, float* grad_weights,
  const int64 batch_size, const int64 output_size,
  const int64 w_d0, const int64 w_d1, const int64 w_d2,
  const int64 x_d0, const int64 x_d1,
  const int64 o_d0, const int64 o_d1, const int64 o_d2
)
{
  CUDA_1D_KERNEL_LOOP(i, output_size)
  {
    // So here we have w[ci,cj,e_out,e_in]
    const int64 ci    = i / w_d0;
    const int64 cj    = (i % w_d0) / w_d1;
    const int64 e_out = (i % w_d1) / w_d2;
    const int64 e_in  = i % w_d2;

    // Then, we can have a look at computing the array indices for
    // in and grad
    int64 input_idx   = ci * x_d1 + e_in;               // (b == 0)
    int64 grad_idx    = ci * o_d1 + cj * o_d2 + e_out;  // (b == 0)

    // Initilize result
    float result      = 0.0;
    // We only iterate over b, since we have the other indices already
    for (int64 b = 0; b < batch_size; b++)
    {
      result += ldg(grad + grad_idx) * ldg(input + input_idx);
      // Next elements can be found by jumping to the next batch
      input_idx += x_d0;
      grad_idx  += o_d0;
    }
    grad_weights[i] = result;
  }
}


void launchCapsulePredictionGrad(
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

  // Launch input gradient kernel
  CudaLaunchConfig config = GetCudaLaunchConfig(grad_input.size(), d);
  capsulePredictionInputGradKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      grad.data(), weights.data(), grad_input.data(),
      w_d0, x_d0, x_d1, o_d0, o_d1, out_caps, out_dim, in_dim,
      grad_input.size());

  // Launch weight gradient kernel
  config = GetCudaLaunchConfig(grad_weights.size(), d);
  capsulePredictionWeightsGradKernel
    <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      grad.data(), input.data(), grad_weights.data(), batch_size,
      grad_weights.size(), w_d0, w_d1, w_d2, x_d0, x_d1, o_d0, o_d1, o_d2);
}


}  // namespace TensorFlow
#endif
