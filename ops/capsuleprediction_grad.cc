#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow
{
typedef Eigen::GpuDevice GPUDevice;

using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;
using shape_inference::DimensionHandle;

REGISTER_OP("CapsulePredictionGrad")
    .Input("grad: T")
    .Input("input: T")
    .Input("weights: T")
    .Output("grad_input: T")
    .Output("grad_weights: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle in_shape, weights_shape, grad_shape;
      // Ensures we have the right input shapes
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 4, &grad_shape));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 3, &in_shape));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(2), 4, &weights_shape));

      // Initialize shape of gradients and assign them to output shapes
      ShapeHandle grad_in_shape(in_shape);
      ShapeHandle grad_weights_shape(weights_shape);
      ctx->set_output(0, grad_in_shape);
      ctx->set_output(1, grad_weights_shape);

      return Status::OK();
    });

// Forward declaration
void launchCapsulePredictionGrad(
  const GPUDevice& d,
  typename TTypes<float, 3>::ConstTensor input,
  typename TTypes<float, 4>::ConstTensor weights,
  typename TTypes<float, 4>::ConstTensor grad,
  typename TTypes<float, 3>::Tensor grad_input,
  typename TTypes<float, 4>::Tensor grad_weights);

class CapsulePredictionGradOp : public OpKernel
{
 public:
  explicit CapsulePredictionGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override
  {
    // Get the input tensors
    const Tensor& grad = ctx->input(0);
    const Tensor& input = ctx->input(1);
    const Tensor& weights = ctx->input(2);

    // Get the shapes so that we can allocate outputs
    const TensorShape& input_shape(input.shape());
    const TensorShape& weights_shape(weights.shape());

    // Allocate outputs
    Tensor* grad_input = nullptr;
    Tensor* grad_weights = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_shape, &grad_input));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, weights_shape, &grad_weights));

    // Get the Eigen tensors and pass them on to the launch function
    auto input_tensor         = input.tensor<float, 3>();
    auto weights_tensor       = weights.tensor<float, 4>();
    auto grad_tensor          = grad.tensor<float, 4>();
    auto grad_input_tensor    = grad_input->tensor<float, 3>();
    auto grad_weights_tensor  = grad_weights->tensor<float, 4>();
    launchCapsulePredictionGrad(
      ctx->eigen_device<GPUDevice>(), input_tensor, weights_tensor, grad_tensor,
      grad_input_tensor, grad_weights_tensor
    );
  }
};


REGISTER_KERNEL_BUILDER(Name("CapsulePredictionGrad")
                      .Device(DEVICE_GPU)
                      .TypeConstraint<float>("T"),
                  CapsulePredictionGradOp)


}  // namespace tensorflow
