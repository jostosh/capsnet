// #include "capsmatmul_functor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

// #define EIGEN_USE_GPU

namespace tensorflow
{
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

using shape_inference::ShapeHandle;
using shape_inference::InferenceContext;
using shape_inference::DimensionHandle;

REGISTER_OP("CapsMatMulGrad")
    .Input("grad: T")
    .Input("input: T")
    .Input("weights: T")
    .Output("grad_input: T")
    .Output("grad_weights: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* ctx) {
      ShapeHandle in_shape, weights_shape, grad_shape;
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 4, &grad_shape));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 3, &in_shape));
      TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(2), 4, &weights_shape));

      ShapeHandle grad_in_shape(in_shape);
      ShapeHandle grad_weights_shape(weight_shape);
      ctx->set_output(0, grad_in_shape);
      ctx->set_output(1, grad_weights_shape);

      return Status::OK();
    });

void launch_capsmatmul_grad(
  const GPUDevice& d,
  typename TTypes<float, 3>::ConstTensor input,
  typename TTypes<float, 4>::ConstTensor weights,
  typename TTypes<float, 4>::ConstTensor grad,
  typename TTypes<float, 3>::Tensor grad_input,
  typename TTypes<float, 4>::Tensor grad_weights);


template <typename T>
class CapsMatMulGradOp : public OpKernel
{
 public:
  explicit CapsMatMulGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override
  {
    //--Grab the input tensor - params--//
    const Tensor& grad = ctx->input(0);
    const Tensor& input = ctx->input(1);
    //--Grab the input tensor - indices--//
    const Tensor& weights = ctx->input(2);

    const TensorShape& input_shape(input.shape());
    const TensorShape& weights_shape(weights.shape());

    //--Create an output tensor--//
    Tensor* grad_input = nullptr;
    Tensor* grad_weights = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input_shape, &grad_input));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, weights_shape, &grad_weights));

    auto input_tensor = input.tensor<float, 3>();
    auto weights_tensor = weights.tensor<float, 4>();
    auto grad_tensor = grad.tensor<float, 4>();
    auto grad_input_tensor = grad_input->tensor<float, 3>();
    auto grad_weights_tensor = grad_weights->tensor<float, 4>();
    launch_capsmatmul_grad(
      ctx->eigen_device<GPUDevice>(), input_tensor, weights_tensor, grad_tensor,
      grad_input_tensor, grad_weights_tensor
    );
  }
};


REGISTER_KERNEL_BUILDER(Name("CapsMatMulGrad")
                      .Device(DEVICE_GPU)
                      .TypeConstraint<float>("T"),
                  CapsMatMulGradOp<float>);


}  // namespace tensorflow
