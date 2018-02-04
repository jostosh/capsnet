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
  typename TTypes<float, 3>::Tensor x,
  typename TTypes<float, 4>::Tensor weights,
  typename TTypes<float, 4>::ConstTensor grad);


template <typename T>
class CapsMatMulGradOp : public OpKernel
{
 public:
  explicit CapsMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override
  {
    //--Grab the input tensor - params--//
    const Tensor& input = ctx->input(0);

    //--Grab the input tensor - indices--//
    const Tensor& weights = ctx->input(1);

    const TensorShape& input_shape(input.shape());
    TensorShape output_shape(weights.shape());
    output_shape.InsertDim(0, input_shape.dim_size(0));
    output_shape.RemoveDim(4);

    //--Create an output tensor--//
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    auto input_tensor = input.tensor<float, 3>();
    auto weights_tensor = weights.tensor<float, 4>();
    auto output_tensor = output->tensor<float, 4>();
    launch(ctx->eigen_device<GPUDevice>(), input_tensor, weights_tensor,
      output_tensor);
  }
};


REGISTER_KERNEL_BUILDER(Name("CapsMatMul")
                      .Device(DEVICE_GPU)
                      .TypeConstraint<float>("T"),
                  CapsMatMulOp<float>);


#undef REGISTER_CAPSMATMUL

}  // namespace tensorflow
