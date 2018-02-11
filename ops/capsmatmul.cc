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

REGISTER_OP("CapsMatMul")
.Input("input: T")
.Input("weights: T")
.Output("output: T")
.Attr("T: type")
.SetShapeFn([](InferenceContext* ctx) {
    // Get shapes and ensure correct dimensionality
    ShapeHandle in_shape;
    ShapeHandle weights_shape;
    TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(0), 3, &in_shape));
    TF_RETURN_IF_ERROR(ctx->WithRank(ctx->input(1), 4, &weights_shape));

    // Construct and set the output shape
    DimensionHandle out_d0, out_d1, out_d2, out_d3;
    std::vector<DimensionHandle> out_dims;
    out_dims.push_back(ctx->MakeDim(ctx->Dim(ctx->input(0), 0)));
    out_dims.push_back(ctx->MakeDim(ctx->Dim(ctx->input(1), 0)));
    out_dims.push_back(ctx->MakeDim(ctx->Dim(ctx->input(1), 1)));
    out_dims.push_back(ctx->MakeDim(ctx->Dim(ctx->input(1), 2)));
    ShapeHandle out_shape = ctx->MakeShape(out_dims);
    ctx->set_output(0, out_shape);

    return Status::OK();
});

void launch(
  const GPUDevice& d,
  typename TTypes<float, 3>::ConstTensor x,
  typename TTypes<float, 4>::ConstTensor weights,
  typename TTypes<float, 4>::Tensor out);

class CapsMatMulOp : public OpKernel
{
 public:
  explicit CapsMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override
  {
    // Get inputs
    const Tensor& input = ctx->input(0);
    const Tensor& weights = ctx->input(1);

    // Setup output shape
    const TensorShape& input_shape(input.shape());
    TensorShape output_shape(weights.shape());
    output_shape.InsertDim(0, input_shape.dim_size(0));
    output_shape.RemoveDim(4);

    // Allocate output tensor
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // Get the Eigen tensors and pass them on the launcher
    auto input_tensor   = input.tensor<float, 3>();
    auto weights_tensor = weights.tensor<float, 4>();
    auto output_tensor  = output->tensor<float, 4>();
    launch(ctx->eigen_device<GPUDevice>(), input_tensor, weights_tensor,
      output_tensor);
  }
};


REGISTER_KERNEL_BUILDER(
        Name("CapsMatMul")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
        CapsMatMulOp)

}  // namespace tensorflow
