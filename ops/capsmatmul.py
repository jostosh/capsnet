import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

op_module = tf.load_op_library('ops/capsmatmul_op.so')


@ops.RegisterGradient("CapsMatMul")
def _capsmatmul_grad(op, grad):

    return op_module.caps_mat_mul_grad(grad, op.inputs[0], op.inputs[1])


def capsmatmul(x, weights):

    return op_module.caps_mat_mul(x, weights)