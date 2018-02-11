import tensorflow as tf
from tensorflow.python.framework import ops

op_module = tf.load_op_library('ops/capsuleprediction_op.so')


@ops.RegisterGradient("CapsulePrediction")
def _capsule_prediction_grad(op, grad):
    """ Computes gradient for capsule prediction operation """
    return op_module.capsule_prediction_grad(grad, op.inputs[0], op.inputs[1])


def capsule_prediction(x, weights):
    """ Computes capsule prediction """
    return op_module.capsule_prediction(x, weights)
