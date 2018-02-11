import tensorflow as tf
from ops.capsuleprediction import capsule_prediction
import numpy as np
from parameterized import parameterized
import itertools


class CapsulePredictionOpTest(tf.test.TestCase):

    @staticmethod
    def _numpy_capsule_prediction(x, weights):
        """ Generate the output for x and weights with numpy """
        batch_size, in_caps, in_dim = x.shape
        _, out_caps, out_dim, _ = weights.shape

        out_shape = (batch_size, in_caps, out_caps, out_dim)
        out = np.zeros(out_shape)

        for b in range(batch_size):
            for i in range(in_caps):
                for j in range(out_caps):
                    for c in range(out_dim):
                        out[b, i, j, c] = np.dot(x[b, i], weights[i, j, c])
        return out

    @parameterized.expand([
        (batch_size, in_caps, out_caps, in_dim, out_dim) for
        batch_size, in_caps, out_caps, in_dim, out_dim in
        itertools.product([4, 8], [4, 8], [4, 8], [4, 8], [4, 8])
    ])
    def test_capsule_prediction_op(self, batch_size, in_caps, out_caps, in_dim,
                                   out_dim):
        """ Tests the forward capsmatmul op """
        x = np.random.rand(batch_size, in_caps, in_dim)
        weights = np.random.rand(in_caps, out_caps, out_dim, in_dim)

        truth = self._numpy_capsule_prediction(x, weights)
        with self.test_session() as sess:
            x_ph = tf.placeholder(tf.float32, x.shape)
            w_ph = tf.placeholder(tf.float32, weights.shape)

            ret = capsule_prediction(x_ph, w_ph)
            out = sess.run(ret, {x_ph: x, w_ph: weights})
        self.assertAllClose(truth, out)


    @parameterized.expand([
        (batch_size, in_caps, out_caps, in_dim, out_dim) for
        batch_size, in_caps, out_caps, in_dim, out_dim in
        itertools.product([4, 8], [4, 8], [4, 8], [4, 8], [4, 8])
    ])
    def test_capsule_prediction_weights_grad(self, batch_size, in_caps, out_caps, in_dim, out_dim):
        """ Tests gradient of output w.r.t. weights """
        x = np.random.rand(batch_size, in_caps, in_dim)
        weights = np.random.rand(in_caps, out_caps, out_dim, in_dim)
        out_shape = (batch_size, in_caps, out_caps, out_dim)

        with self.test_session():
            x_ph = tf.placeholder(tf.float32, x.shape)
            w_ph = tf.placeholder(tf.float32, weights.shape)
            fd = {x_ph: x, w_ph: weights}

            caps_out = capsule_prediction(x_ph, w_ph)
            grad_w = tf.test.compute_gradient(
                w_ph, weights.shape, caps_out, out_shape, extra_feed_dict=fd
            )

        self.assertAllClose(grad_w[0], grad_w[1], atol=1e-3, rtol=1e-3)

    @parameterized.expand([
        (batch_size, in_caps, out_caps, in_dim, out_dim) for
        batch_size, in_caps, out_caps, in_dim, out_dim in
        itertools.product([4, 8], [4, 8], [4, 8], [4, 8], [4, 8])
    ])
    def test_capsule_prediction_input_grad(self, batch_size, in_caps, out_caps, in_dim, out_dim):
        """ Tests gradient of output w.r.t. x """
        x = np.random.rand(batch_size, in_caps, in_dim)
        weights = np.random.rand(in_caps, out_caps, out_dim, in_dim)
        out_shape = (batch_size, in_caps, out_caps, out_dim)

        with self.test_session():
            x_ph = tf.placeholder(tf.float32, x.shape)
            w_ph = tf.placeholder(tf.float32, weights.shape)
            fd = {x_ph: x, w_ph: weights}
            caps_out = capsule_prediction(x_ph, w_ph)
            grad_x = tf.test.compute_gradient(
                x_ph, x.shape, caps_out, out_shape, extra_feed_dict=fd
            )

        self.assertAllClose(grad_x[0], grad_x[1], atol=1e-3, rtol=1e-3)
