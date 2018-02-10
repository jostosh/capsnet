import tensorflow as tf
from ops.capsmatmul import capsmatmul
import numpy as np
from parameterized import parameterized
import itertools


class CapsMatMulOpTest(tf.test.TestCase):

    def _numpy_capsmatmul(self, x, weights):
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
        itertools.product(
            range(2, 32, 16), range(32, 256, 128), range(32, 256, 128), range(8, 32, 16),
            range(4, 32, 16))
    ])
    def test_capsmatmul_op(self, batch_size, in_caps, out_caps, in_dim, out_dim):
        """ Tests the forward capsmatmul op """
        x = np.random.rand(batch_size, in_caps, in_dim)
        weights = np.random.rand(in_caps, out_caps, out_dim, in_dim)

        truth = self._numpy_capsmatmul(x, weights)
        with self.test_session() as sess:
            x_ph = tf.placeholder(tf.float32, x.shape)
            w_ph = tf.placeholder(tf.float32, weights.shape)

            ret = capsmatmul(x_ph, w_ph)
            out = sess.run(ret, {x_ph: x, w_ph: weights})
        self.assertAllClose(truth, out)


    @parameterized.expand([
        (batch_size, in_caps, out_caps, in_dim, out_dim) for
        batch_size, in_caps, out_caps, in_dim, out_dim in
        itertools.product(
            range(2, 32, 16), range(32, 256, 128), range(32, 256, 128), range(8, 32, 16),
            range(4, 32, 16))
    ])
    def test_capsmatmul_grad(self, batch_size, in_caps, out_caps, in_dim, out_dim):
        """ Tests the forward capsmatmul op """
        x = np.random.rand(batch_size, in_caps, in_dim)
        weights = np.random.rand(in_caps, out_caps, out_dim, in_dim)
        out_shape = (batch_size, in_caps, out_caps, out_dim)

        # truth = self._numpy_capsmatmul(x, weights)
        with self.test_session():
            x_ph = tf.placeholder(tf.float32, x.shape)
            w_ph = tf.placeholder(tf.float32, weights.shape)
            fd = {x_ph: x, w_ph: weights}

            caps_out = capsmatmul(x_ph, w_ph)
            # grad_x = tf.test.compute_gradient(x_ph, x.shape, caps_out, out_shape, extra_feed_dict=fd)
            grad_w = tf.test.compute_gradient(w_ph, weights.shape, caps_out, out_shape,
                                              extra_feed_dict=fd)
            # grad_error = tf.test.compute_gradient_error(
            #     w_ph, weights.shape, caps_out, out_shape, extra_feed_dict=fd, delta=1e-5)

        # self.assertAllClose(grad_x[0], grad_x[1])
        self.assertAllClose(grad_w[0], grad_w[1], atol=5e-4, rtol=5e-4)
