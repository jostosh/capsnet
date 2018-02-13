import tensorflow as tf
import numpy as np


batch_size = 2
in_caps = 3
out_caps = 2
in_dim = 3
out_dim = 2

x = np.random.rand(batch_size, in_caps, in_dim)
weights = np.random.rand(in_caps, out_caps, out_dim, in_dim)

out_shape = (batch_size, in_caps, out_caps, out_dim)

op_module = tf.load_op_library('capsmatmul_op.so')

x_ph = tf.placeholder(tf.float32, x.shape)
w_ph = tf.placeholder(tf.float32, weights.shape)

ret = op_module.caps_mat_mul(x_ph, w_ph)

with tf.Session() as sess:
    out = sess.run(ret, feed_dict={x_ph: x, w_ph: weights})

out_truth = np.zeros(out_shape)

for b in range(batch_size):
    for i in range(in_caps):
        for j in range(out_caps):
            for c in range(out_dim):
                out_truth[b, i, j, c] = np.dot(x[b, i], weights[i, j, c])

print(sorted(out_truth.reshape((out_truth.size,))))
print(sorted(out.reshape((out.size,))))

print(np.allclose(out_truth, out))
