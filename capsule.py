import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def network():
    x = tf.placeholder(tf.float32, [None, 28 * 28])
    labels = tf.placeholder(tf.int64, [None])
    images = tf.reshape(x, (-1, 28, 28, 1))
    conv1_out = tf.layers.conv2d(images, 256, 9, activation=tf.nn.relu)
    pc = primary_caps(conv1_out, kernel_size=9, strides=(2, 2), capsules=32, dim=8)
    v_j = digit_caps(pc, n_capsules=10, dim=16)
    digit_norms = tf.norm(v_j, axis=-1)
    reconstruction = decoder(v_j, labels)
    total_loss = caps_loss(digit_norms, labels) + 0.0005 * decoder_loss(x, reconstruction)
    train_op = tf.train.AdamOptimizer().minimize(total_loss)
    acc = accuracy(digit_norms, labels)
    return x, labels, train_op, acc


def caps_loss(digit_norms, labels, m_plus=0.9, m_minus=0.1, down_weighting=0.5):
    T_c = tf.one_hot(labels, depth=10)
    L_c = T_c * tf.square(tf.maximum(0.0, m_plus - digit_norms)) + \
        down_weighting * (1.0 - T_c) * tf.square(tf.maximum(0.0, m_minus - digit_norms))
    return tf.reduce_sum(L_c)


def accuracy(digit_norms, labels):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(digit_norms, axis=-1), labels), tf.float32))


def decoder_loss(x, reconstruction):
    return tf.nn.l2_loss(x - reconstruction)


def decoder(v_j, labels):
    mask = tf.expand_dims(tf.one_hot(labels, depth=10), 2)
    masked_input = tf.reshape(v_j * mask, (-1, 16 * 10))
    fc1 = tf.layers.dense(masked_input, 512, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu)
    fc3 = tf.layers.dense(fc2, 784, activation=tf.nn.sigmoid)
    return fc3


def primary_caps(x, kernel_size, strides, capsules, dim, name="PrimaryCaps"):
    with tf.variable_scope(name):
        n_in = x.shape.as_list()[-1]
        kernels = tf.get_variable(
            'kernels', shape=[kernel_size, kernel_size, n_in, capsules * dim],
            initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG", uniform=True)
        )
        bias = tf.get_variable('bias', shape=(capsules * dim,), initializer=tf.zeros_initializer())

        strides = (1,) + strides + (1,)
        preactivation = tf.nn.bias_add(tf.nn.conv2d(x, kernels, strides=strides, padding="VALID"), bias)
        _, w, h, _ = preactivation.shape.as_list()
        out = tf.reshape(preactivation, (-1, w * h * capsules, dim))
    return out


def digit_caps(incoming, n_capsules, dim, name="DigitCaps", neuron_axis=-1, capsule_axis=-2, routing_iters=3):
    with tf.variable_scope(name):
        in_shape = incoming.shape.as_list()
        n_in_capsules = in_shape[capsule_axis]
        dim_in_capsules = in_shape[neuron_axis]
        W_ij = tf.get_variable("weights", shape=[n_in_capsules, n_capsules, dim, dim_in_capsules],
                               initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG", uniform=True))
        u_i = tf.reshape(incoming, (-1, n_in_capsules, 1, 1, dim_in_capsules))
        u_hat = tf.reduce_sum(tf.expand_dims(W_ij, dim=0) * u_i, axis=-1)

        logits_shape = tf.stack([tf.shape(incoming)[0], n_in_capsules, n_capsules])

        def capsule_out(b_ij):
            c_ij = tf.nn.softmax(b_ij)
            s_j = tf.reduce_sum(tf.reshape(c_ij, (-1, n_in_capsules, n_capsules, 1)) * u_hat, axis=1)
            norms = tf.norm(s_j, keep_dims=True)
            norms2 = tf.square(norms)
            v_j = norms2 / (1 + norms2) * (s_j / norms)
            return v_j

        def body(iter, b_ij):
            v_j = capsule_out(b_ij)
            a_ij = tf.reduce_sum(tf.expand_dims(v_j, axis=1) * u_hat, axis=3)
            logits = tf.reshape(b_ij + a_ij, (-1, n_in_capsules, n_capsules))
            return [iter + 1, logits]

        i = tf.constant(0)
        routing_result = tf.while_loop(
            lambda i, b_ij: tf.less(i, routing_iters),
            body,
            [i, tf.zeros(logits_shape)]
        )
        v_j = capsule_out(routing_result[1])

    return v_j


def evaluate_on_test():
    test_scores = []
    test_epochs = mnist.test.epochs_completed
    while mnist.test.epochs_completed == test_epochs:
        images, labels = mnist.test.next_batch(batch_size=32)
        test_scores.append(sess.run(acc, feed_dict={x: images, y: labels}))
    print("Epoch {}, accuracy on test: {}".format(epochs, sum(test_scores) / len(test_scores)))


if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', reshape=True, one_hot=False)
    x, y, train_op, acc = network()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    epochs = mnist.train.epochs_completed
    step = 0
    while epochs != 10:
        images, labels = mnist.train.next_batch(batch_size=32)
        _, score = sess.run([train_op, acc], feed_dict={x: images, y: labels})
        if step % 10 == 0:
            print("Epoch {}, step {}, train accuracy: {}".format(epochs, step, score))

        if mnist.train.epochs_completed > epochs:
            evaluate_on_test()
        epochs = mnist.train.epochs_completed
        step += 1

