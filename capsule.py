import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from ops.capsuleprediction import capsule_prediction
import tqdm
import argparse


def build_caps_net(steps_per_epoch):
    """
    Builds the network, returns input placeholder, label placeholder, training Op, accuracy Op
    """
    # Define placeholders
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    labels = tf.placeholder(tf.int64, [None])

    # Define capsule network
    conv1_out = tf.layers.conv2d(x, 256, 9, activation=tf.nn.relu)
    pc = primary_caps(conv1_out, kernel_size=9, strides=(2, 2), capsules=32, dim=8)
    v_j = digit_caps(pc, n_digit_caps=10, dim_digit_caps=16)
    digit_norms = tf.norm(v_j, axis=-1)

    # Reconstruction from decoder
    reconstruction = decoder(v_j, labels)

    # Define loss
    total_loss = caps_loss(digit_norms, labels) + \
                 args.decoder_lambda * reconstruction_loss(x, reconstruction)

    # Define all Ops needed for training training and evaluation
    global_step = tf.Variable(0, trainable=False)
    initial_learning_rate = 1e-3
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate, global_step=global_step, decay_steps=steps_per_epoch, decay_rate=0.9
    )
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)
    acc = accuracy(digit_norms, labels)
    return x, labels, train_op, acc


def caps_loss(digit_norms, labels, m_plus=0.9, m_minus=0.1, down_weighting=0.5):
    """ Defines the loss of the network as given in the paper """
    T_c = tf.one_hot(labels, depth=10)
    L_c = T_c * tf.square(tf.maximum(0.0, m_plus - digit_norms)) + \
        down_weighting * (1.0 - T_c) * tf.square(tf.maximum(0.0, digit_norms - m_minus))
    return tf.reduce_sum(L_c)


def accuracy(digit_norms, labels):
    """ Defines Op to determine prediction accuracy """
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(digit_norms, axis=-1), labels), tf.float32))


def reconstruction_loss(x, reconstruction):
    """ Squared loss for reconstruction. Note that tf.nn.l2_loss(x) computes 1/2 * (x ** 2) """
    return tf.nn.l2_loss(x - reconstruction)


def decoder(v_j, labels):
    """ Define decoder """
    # Mask for selecting target capsule
    mask = tf.expand_dims(tf.one_hot(labels, depth=10), 2)
    masked_digit_caps = tf.reshape(v_j * mask, (-1, 16 * 10))
    decoder_out = decoder_network(masked_digit_caps)
    return tf.reshape(decoder_out, (-1, 28, 28, 1))


def decoder_network(x, reuse=None):
    """ Defines decoder as given in paper """
    fc1 = tf.layers.dense(x, 512, activation=tf.nn.relu, name="DecoderFC1", reuse=reuse)
    fc2 = tf.layers.dense(fc1, 1024, activation=tf.nn.relu, name="DecoderFC2", reuse=reuse)
    fc3 = tf.layers.dense(fc2, 784, activation=tf.nn.sigmoid, name="DecoderOut", reuse=reuse)
    return fc3


def primary_caps(x, kernel_size, strides, capsules, dim, name="PrimaryCaps"):
    """ Primary capsule layer. Linear convolution with reshaping to account for capsules. """
    preactivation = tf.layers.conv2d(
        x, capsules * dim, kernel_size, strides=strides, activation=tf.identity, name=name,
        kernel_initializer=tf.keras.initializers.glorot_uniform()
    )
    _, w, h, _ = preactivation.shape.as_list()
    out = tf.reshape(preactivation, (-1, w * h * capsules, dim))
    return squash(out)


def squash(s_j):
    """ Squashing function as given in the paper """
    squared_norms = tf.reduce_sum(tf.square(s_j), axis=-1, keepdims=True)
    return s_j * squared_norms / (1 + squared_norms) / tf.sqrt(squared_norms + 1e-8)


def digit_caps(incoming, n_digit_caps, dim_digit_caps, name="DigitCaps", neuron_axis=-1,
               capsule_axis=-2, routing_iters=3):
    """ Digit capsule layer """
    with tf.variable_scope(name):
        # Get number of capsules and dimensionality of previous layer
        in_shape = incoming.shape.as_list()
        n_primary_caps = in_shape[capsule_axis]
        dim_primary_caps = in_shape[neuron_axis]
        # Initialize all weight matrices
        w_shape = [n_primary_caps, n_digit_caps, dim_digit_caps, dim_primary_caps] \
            if args.custom_op else [n_primary_caps, n_digit_caps * dim_digit_caps, dim_primary_caps]

        W_ij = tf.get_variable(
            "weights", shape=w_shape,
            initializer=tf.keras.initializers.glorot_uniform()
        )
        # Initialize routing logits, the leading axis with size 1 is added for convenience.
        b_ij = tf.get_variable(
            "logits", shape=[1, n_primary_caps, n_digit_caps], initializer=tf.zeros_initializer(),
            trainable=args.logits_trainable
        )
        if args.custom_op:
            # Custom op
            u_hat = capsule_prediction(incoming, W_ij)
        else:
            # Reshape and transpose hacking
            u_i = tf.transpose(incoming, (1, 2, 0))
            u_hat = tf.matmul(W_ij, u_i)
            u_hat = tf.reshape(
                tf.transpose(u_hat, (2, 0, 1)), (-1, n_primary_caps, n_digit_caps, dim_digit_caps)
            )

        def capsule_out(b_ij):
            """ Given the logits b_ij, computes the output of this layer. """
            c_ij = tf.nn.softmax(b_ij, axis=2)
            s_j = tf.reduce_sum(
                tf.reshape(c_ij, (-1, n_primary_caps, n_digit_caps, 1)) * u_hat, axis=1
            )
            v_j = squash(s_j)
            return v_j

        def routing_iteration(iter, logits):
            """
            Given a set of logits, computes the new logits using the routing definition from the
            paper.
            """
            v_j = capsule_out(logits)
            a_ij = tf.reduce_sum(tf.expand_dims(v_j, axis=1) * u_hat, axis=3)
            logits = tf.reshape(logits + a_ij, (-1, n_primary_caps, n_digit_caps))
            return [iter + 1, logits]

        # Compute routing
        i = tf.constant(0)
        routing_result = tf.while_loop(
            lambda i, logits: tf.less(i, routing_iters),
            routing_iteration,
            [i, tf.tile(b_ij, tf.stack([tf.shape(incoming)[0], 1, 1]))]
        )
        # Second element of the result contains our final logits
        v_j = capsule_out(routing_result[1])

    return v_j


def evaluate_on_test(epoch):
    test_scores = []
    test_epochs = mnist.test.epochs_completed
    while mnist.test.epochs_completed == test_epochs:
        images, labels = mnist.test.next_batch(batch_size=args.batch_size)
        test_scores.append(sess.run(acc, feed_dict={x: images, y: labels}))
    mean_accuracy = np.mean(test_scores)
    with open(args.logs, 'a') as f:
        f.write("{},{}\n".format(epoch, mean_accuracy))
    print("Epoch {}, accuracy on test: {:.3f}".format(epoch, mean_accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", dest="single")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decoder_lambda", type=float, default=0.001)
    parser.add_argument("--shift", type=float, default=2/28)
    parser.add_argument("--logs", default="logs.csv")
    parser.add_argument("--epochs", default=50)
    parser.add_argument(
        "--logits_trainable", default=False, action="store_true", dest="logits_trainable"
    )
    parser.add_argument("--custom_op", action="store_true", dest="custom_op")
    parser.add_argument("--no_pbar", action="store_false", dest="pbar")
    parser.add_argument("--datadir", default="MNIST_data")
    parser.set_defaults(pbar=True)
    args = parser.parse_args()

    with open(args.logs, 'w') as f:
        f.write("epoch,accuracy\n")

    mnist = input_data.read_data_sets(args.datadir, validation_size=0, reshape=False, one_hot=False)
    steps_per_epoch = np.ceil(mnist.train.num_examples / args.batch_size)
    x, y, train_op, acc = build_caps_net(steps_per_epoch)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=args.shift,
        height_shift_range=args.shift,
        fill_mode='constant',
        cval=0.
    ).flow(mnist.train.images, mnist.train.labels, batch_size=args.batch_size)

    print("Beginning training")
    for epoch in range(args.epochs):
        pbar = tqdm.tqdm(range(int(steps_per_epoch))) if args.pbar else range(int(steps_per_epoch))
        total_score = 0
        for s in pbar:
            images, labels = train_generator.next()
            _, score = sess.run([train_op, acc], feed_dict={x: images, y: labels})
            total_score += score
            if args.pbar:
                pbar.set_description(
                    "Epoch ({:02d}/10) | Train accuracy: {:.3f}".format(epoch, total_score / (s+1)))
        evaluate_on_test(epoch)

