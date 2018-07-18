import math
import sys

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import Ellipse
from tensorflow.examples.tutorials.mnist import input_data

# parameters #
##############

beta = 1e-4     # Lagrange multiplier

vsz2 = 5        # visible square size
bsz2 = 7        # buffer square size
hsz = 10         # hiddens linear size

vsz = vsz2**2  # visible linear size
# esz = 784 - bsz2**2  # environment linear size
esz = 784
# models #
##########
mnist_data = input_data.read_data_sets("MNIST_data/")

vis = tf.placeholder(tf.float32, [None, vsz], 'visible')
env = tf.placeholder(tf.float32, [None, esz], 'environment')

ds = tf.contrib.distributions


def encoder(vis):
    net = tf.layers.dense(vis, units=1024, activation=tf.nn.relu)
    net = tf.layers.dense(net, units=1024, activation=tf.nn.relu)
    params = tf.layers.dense(net, units=hsz*2)
    mu, rho = params[:, :hsz], params[:, hsz:]
    encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)
    return encoding


def decoder(encoding_sample):
    net = tf.layers.dense(encoding_sample, units=1024)
    net = tf.layers.dense(net, units=1024, activation=tf.nn.relu)
    net = tf.layers.dense(net, units=esz)
    return net


prior = ds.Normal(0.0, 1.0)

# create graph #
################
with tf.variable_scope('encoder'):
    encoding = encoder(vis)

with tf.variable_scope('decoder'):
    lat = encoding.mean()
    penv = decoder(encoding.sample())   # predicted environment
    penv_sigmoid = tf.nn.sigmoid(penv)

# class_loss = tf.losses.softmax_cross_entropy(
#    logits=logits, onehot_labels=one_hot_labels) / math.log(2)

# pred_loss = tf.losses.mean_squared_error(
    # penv, env) / math.log(2)

pred_loss = tf.reduce_mean(tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=penv, labels=env), 0))

info_loss = tf.reduce_sum(tf.reduce_mean(
    ds.kl_divergence(encoding, prior), 0)) / math.log(2)

total_loss = pred_loss + beta * info_loss

# setup #
#########
batch_size = 100
steps_per_batch = int(mnist_data.train.num_examples / batch_size)

global_step = tf.train.get_or_create_global_step()
# learning_rate = tf.train.exponential_decay(1e-4, global_step,
# decay_steps=2*steps_per_batch,
# decay_rate=0.97, staircase=True)
learning_rate = 1e-4
opt = tf.train.AdamOptimizer(learning_rate, 0.51)

ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
ma_update = ma.apply(tf.model_variables())

train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                   global_step,
                                                   update_ops=[ma_update])

# train #
#########

rstart, cstart = (int(x) for x in sys.argv[1:])


def separate_image(imgs):
    """
    separates image into visible patch and environment
    """
    vis_data = np.zeros((imgs.shape[0], vsz))
    env_data = np.zeros((imgs.shape[0], esz))

    for idx, img in enumerate(imgs):
        vis_data[idx, :] = np.reshape(
            np.reshape(imgs[idx, :], (28, 28))[rstart:rstart+5, cstart:cstart+5], -1)

        # env = np.reshape(imgs[idx, :], (28, 28))
        # env[8:15, 8:15] = -1
        # env = np.reshape(env, -1)
        # env = env[env != -1]
        # env_data[idx, :] = env
        env_data[idx, :] = img

    return vis_data, env_data


def build_prior():
    """
    build a prior distribution for the things
    """
    imgs = mnist_data.test.images
    vis_data, _ = separate_image(imgs)


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    def evaluate(epoch, debug=False):
        """
        code to evaluate and visualize model
        """
        imgs = mnist_data.test.images
        labels = mnist_data.test.labels
        vis_data, env_data = separate_image(imgs)
        loss, prediction, latent = sess.run([pred_loss, penv_sigmoid, lat],
                                            feed_dict={vis: vis_data, env: env_data})

        if epoch % 10 == 0 and debug:
            idx = np.random.randint(imgs.shape[0], size=5)
            img = np.reshape(imgs[idx, :], (28*5, 28))
            pimg = np.reshape(prediction[idx, :], (28*5, 28))

            plt.figure()
            plt.matshow(np.concatenate((img, pimg), axis=1),
                        cmap=plt.cm.gray)
            plt.savefig('out/images_%02d_%02d_%02d.png' %
                        (rstart, cstart, epoch // 10), bbox_inches='tight')

        return loss

    for epoch in range(51):
        for step in range(steps_per_batch):
            imgs, _ = mnist_data.train.next_batch(batch_size)
            vis_data, env_data = separate_image(imgs)

            sess.run(train_tensor, feed_dict={vis: vis_data, env: env_data})
        print("Epoch: %02d\tLoss:%.3f" % (epoch, evaluate(epoch, debug=True)))
        sys.stdout.flush()

    saver.save(sess, "store/model_%02d_%02d.cpkt" % (rstart, cstart))
