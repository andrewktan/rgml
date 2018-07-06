import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# parameters #
##############

beta = 1e-4     # Lagrange multiplier

vsz2 = 5        # visible square size
bsz2 = 7        # buffer square size
hsz = 1         # hiddens linear size

vsz = vsz2**2  # visible linear size
# esz = 784 - bsz2**2  # environment linear size
esz = 784

# models #
##########
sess = tf.InteractiveSession()

mnist_data = input_data.read_data_sets("MNIST_data/")

vis = tf.placeholder(tf.float32, [None, vsz], 'visible')
env = tf.placeholder(tf.float32, [None, esz], 'environment')

layers = tf.contrib.layers
ds = tf.contrib.distributions


def encoder(vis):
    net = layers.relu(2*vis-1, 1024)
    net = layers.relu(net, 1024)
    params = layers.linear(net, hsz*2)
    mu, rho = params[:, :hsz], params[:, hsz:]
    encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)
    return encoding


def decoder(encoding_sample):
    net = layers.linear(encoding_sample, 1024)
    net = layers.relu(net, 1024)
    net = layers.relu(net, esz)
    return net


prior = ds.Normal(0.0, 1.0)

# create graph #
################
with tf.variable_scope('encoder'):
    encoding = encoder(vis)

with tf.variable_scope('decoder'):
    penv = decoder(encoding.sample())   # predicted environment

# class_loss = tf.losses.softmax_cross_entropy(
#    logits=logits, onehot_labels=one_hot_labels) / math.log(2)

pred_loss = tf.losses.mean_squared_error(
    penv, env) / math.log(2)

info_loss = tf.reduce_sum(tf.reduce_mean(
    ds.kl_divergence(encoding, prior), 0)) / math.log(2)

total_loss = pred_loss + beta * info_loss

# setup #
#########
batch_size = 100
steps_per_batch = int(mnist_data.train.num_examples / batch_size)

global_step = tf.train.get_or_create_global_step()
learning_rate = tf.train.exponential_decay(1e-4, global_step,
                                           decay_steps=2*steps_per_batch,
                                           decay_rate=0.97, staircase=True)
opt = tf.train.AdamOptimizer(learning_rate, 0.5)

ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
ma_update = ma.apply(tf.model_variables())

train_tensor = tf.contrib.training.create_train_op(total_loss, opt,
                                                   global_step,
                                                   update_ops=[ma_update])

tf.global_variables_initializer().run()

# train #
#########


def separate_image(imgs):
    """
    separates image into visible patch and environment
    """
    vis_data = np.zeros((imgs.shape[0], vsz))
    env_data = np.zeros((imgs.shape[0], esz))

    for idx, img in enumerate(imgs):
        vis_data[idx, :] = np.reshape(
            np.reshape(imgs[idx, :], (28, 28))[15:20, 15:20], -1)

        # env = np.reshape(imgs[idx, :], (28, 28))
        # env[8:15, 8:15] = -1
        # env = np.reshape(env, -1)
        # env = env[env != -1]
        # env_data[idx, :] = env
        env_data[idx, :] = img

    return vis_data, env_data


def evaluate(epoch):
    imgs = mnist_data.test.images
    vis_data, env_data = separate_image(imgs)
    loss, prediction = sess.run([pred_loss, penv],
                                feed_dict={vis: vis_data, env: env_data})

    if epoch % 10 == 0:
        img = np.reshape(imgs[0:5, :], (28*5, 28))
        pimg = np.reshape(prediction[0:5, :], (28*5, 28))
        plt.matshow(np.concatenate((img, pimg), axis=1), cmap=plt.cm.gray)
        plt.show()

    return loss


for epoch in range(200):
    for step in range(steps_per_batch):
        imgs, _ = mnist_data.train.next_batch(batch_size)
        vis_data, env_data = separate_image(imgs)

        sess.run(train_tensor, feed_dict={vis: vis_data, env: env_data})
    print("Epoch: %02d\tLoss:%.3f" % (epoch, evaluate(epoch)))
    sys.stdout.flush()
