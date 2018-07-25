
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from cifar_iterator import *


class VariationalAutoencoder():
    def __init__(self,
                 latent_dimensions=10,
                 num_epochs=10,
                 learning_rate=1e-3,
                 num_train=50000,
                 batch_size=32):

        # hyperparameters
        self.LATENT_DIM = latent_dimensions
        self.NUM_EPOCHS = num_epochs
        self.LEARNING_RATE = learning_rate
        self.NUM_TRAIN = num_train
        self.BATCH_SIZE = batch_size

        # initialize
        self.parameters = []

        with tf.variable_scope('image_input'):
            self.x_placeholder = tf.placeholder('float',
                                                shape=[None, 32, 32, 1],
                                                name='x-input')

        with tf.variable_scope('learning_parameters'):
            self.lr_placeholder = tf.placeholder('float',
                                                 None,
                                                 name='learning_rate')

        dsfile = '/Users/andrew/Documents/rgml/cifar-10_data/data_all'

        self.trainset = CIFARIterator(dsfile,
                                      grayscale=True,
                                      mb_size=self.BATCH_SIZE)

        self._create_network()
        self._create_loss()
        self._create_optimizer(self.parameters)

    def _create_network(self):
        self._encoder_network()
        self.eps_placeholder = tf.placeholder(
            'float', shape=[None, self.LATENT_DIM])

        with tf.variable_scope('sample_latent'):
            self.z = tf.add(self.z_mean,
                            tf.multiply(
                                tf.sqrt(tf.exp(self.z_log_sigma_sq)),
                                self.eps_placeholder
                            ))

        self._decoder_network()

    def _create_loss(self):
        with tf.variable_scope('loss_layer'):

            x_vectorized = tf.reshape(
                self.x_placeholder, [-1, 1024], name='x-vectorized')
            x_reconstr_mean_logits_vectorized = tf.reshape(
                self.x_reconstr_mean_logits, [-1, 1024], name='x_reconstr_mean_vectorized')

            # pixel_loss = tf.reduce_sum(
            # tf.square(x_reconstr_mean_vectorized - x_vectorized), 1)
            pixel_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=x_reconstr_mean_logits_vectorized, labels=x_vectorized))

            self.pixel_loss = pixel_loss

            self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq -
                                                    tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

            self.latent_loss_mean = tf.reduce_mean(self.latent_loss)
            self.pixel_loss_mean = tf.reduce_mean(self.pixel_loss)

            self.cost = tf.reduce_mean(
                self.latent_loss + self.pixel_loss, name='cost_function')   # average over batch

    def _create_optimizer(self, variables):
        with tf.variable_scope('train'):
            self.train_Step = tf.train.AdamOptimizer(
                learning_rate=self.lr_placeholder).minimize(self.cost, var_list=variables)

    def _encoder_network(self):
        images = self.x_placeholder

        # conv 1.1
        with tf.variable_scope('enc_conv1_1'):
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 1, 32],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)

            conv = tf.nn.conv2d(images,
                                filter=kernel,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            biases = tf.Variable(
                tf.constant(0.1, shape=[32], dtype=tf.float32),
                trainable=True,
                name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.enc_conv1_1 = tf.nn.relu(out, name='enc_conv1_1')
            self.parameters += [kernel, biases]

        # conv 1.2
        with tf.variable_scope('enc_conv1_2'):
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 32, 32],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)

            conv = tf.nn.conv2d(self.enc_conv1_1,
                                filter=kernel,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            biases = tf.Variable(
                tf.constant(0.1, shape=[32], dtype=tf.float32),
                trainable=True,
                name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.enc_conv1_2 = tf.nn.relu(out, name='enc_conv1_2')
            self.parameters += [kernel, biases]

        # maxpool 1
        with tf.variable_scope('enc_maxpool1'):
            self.enc_pool1 = tf.nn.max_pool(self.enc_conv1_2,
                                            ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1],
                                            padding='SAME',
                                            name='enc_pool1')
        # conv 1.2
        with tf.variable_scope('enc_conv2_1'):
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 32, 64],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)

            conv = tf.nn.conv2d(self.enc_pool1,
                                filter=kernel,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            biases = tf.Variable(
                tf.constant(0.1, shape=[64], dtype=tf.float32),
                trainable=True,
                name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.enc_conv2_1 = tf.nn.relu(out, name='enc_conv2_1')
            self.parameters += [kernel, biases]

        # conv 2.2
        with tf.variable_scope('enc_conv2_2'):
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 64, 64],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)

            conv = tf.nn.conv2d(self.enc_conv2_1,
                                filter=kernel, strides=[1, 1, 1, 1],
                                padding='SAME')

            biases = tf.Variable(
                tf.constant(0.1, shape=[64], dtype=tf.float32),
                trainable=True,
                name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.enc_conv2_2 = tf.nn.relu(out, name='enc_conv2_2')
            self.parameters += [kernel, biases]

        # maxpool 2
        with tf.variable_scope('enc_maxpool2'):
            self.enc_pool2 = tf.nn.max_pool(self.enc_conv2_2,
                                            ksize=[1, 2, 2, 1],
                                            strides=[1, 2, 2, 1],
                                            padding='SAME',
                                            name='enc_pool2')

        prev_dim = int(np.prod(self.enc_pool2.get_shape()[1:]))

        pool2_flat = tf.reshape(self.enc_pool2, [-1, prev_dim])

        # fc 1
        with tf.variable_scope('enc_z_mean'):
            fc_w = tf.get_variable(name='weights',
                                   shape=[prev_dim, self.LATENT_DIM],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=True)

            fc_b = tf.get_variable(name='biases',
                                   shape=[self.LATENT_DIM],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=True)

            self.z_mean = tf.nn.bias_add(
                tf.matmul(pool2_flat, fc_w), fc_b)

            self.parameters += [fc_w, fc_b]

        # fc 2
        with tf.variable_scope('enc_z_variance'):
            fc_w = tf.get_variable(name='weights',
                                   shape=[prev_dim, self.LATENT_DIM],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=True)

            fc_b = tf.get_variable(name='biases',
                                   shape=[self.LATENT_DIM],
                                   dtype=tf.float32,
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=True)

            self.z_log_sigma_sq = tf.nn.bias_add(
                tf.matmul(pool2_flat, fc_w), fc_b)

            self.parameters += [fc_w, fc_b]

    def _decoder_network(self):
        with tf.variable_scope('dec_fc'):
            fc_g_w = tf.get_variable(name='weights',
                                     shape=[self.LATENT_DIM, 4096],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)

            fc_g_b = tf.get_variable(name='biases',
                                     shape=[4096],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)

            a_1 = tf.nn.bias_add(tf.matmul(self.z, fc_g_w), fc_g_b)
            self.g_1 = tf.nn.relu(a_1)
            self.parameters += [fc_g_w, fc_g_b]

        with tf.variable_scope('dec_reshape'):
            g_1_images = tf.reshape(self.g_1, [-1, 8, 8, 64])

        # scale up to size 16 x 16 x 64
        resized_1 = None
        with tf.variable_scope('dec_resize1'):
            resized_1 = tf.image.resize_images(g_1_images,
                                               [16, 16],
                                               method=tf.image.ResizeMethod.BILINEAR)

        # deconv 1.1
        with tf.variable_scope('dec_conv1_1'):
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 64, 32],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)

            conv = tf.nn.conv2d(resized_1,
                                filter=kernel,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            biases = tf.Variable(
                tf.constant(0.1, shape=[32], dtype=tf.float32),
                trainable=True,
                name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.dec_conv1_1 = tf.nn.relu(out, name='dec_conv1_1')
            self.parameters += [kernel, biases]

        # deconv 1.2
        with tf.variable_scope('dec_conv1_2'):
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 32, 32],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)

            conv = tf.nn.conv2d(self.dec_conv1_1,
                                filter=kernel,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            biases = tf.Variable(
                tf.constant(0.1, shape=[32], dtype=tf.float32),
                trainable=True,
                name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.dec_conv1_2 = tf.nn.relu(out, name='dec_conv1_2')
            self.parameters += [kernel, biases]

        # scale up to size 32 x 32 x 32
        resized_2 = None
        with tf.variable_scope('dec_resize2'):
            resized_2 = tf.image.resize_images(self.dec_conv1_2,
                                               [32, 32],
                                               method=tf.image.ResizeMethod.BILINEAR)

        # deconv 2.1
        with tf.variable_scope('dec_conv2_1'):
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 32, 1],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)

            conv = tf.nn.conv2d(resized_2,
                                filter=kernel,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            biases = tf.Variable(
                tf.constant(0.1, shape=[1], dtype=tf.float32),
                trainable=True,
                name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.dec_conv2_1 = tf.nn.relu(out, name='dec_conv2_1')
            self.parameters += [kernel, biases]

        # deconv 2.2
        with tf.variable_scope('dec_conv2_2'):
            kernel = tf.get_variable(name='weights',
                                     shape=[3, 3, 1, 1],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer(),
                                     trainable=True)

            conv = tf.nn.conv2d(self.dec_conv2_1,
                                filter=kernel,
                                strides=[1, 1, 1, 1],
                                padding='SAME')

            biases = tf.Variable(
                tf.constant(0.1, shape=[1], dtype=tf.float32),
                trainable=True,
                name='biases')

            out = tf.nn.bias_add(conv, biases)

            self.dec_conv2_2 = out
            self.parameters += [kernel, biases]

        with tf.variable_scope('dec_output'):
            # self.x_reconstr_mean = tf.sigmoid(self.dec_conv2_2)
            self.x_reconstr_mean_logits = self.dec_conv2_2

    def train(self, num_epochs_to_display=1):
        tc = 0
        fc = 0
        lc = 0
        pc = 0

        costs = {}
        costs['latent'] = []
        costs['pixel'] = []
        costs['total'] = []

        current_lr = 1e-4

        init = tf.initialize_all_variables()
        saver = tf.train.Saver(self.parameters)

        current_epoch_cost = 0
        current_rec_cost = 0
        current_lat_cost = 0
        current_pix_cost = 0

        ITERATIONS_PER_EPOCH = int(self.NUM_TRAIN/self.BATCH_SIZE)

        with tf.Session() as sess:
            sess.run(init)

            batch_images, _ = self.trainset.next_batch()
            batch_images = np.reshape(batch_images, [-1, 32, 32, 1])

            t0 = time.time()

            eps = np.random.normal(loc=0.0,
                                   scale=1.0,
                                   size=(self.BATCH_SIZE, self.LATENT_DIM))

            tc, lc, pc = sess.run([self.cost,
                                   self.latent_loss_mean,
                                   self.pixel_loss_mean],
                                  feed_dict={self.x_placeholder: batch_images,
                                             self.eps_placeholder: eps}
                                  )

            # Append them to lists
            costs['total'].append(fc)
            costs['pixel'].append(pc)
            costs['latent'].append(lc)

            t1 = time.time()
            print(
                'Initial Cost: {:2f}  = {:.2f} L + {:.2f} P -- time taken {:.2f}'.format(tc, lc, pc, t1-t0))

            t0 = t1

            # Train for several epochs
            for epoch in range(self.NUM_EPOCHS):

                print("learning rate : {}".format(current_lr))

                for i in range(ITERATIONS_PER_EPOCH):
                    # pick a mini batch
                    batch_images, _ = self.trainset.next_batch()
                    batch_images = np.reshape(batch_images, [-1, 32, 32, 1])

                    eps = np.random.normal(loc=0.0,
                                           scale=1.0,
                                           size=(self.BATCH_SIZE, self.LATENT_DIM))

                    tc, lc, pc = sess.run([self.cost, self.latent_loss_mean, self.pixel_loss_mean],
                                          feed_dict={self.x_placeholder: batch_images, self.eps_placeholder: eps, self.lr_placeholder: current_lr})

                    current_epoch_cost += tc
                    current_lat_cost += lc
                    current_pix_cost += pc

                # print stats
                if epoch % num_epochs_to_display == 0:
                    t1 = time.time()
                    print(' epoch: {}/{} -- cost {:.2f} = {:.2f} L + {:.2f} P -- time taken {:.2f}'.format(
                        epoch+1, self.NUM_EPOCHS, current_epoch_cost, current_lat_cost, current_pix_cost, t1-t0))
                    # Reset the timer
                    t0 = t1

                # reset costs
                current_epoch_cost = 0
                current_rec_cost = 0
                current_lat_cost = 0
                current_pix_cost = 0


if __name__ == '__main__':
    VAE = VariationalAutoencoder()
    VAE.train()
