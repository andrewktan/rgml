import numpy as np
import tensorflow as tf

from cifar_dataset import *

# hyperparameters
NUMEL = 1024
NUMCLUSTER = 2

NUMCLASS = 10


class LinearClassifier():

    def __init__(self, dstrain, dstest, sz=32, num_clusters=2):
        # hyperparameters
        self.sz = sz
        self.NUMEL = self.sz**2
        self.NUMCLUSTER = num_clusters
        self.NUMCLASS = 10
        self.NUMEPOCHS = 100
        self.MBSIZE = 32
        self.LR = 2e-4

        self.parameters = []

        with tf.variable_scope('image_input'):
            self.image_placeholder = tf.placeholder(tf.int64,
                                                    shape=[None, self.NUMEL],
                                                    name='image-input')

            self.image_onehot = tf.reshape(
                tf.one_hot(self.image_placeholder, self.NUMCLUSTER),
                [-1, self.NUMEL * self.NUMCLUSTER],
                name='image-input-onehot')

            self.label_placeholder = tf.placeholder(tf.int64,
                                                    shape=[None, 1],
                                                    name='image-label')

            self.label_onehot = tf.one_hot(self.label_placeholder,
                                           self.NUMCLASS,
                                           name='image-label-onehot')

        with tf.variable_scope('learning_parameters'):
            self.lr_placeholder = tf.placeholder('float',
                                                 None,
                                                 name='learning_rate')

        self.dstrain = dstrain
        self.dstest = dstest

        self._model()
        self._create_loss()
        self._create_optimizer(self.parameters)
        self._evaluate()

    def _model(self):
        with tf.variable_scope('linear_layer'):
            W = tf.Variable(
                tf.truncated_normal(shape=[self.NUMEL*self.NUMCLUSTER,
                                           self.NUMCLASS],
                                    mean=0.0,
                                    stddev=1.0 /
                                    np.sqrt(self.NUMEL*self.NUMCLUSTER),
                                    dtype=tf.float32),
                trainable=True,
                name='weights')

            b = tf.Variable(
                tf.constant(0.1, shape=[self.NUMCLASS], dtype=tf.float32),
                trainable=True,
                name='biases')

            self.parameters += [W, b]

            self.logits = tf.nn.bias_add(
                tf.matmul(self.image_onehot, W),
                b)

            self.prediction = tf.nn.softmax(self.logits)

    def _create_loss(self):
        with tf.variable_scope('loss_layer'):
            cost = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.label_onehot)

            self.cost = tf.reduce_mean(cost)

    def _create_optimizer(self, variables):
        with tf.variable_scope('train'):
            self.train_step = tf.train.AdamOptimizer(
                learning_rate=self.lr_placeholder, beta1=0.5).minimize(self.cost, var_list=variables)

    def _evaluate(self):
        with tf.variable_scope('test'):
            correct_prediction = tf.equal(
                tf.argmax(self.prediction, axis=1), self.label_placeholder[:, 0])
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

    def train(self):
        init = tf.initialize_all_variables()

        learning_rate = self.LR

        ITERATIONS_PER_BATCH = 50000//32

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self.NUMEPOCHS):
                tc = 0
                for i in range(ITERATIONS_PER_BATCH):
                    batch_images, batch_labels = self.dstrain.next_batch()
                    _, cost, prediction, lonehot = sess.run([self.train_step, self.cost, self.prediction, self.label_onehot],
                                                            feed_dict={self.image_placeholder: batch_images,
                                                                       self.label_placeholder: batch_labels,
                                                                       self.lr_placeholder: learning_rate})

                    tc += cost / ITERATIONS_PER_BATCH

                test_images, test_labels = self.dstest.next_batch()

                _, accuracy = sess.run([self.cost, self.accuracy],
                                       feed_dict={self.image_placeholder: test_images,
                                                  self.label_placeholder: test_labels})

                print("Epoch: %03d\tCost: %.3f\tAccuracy: %.3f" %
                      (epoch, tc, accuracy))


if __name__ == '__main__':
    sz = 10

    dspath_train = '/Users/andrew/Documents/rgml/cifar-10_data/data_all_cg_03_15'
    dspath_test = '/Users/andrew/Documents/rgml/cifar-10_data/test_batch_cg_03_15'

    dstrain = CIFARDataset(dspath_train, sz=10)
    dstest = CIFARDataset(dspath_test, sz=10, mb_size=-1)

    LC = LinearClassifier(dstrain, dstest, sz=sz, num_clusters=15)
    LC.train()
