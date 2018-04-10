from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pickle as pkl
from sklearn.manifold import TSNE

from flip_gradient import flip_gradient
from utils import *
import time

from tensorflow.examples.tutorials.mnist import input_data

class DANN(object):
    #初始化各类定义
    def __init__(self, sess):
        self.X = tf.placeholder(tf.uint8, [None, 28, 28, 3])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])
        self.batch_size = 64
        self.sess = sess
        self.model_name = "DANN"

        self.mnist = input_data.read_data_sets('./data/mnist/', one_hot=True)

        # Process MNIST
        self.mnist_train = (self.mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
        self.mnist_train = np.concatenate([self.mnist_train, self.mnist_train, self.mnist_train], 3)
        self.mnist_test = (self.mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
        self.mnist_test = np.concatenate([self.mnist_test, self.mnist_test, self.mnist_test], 3)

        # Load MNIST-M
        self.mnistm = pkl.load(open('./data/mnistm/mnistm.pkl', 'rb'))
        self.mnistm_train = self.mnistm['train']
        self.mnistm_test = self.mnistm['test']
        self.mnistm_valid = self.mnistm['valid']
        # mnistm_train, mnistm_test, mnistm_valid, _ = load_mnist_M("mnistm")

        # Compute pixel mean for normalizing data
        self.pixel_mean = np.vstack([self.mnist_train, self.mnistm_train]).mean((0, 1, 2))

        # Create a mixed dataset for TSNE visualization
        num_test = 500
        self.combined_test_imgs = np.vstack([self.mnist_test[:num_test], self.mnistm_test[:num_test]])
        self.combined_test_labels = np.vstack([self.mnist.test.labels[:num_test], self.mnist.test.labels[:num_test]])
        self.combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
                                          np.tile([0., 1.], [num_test, 1])])

    def build_model(self):

        X_input = (tf.cast(self.X, tf.float32) - self.pixel_mean) / 255.

        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([5, 5, 3, 32])
            b_conv0 = bias_variable([32])
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([5, 5, 32, 48])
            b_conv1 = bias_variable([48])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            # The domain-invariant feature
            self.feature = tf.reshape(h_pool1, [-1, 7 * 7 * 48])

        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [self.batch_size // 2, -1])
            classify_feats = tf.cond(self.train, source_features, all_features)

            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [self.batch_size // 2, -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)

            W_fc0 = weight_variable([7 * 7 * 48, 100])
            b_fc0 = bias_variable([100])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            W_fc1 = weight_variable([100, 100])
            b_fc1 = bias_variable([100])
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = weight_variable([100, 10])
            b_fc2 = bias_variable([10])
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)

            d_W_fc0 = weight_variable([7 * 7 * 48, 100])
            d_b_fc0 = bias_variable([100])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([100, 2])
            d_b_fc1 = bias_variable([2])
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1

            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)

    def train_and_evaluate(self, training_mode, num_steps=8600):
        """Helper to run the model with different training modes."""
        graph = tf.get_default_graph()
        with graph.as_default():
            learning_rate = tf.placeholder(tf.float32, [])

            pred_loss = tf.reduce_mean(self.pred_loss)
            domain_loss = tf.reduce_mean(self.domain_loss)
            total_loss = pred_loss + domain_loss

            regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
            dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

            # Evaluation
            correct_label_pred = tf.equal(tf.argmax(self.classify_labels, 1), tf.argmax(self.pred, 1))
            label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
            correct_domain_pred = tf.equal(tf.argmax(self.domain, 1), tf.argmax(self.domain_pred, 1))
            domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))
        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run()

            # Batch generators
            gen_source_batch = batch_generator(
                [self.mnist_train, self.mnist.train.labels], self.batch_size // 2)
            gen_target_batch = batch_generator(
                [self.mnistm_train, self.mnist.train.labels], self.batch_size // 2)
            gen_source_only_batch = batch_generator(
                [self.mnist_train, self.mnist.train.labels], self.batch_size)
            gen_target_only_batch = batch_generator(
                [self.mnistm_train, self.mnist.train.labels], self.batch_size)

            domain_labels = np.vstack([np.tile([1., 0.], [self.batch_size // 2, 1]),
                                       np.tile([0., 1.], [self.batch_size // 2, 1])])

            # Training loop
            start_time = time.time()
            for i in range(num_steps):

                # Adaptation param and learning rate schedule as described in the paper
                p = float(i) / num_steps
                l = 2. / (1. + np.exp(-10. * p)) - 1
                lr = 0.01 / (1. + 10 * p) ** 0.75

                # Training step
                if training_mode == 'dann':

                    X0, y0 = next(gen_source_batch)
                    X1, y1 = next(gen_target_batch)
                    X = np.vstack([X0, X1])
                    y = np.vstack([y0, y1])

                    _, batch_loss, dloss, ploss, d_acc, p_acc = sess.run(
                        [dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
                        feed_dict={self.X: X, self.y: y, self.domain: domain_labels,
                                   self.train: True, self.l: l, learning_rate: lr})
                    feature = sess.run([self.feature],
                                       feed_dict={self.X: X, self.y: y, self.train: False, self.l: l,
                                                  learning_rate: lr})

                    if np.mod(i, 50) == 0:
                        print("Epoch: [%2d] time: %4.4f, batch_loss= %.4f, d_acc= %.4f, p_acc= %.4f" \
                              % (i, time.time() - start_time, batch_loss, d_acc, p_acc))

                elif training_mode == 'source':
                    X, y = next(gen_source_only_batch)
                    _, batch_loss = sess.run([regular_train_op, pred_loss],
                                             feed_dict={self.X: X, self.y: y, self.train: False,
                                                        self.l: l, learning_rate: lr})
                    feature = sess.run([self.feature],
                                       feed_dict={self.X: X, self.y: y, self.train: False,
                                                  self.l: l, learning_rate: lr})
                    if np.mod(i, 50) == 0:
                        print("Epoch: [%2d] time: %4.4f, batch_loss= %.4f" \
                              % (i, time.time() - start_time, batch_loss))

                elif training_mode == 'target':
                    X, y = next(gen_target_only_batch)
                    _, batch_loss = sess.run([regular_train_op, pred_loss],
                                             feed_dict={self.X: X, self.y: y, self.train: False,
                                                        self.l: l, learning_rate: lr})
                    feature = sess.run([self.feature],
                                       feed_dict={self.X: X, self.y: y, self.train: False,
                                                  self.l: l, learning_rate: lr})
                    if np.mod(i, 50) == 0:
                        print("Epoch: [%2d] time: %4.4f, batch_loss= %.4f" \
                              % (i, time.time() - start_time, batch_loss))

            # print(feature)
            # Compute final evaluation on test data
            source_acc = sess.run(label_acc,
                                  feed_dict={self.X: self.mnist_test, self.y: self.mnist.test.labels,
                                             self.train: False})

            target_acc = sess.run(label_acc,
                                  feed_dict={self.X: self.mnistm_test, self.y: self.mnist.test.labels,
                                             self.train: False})

            test_domain_acc = sess.run(domain_acc,
                                       feed_dict={self.X: self.combined_test_imgs,
                                                  self.domain: self.combined_test_domain, self.l: 1.0})

            test_emb = sess.run(self.feature, feed_dict={self.X: self.combined_test_imgs})

        return source_acc, target_acc, test_domain_acc, test_emb, feature