from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np

import utils


# Latent-feature discriminative model (M1)
class M1(object):
    def __init__(self, x_dim, z_dim):
        self.x_dim, self.dim_z = x_dim, z_dim
        self.G = tf.Graph()

        # Computation graph of the M1 model
        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
            self.phase = tf.placeholder(tf.bool, name='phase')

            # encoder with 2 hidden layers, each with 600 hidden units
            self.encoder_hidden_w0 = tf.get_variable("M1_encoder_w0", [x_dim, 600],
                                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                              dtype=tf.float32))
            self.encoder_hidden_b0 = tf.get_variable("M1_encoder_b0", [600],
                                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            self.encoder_hidden_h0 = tf.nn.softplus(tf.matmul(self.x, self.encoder_hidden_w0) + self.encoder_hidden_b0)

            self.encoder_hidden_w1 = tf.get_variable("M1_encoder_w1", [600, 600],
                                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                              dtype=tf.float32))
            self.encoder_hidden_b1 = tf.get_variable("M1_encoder_b1", [600],
                                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))

            self.encoder_hidden_z0 = tf.matmul(self.encoder_hidden_h0, self.encoder_hidden_w1) + self.encoder_hidden_b1

            # Batch Normalization before activation
            self.encoder_hidden_z0_batch_norm = tf.layers.batch_normalization(self.encoder_hidden_z0, center=True,
                                                                             scale=True, training=self.phase)

            self.encoder_hidden_h1 = tf.nn.softplus(self.encoder_hidden_z0_batch_norm)

            # Mean of latent Gaussian posterior
            self.encoder_mu_w = tf.get_variable("M1_encoder_mu_w", [600, z_dim],
                                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                         dtype=tf.float32))
            self.encoder_mu_b = tf.get_variable("M1_encoder_mu_b", [z_dim],
                                                initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            self.encoder_mu = tf.matmul(self.encoder_hidden_h1, self.encoder_mu_w) + self.encoder_mu_b

            # Variance of latent Gaussian posterior
            self.encoder_logvar_w = tf.get_variable("M1_encoder_logvar_w", [600, z_dim],
                                                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                             dtype=tf.float32))
            self.encoder_logvar_b = tf.get_variable("M1_encoder_logvar_b", [z_dim],
                                                    initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            self.encoder_logvar = tf.matmul(self.encoder_hidden_h1, self.encoder_logvar_w) + self.encoder_logvar_b

            # Latent variable
            self.z = self.encoder_mu + tf.random_normal(shape=tf.shape(self.encoder_logvar), mean=0.0,
                                                        stddev=1.0) * tf.exp(0.5 * self.encoder_logvar)

            # decoder with 2 hidden layers, each with 600 hidden units
            self.decoder_hidden_w0 = tf.get_variable("M1_decoder_w0", [z_dim, 600],
                                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                              dtype=tf.float32))
            self.decoder_hidden_b0 = tf.get_variable("M1_decoder_b0", [600],
                                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))

            self.decoder_hidden_z0 = tf.matmul(self.z, self.decoder_hidden_w0) + self.decoder_hidden_b0

            # Batch Normalization before activation
            self.decoder_hidden_z0_batch_norm = tf.layers.batch_normalization(self.decoder_hidden_z0, center=True,
                                                                             scale=True, training=self.phase)

            self.decoder_hidden_h0 = tf.nn.softplus(self.decoder_hidden_z0_batch_norm)

            self.decoder_hidden_w1 = tf.get_variable("M1_decoder_w1", [600, 600],
                                                     initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                              dtype=tf.float32))
            self.decoder_hidden_b1 = tf.get_variable("M1_decoder_b1", [600],
                                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            self.decoder_hidden_h1 = tf.nn.softplus(
                tf.matmul(self.decoder_hidden_h0, self.decoder_hidden_w1) + self.decoder_hidden_b1)

            # Reconstructed input
            self.decoder_xhat_w = tf.get_variable("M1_decoder_xhat_w", [600, x_dim],
                                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                           dtype=tf.float32))
            self.decoder_xhat_b = tf.get_variable("M1_decoder_xhat_b", [x_dim],
                                                  initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            self.decoder_xhat = tf.sigmoid(tf.matmul(self.decoder_hidden_h1, self.decoder_xhat_w) + self.decoder_xhat_b)

            # Objective function
            self.KLD = -0.5 * tf.reduce_sum(
                1.0 + self.encoder_logvar - tf.pow(self.encoder_mu, 2) - tf.exp(self.encoder_logvar),
                reduction_indices=1)
            self.likelihood = tf.reduce_sum(
                self.x * tf.log(tf.clip_by_value(self.decoder_xhat, 1e-10, self.decoder_xhat)) + (
                        1.0 - self.x) * tf.log(tf.clip_by_value(1.0 - self.decoder_xhat, 1e-10, 1.0)),
                reduction_indices=1)
            self.loss = tf.reduce_mean(self.KLD - self.likelihood)

            # Evaluate training
            self.eval_likelihood = tf.reduce_mean(self.likelihood)

            self.saver = tf.train.Saver()
            self.session = tf.Session()

    def train(self, x, epochs, batch_size, learning_rate=0.0003, beta1=0.9, beta2=0.999, save_path=None, load_path=None,
              plot=False):
        self.num_examples = x.shape[0]
        self.batch_size = batch_size
        self.num_batches = self.num_examples // self.batch_size

        if save_path is None:
            self.save_path = "model/VAE.ckpt"
        else:
            self.save_path = save_path

        # For Batch Normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with self.G.as_default():
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
                self.train_step = self.optimizer.minimize(self.loss)

        with self.session as sess:
            tf.global_variables_initializer().run()

            if load_path == "default":
                self.saver.restore(sess, self.save_path)
            elif load_path is not None:
                self.saver.restore(sess, load_path)

            for epoch in range(epochs):
                best_eval_likelihood = -np.inf

                np.random.shuffle(x)

                for batch in range(self.num_batches):
                    _, batch_loss, batch_likelihood = sess.run([self.train_step, self.loss, self.eval_likelihood],
                                                               feed_dict={self.x: x[batch * self.batch_size:(
                                                                                                                        batch + 1) * self.batch_size],
                                                                          self.phase: True})

                    if batch_likelihood > best_eval_likelihood:
                        best_eval_likelihood = batch_likelihood
                        self.saver.save(sess, self.save_path)

                print("Loss at Epoch {}: {}".format(epoch, batch_loss))
                print("Likelihood at Epoch {}: {}".format(epoch, batch_likelihood))

                if plot and (epoch % 10 == 0):
                    # Get reconstructions from 25 random data points
                    [sample] = sess.run([self.decoder_xhat], feed_dict={self.x: x[0:25, :], self.phase: True})

                    utils.plot_meshgrid(x[0:25, :], sample, epoch)
                    self.saver.save(sess, self.save_path)

class M2(object):
    def __init__(self, z1_dim, z2_dim, y_dim, num_examples, num_labelled, num_batches, alpha=0.1):
        self.z1_dim, self.z2_dim, self.y_dim = z1_dim, z2_dim, y_dim
        self.num_examples, self.num_labelled, self.num_batches = num_examples, num_labelled, num_batches
        self.num_unlabelled = num_examples - num_labelled
        self.batch_size = num_examples // num_batches
        self.num_labelled_batch = num_labelled // num_batches
        self.num_unlabelled_batch = self.num_unlabelled // num_batches
        self.alpha = alpha * (float(self.batch_size) / self.num_labelled_batch)

        self.G = tf.Graph()

        with self.G.as_default():
            self.labelled_z1_mu = tf.placeholder(tf.float32, [None, z1_dim])
            self.labelled_z1_logvar = tf.placeholder(tf.float32, [None, z1_dim])

            self.unlabelled_z1_mu = tf.placeholder(tf.float32, [None, z1_dim])
            self.unlabelled_z1_logvar = tf.placeholder(tf.float32, [None, z1_dim])

            self.y = tf.placeholder(tf.float32, [None, y_dim])

            # Labelled points

            # Classifier network with 1 hidden layer, 500 hidden units
            self.labelled_z1 = self.labelled_z1_mu + tf.random_normal(shape=tf.shape(self.labelled_z1_logvar),
                                                                      mean=0.0,
                                                                      stddev=1.0) * tf.exp(
                0.5 * self.labelled_z1_logvar)

            self.classifier_hidden_w0 = tf.get_variable("M2_classifier_w0", [z1_dim, 500],
                                                        initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                 stddev=0.001,
                                                                                                 dtype=tf.float32))
            self.classifier_hidden_b0 = tf.get_variable("M2_classifier_b0", [500],
                                                        initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                 stddev=0.001,
                                                                                                 dtype=tf.float32))
            self.classifier_hidden_h0 = tf.nn.softplus(
                tf.matmul(self.labelled_z1, self.classifier_hidden_w0) + self.classifier_hidden_b0)

            self.classifier_y_w = tf.get_variable("M2_classifier_y_w", [500, y_dim],
                                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                           dtype=tf.float32))
            self.classifier_y_b = tf.get_variable("M2_classifier_y_b", [y_dim],
                                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                           dtype=tf.float32))
            self.classifier_y = tf.nn.softmax(
                tf.matmul(self.classifier_hidden_h0, self.classifier_y_w) + self.classifier_y_b)

            # self.classifier_y = tf.Print(self.classifier_y, [self.classifier_y], "q(y|x)", summarize=10)

            # Encoder network with 1 hidden layer, 500 hidden units
            self.encoder_combined_input = tf.concat([self.labelled_z1, self.y], 1)

            self.encoder_hidden_w0 = tf.get_variable("M2_encoder_w0", [z1_dim + y_dim, 500],
                                                     initializer=tf.random_normal_initializer(mean=0.0,
                                                                                              stddev=0.001,
                                                                                              dtype=tf.float32))
            self.encoder_hidden_b0 = tf.get_variable("M2_encoder_b0", [500],
                                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            self.encoder_hidden_h0 = tf.nn.softplus(
                tf.matmul(self.encoder_combined_input, self.encoder_hidden_w0) + self.encoder_hidden_b0)

            self.encoder_z2_mu_w = tf.get_variable("M2_encoder_z2_mu_w", [500, z2_dim],
                                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                            dtype=tf.float32))
            self.encoder_z2_mu_b = tf.get_variable("M2_encoder_z2_mu_b", [z2_dim],
                                                   initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001,
                                                                                            dtype=tf.float32))
            self.encoder_z2_mu = tf.matmul(self.encoder_hidden_h0, self.encoder_z2_mu_w) + self.encoder_z2_mu_b

            self.encoder_z2_logvar_w = tf.get_variable("M2_encoder_z2_logvar_w", [500, z2_dim],
                                                       initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                stddev=0.001,
                                                                                                dtype=tf.float32))
            self.encoder_z2_logvar_b = tf.get_variable("M2_encoder_z2_logvar_b", [z2_dim],
                                                       initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                stddev=0.001,
                                                                                                dtype=tf.float32))
            self.encoder_z2_logvar = tf.matmul(self.encoder_hidden_h0,
                                               self.encoder_z2_logvar_w) + self.encoder_z2_logvar_b

            self.encoder_z2 = self.encoder_z2_mu + tf.random_normal(shape=tf.shape(self.encoder_z2_logvar),
                                                                    mean=0.0,
                                                                    stddev=1.0) * tf.exp(
                0.5 * self.encoder_z2_logvar)

            # Decoder network with 1 hidden layer, 500 hidden units
            self.decoder_combined_input = tf.concat([self.encoder_z2, self.y], 1)

            self.decoder_hidden_w0 = tf.get_variable("M2_decoder_w0", [z2_dim + y_dim, 500],
                                                     initializer=tf.random_normal_initializer(mean=0.0,
                                                                                              stddev=0.001,
                                                                                              dtype=tf.float32))
            self.decoder_hidden_b0 = tf.get_variable("M2_decoder_b0", [500],
                                                     initializer=tf.constant_initializer(0.0, dtype=tf.float32))
            self.decoder_hidden_h0 = tf.nn.softplus(
                tf.matmul(self.decoder_combined_input, self.decoder_hidden_w0) + self.decoder_hidden_b0)

            self.decoder_z1hat_mu_w = tf.get_variable("M2_decoder_z1hat_mu_w", [500, z1_dim],
                                                      initializer=tf.random_normal_initializer(mean=0.0,
                                                                                               stddev=0.001,
                                                                                               dtype=tf.float32))
            self.decoder_z1hat_mu_b = tf.get_variable("M2_decoder_z1hat_mu_b", [z1_dim],
                                                      initializer=tf.random_normal_initializer(mean=0.0,
                                                                                               stddev=0.001,
                                                                                               dtype=tf.float32))
            self.decoder_z1hat_mu = tf.matmul(self.decoder_hidden_h0,
                                              self.decoder_z1hat_mu_w) + self.decoder_z1hat_mu_b

            self.decoder_z1hat_logvar_w = tf.get_variable("M2_decoder_z1hat_logvar_w", [500, z1_dim],
                                                          initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                   stddev=0.001,
                                                                                                   dtype=tf.float32))
            self.decoder_z1hat_logvar_b = tf.get_variable("M2_decoder_z1hat_logvar_b", [z1_dim],
                                                          initializer=tf.random_normal_initializer(mean=0.0,
                                                                                                   stddev=0.001,
                                                                                                   dtype=tf.float32))
            self.decoder_z1hat_logvar = tf.matmul(self.decoder_hidden_h0,
                                                  self.decoder_z1hat_logvar_w) + self.decoder_z1hat_logvar_b

            # Objective function
            self.Lxy = self.L([self.decoder_z1hat_mu, self.decoder_z1hat_logvar], self.labelled_z1, self.y,
                              [self.encoder_z2, self.encoder_z2_mu, self.encoder_z2_logvar])

            # self.Lxy = tf.Print(self.Lxy, [self.Lxy], "Labelled loss")

            # Unlabelled points

            # Reuse weights from classifier network with 1 hidden layer, 500 hidden units
            self.unlabelled_z1 = self.unlabelled_z1_mu + tf.random_normal(shape=tf.shape(self.unlabelled_z1_logvar),
                                                                          mean=0.0, stddev=1.0) * tf.exp(
                0.5 * self.unlabelled_z1_logvar)
            self.predictor_hidden_h0 = tf.nn.softplus(
                tf.matmul(self.unlabelled_z1, self.classifier_hidden_w0) + self.classifier_hidden_b0)
            self.predictor_y = tf.nn.softmax(
                tf.matmul(self.predictor_hidden_h0, self.classifier_y_w) + self.classifier_y_b)

            # self.predictor_y = tf.Print(self.predictor_y, [self.predictor_y], "q(y|x) from unlabelled data", summarize=10)

            self.unlabelled_z1_tiled = tf.tile(self.unlabelled_z1, [self.y_dim, 1])
            self.unlabelled_y = tf.reshape(tf.tile(tf.eye(self.y_dim), [1, self.num_unlabelled_batch]),
                                           [-1, self.y_dim])

            # Encode unlabelled z1 inputs and y = label into z2 outputs
            self.encoder_combined_unlabelled_input = tf.concat([self.unlabelled_z1_tiled, self.unlabelled_y], 1)
            self.encoder_unlabelled_hidden_h0 = tf.nn.softplus(
                tf.matmul(self.encoder_combined_unlabelled_input, self.encoder_hidden_w0) + self.encoder_hidden_b0)
            self.encoder_unlabelled_z2_mu = tf.matmul(self.encoder_unlabelled_hidden_h0,
                                                      self.encoder_z2_mu_w) + self.encoder_z2_mu_b
            self.encoder_unlabelled_z2_logvar = tf.matmul(self.encoder_hidden_h0,
                                                          self.encoder_z2_logvar_w) + self.encoder_z2_logvar_b
            self.encoder_unlabelled_z2 = self.encoder_unlabelled_z2_mu + tf.random_normal(
                shape=tf.shape(self.encoder_unlabelled_z2_logvar), mean=0.0, stddev=1.0) * tf.exp(
                0.5 * self.encoder_unlabelled_z2_logvar)

            # Decode z2 outputs which came from unlabelled z1 inputs and y = label
            self.decoder_combined_unlabelled_input = tf.concat([self.encoder_unlabelled_z2, self.unlabelled_y], 1)
            self.decoder_unlabelled_hidden_h0 = tf.nn.softplus(
                tf.matmul(self.decoder_combined_unlabelled_input, self.decoder_hidden_w0) + self.decoder_hidden_b0)
            self.decoder_unlabelled_z1hat_mu = tf.matmul(self.decoder_unlabelled_hidden_h0,
                                                         self.decoder_z1hat_mu_w) + self.decoder_z1hat_mu_b
            self.decoder_unlabelled_z1hat_logvar = tf.matmul(self.decoder_unlabelled_hidden_h0,
                                                             self.decoder_z1hat_logvar_w) + self.decoder_z1hat_logvar_b

            self.L_unlabelled_tiled = self.L(
                [self.decoder_unlabelled_z1hat_mu, self.decoder_unlabelled_z1hat_logvar],
                self.unlabelled_z1_tiled, self.unlabelled_y,
                [self.encoder_unlabelled_z2, self.encoder_unlabelled_z2_mu,
                 self.encoder_unlabelled_z2_logvar])
            self.L_unlabelled = tf.transpose(
                tf.reshape(self.L_unlabelled_tiled, [self.y_dim, self.num_unlabelled_batch]))
            self.L_unlabelled = tf.reduce_sum(self.predictor_y * self.L_unlabelled, axis=-1)

            self.post_y_entropy = -tf.reduce_sum(self.predictor_y * tf.log(self.predictor_y + 1e-10), axis=-1)
            self.U = self.L_unlabelled - self.post_y_entropy

            # self.U = tf.Print(self.U, [self.U], "U")

            # for label in range(y_dim):
            # 	self.unlabelled_y = utils.one_hot_tensor(label, self.y_dim, self.num_unlabelled_batch)

            # 	# Encode unlabelled z1 inputs and y = label into z2 outputs
            # 	self.encoder_combined_unlabelled_input = tf.concat([self.unlabelled_z1, self.unlabelled_y], 1)
            # 	self.encoder_unlabelled_hidden_h0 = tf.nn.softplus(tf.matmul(self.encoder_combined_unlabelled_input, self.encoder_hidden_w0) + self.encoder_hidden_b0)
            # 	self.encoder_unlabelled_z2_mu = tf.matmul(self.encoder_unlabelled_hidden_h0, self.encoder_z2_mu_w) + self.encoder_z2_mu_b
            # 	self.encoder_unlabelled_z2_logvar = tf.matmul(self.encoder_hidden_h0, self.encoder_z2_logvar_w) + self.encoder_z2_logvar_b
            # 	self.encoder_unlabelled_z2 = self.encoder_unlabelled_z2_mu + tf.random_normal(shape=tf.shape(self.encoder_unlabelled_z2_logvar), mean=0.0, stddev=1.0) * tf.exp(0.5 * self.encoder_unlabelled_z2_logvar)

            # 	# Decode z2 outputs which came from unlabelled z1 inputs and y = label
            # 	self.decoder_combined_unlabelled_input = tf.concat([self.encoder_unlabelled_z2, self.unlabelled_y], 1)
            # 	self.decoder_unlabelled_hidden_h0 = tf.nn.softplus(tf.matmul(self.decoder_combined_unlabelled_input, self.decoder_hidden_w0) + self.decoder_hidden_b0)
            # 	self.decoder_unlabelled_z1hat_mu = tf.matmul(self.decoder_unlabelled_hidden_h0, self.decoder_z1hat_mu_w) + self.decoder_z1hat_mu_b
            # 	self.decoder_unlabelled_z1hat_logvar = tf.matmul(self.decoder_unlabelled_hidden_h0, self.decoder_z1hat_logvar_w) + self.decoder_z1hat_logvar_b

            # 	cur_L_unlabelled = tf.expand_dims(self.L([self.decoder_unlabelled_z1hat_mu, self.decoder_unlabelled_z1hat_logvar], self.unlabelled_z1, self.unlabelled_y, [self.encoder_unlabelled_z2, self.encoder_unlabelled_z2_mu, self.encoder_unlabelled_z2_logvar]), 1)

            # 	# Sum up L over all possible labels of y
            # 	if label == 0:
            # 		self.L_unlabelled = tf.identity(cur_L_unlabelled)
            # 	else:
            # 		self.L_unlabelled = tf.concat([self.L_unlabelled, cur_L_unlabelled], 1)

            # self.U = tf.reduce_sum(tf.nn.softmax(self.predictor_y) * (self.L_unlabelled - tf.log(tf.nn.softmax(self.predictor_y))), reduction_indices=1)

            # self.cost = -(tf.reduce_mean(self.Lxy) + tf.reduce_mean(self.U) + (self.alpha * tf.reduce_mean(
            #    -tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.classifier_y))))
            self.cost = tf.reduce_mean(self.Lxy) + tf.reduce_mean(self.U) + (self.alpha * tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.classifier_y)))

            self.saver = tf.train.Saver()
            self.session = tf.Session()

    def L(self, z1hat, z1, y, z2):
        log_prior_y = -tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=tf.ones_like(y))
        log_likelihood = -0.5 * (
                tf.reduce_sum(z1hat[1], axis=-1) + tf.reduce_sum(((z1 - z1hat[0]) ** 2) / tf.exp(z1hat[1]),
                                                                 axis=-1) + tf.cast(self.z1_dim,
                                                                                    tf.float32) * tf.log(
            2.0 * np.pi))
        log_prior_z2 = -0.5 * (
                tf.reduce_sum((z2[1] ** 2) + tf.exp(z2[2]), axis=-1) + tf.cast(self.z1_dim, tf.float32) * tf.log(
            2.0 * np.pi))
        log_post_z2 = -0.5 * (
                tf.reduce_sum(1.0 + z2[2], axis=-1) + tf.cast(self.z1_dim, tf.float32) * tf.log(2.0 * np.pi))

        return -1.0 * (log_prior_y + log_likelihood + log_prior_z2 - log_post_z2)

    def train(self, z1, y, unlabelled_z1, epochs, z1_valid, y_valid, learning_rate=0.0003, beta1=0.9, beta2=0.999,
              save_path=None, load_path=None):
        if save_path is None:
            self.save_path = "model/GC.cpkt"
        else:
            self.save_path = save_path

        with self.G.as_default():
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
            self.train_step = self.optimizer.minimize(self.cost)

        labelled_data = np.hstack([z1, y])
        unlabelled_data = unlabelled_z1
        z1_valid_mu, z1_valid_logvar = z1_valid[:, :self.z1_dim], z1_valid[:, self.z1_dim:2 * self.z1_dim]

        with self.session as sess:
            tf.global_variables_initializer().run()

            if load_path == "default":
                self.saver.restore(sess, self.save_path)
            elif load_path is not None:
                self.saver.restore(sess, load_path)

            for epoch in range(epochs):
                np.random.shuffle(labelled_data)
                np.random.shuffle(unlabelled_data)

                for batch in range(self.num_batches):
                    _, batch_cost, batch_likelihood = sess.run([self.train_step, self.cost, self.Lxy], feed_dict={
                        self.labelled_z1_mu: labelled_data[
                                             batch * self.num_labelled_batch:(batch + 1) * self.num_labelled_batch,
                                             :self.z1_dim],
                        self.labelled_z1_logvar: labelled_data[
                                                 batch * self.num_labelled_batch:(
                                                                                         batch + 1) * self.num_labelled_batch,
                                                 self.z1_dim:2 * self.z1_dim],
                        self.unlabelled_z1_mu: unlabelled_data[batch * self.num_unlabelled_batch:(
                                                                                                         batch + 1) * self.num_unlabelled_batch,
                                               :self.z1_dim],
                        self.unlabelled_z1_logvar: unlabelled_data[batch * self.num_unlabelled_batch:(
                                                                                                             batch + 1) * self.num_unlabelled_batch,
                                                   self.z1_dim:2 * self.z1_dim],
                        self.y: labelled_data[batch * self.num_labelled_batch:(batch + 1) * self.num_labelled_batch,
                                2 * self.z1_dim:]})

                # Get validation accuracy
                pred_y = sess.run([self.classifier_y], feed_dict={self.labelled_z1_mu: z1_valid[:, :self.z1_dim],
                                                                  self.labelled_z1_logvar: z1_valid[:,
                                                                                           self.z1_dim:2 * self.z1_dim]})
                pred_y = np.argmax(np.squeeze(pred_y), axis=1)
                batch_valid_accuracy = accuracy_score(np.argmax(y_valid, axis=1), pred_y)

                print("Loss at Epoch {}: {}".format(epoch, batch_cost))
                print("Likelihood at Epoch {}: {}".format(epoch, batch_likelihood))
                print("Validation Accuracy at Epoch {}: {}".format(epoch, batch_valid_accuracy))
