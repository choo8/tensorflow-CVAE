import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import utils

# Latent-feature discriminative model (M1)
class M1(object):
	def __init__ (self, x_dim, z_dim):
		self.x_dim, self.dim_z = x_dim, z_dim
		self.G = tf.Graph()

		# Computation graph of the M1 model
		with self.G.as_default():
			self.x = tf.placeholder(tf.float32, [None, self.x_dim])

			# encoder with 2 hidden layers, each with 600 hidden units
			self.encoder_hidden_w0 = tf.get_variable("M1_encoder_w0", [x_dim, 600], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001, dtype=tf.float32))
			self.encoder_hidden_b0 = tf.get_variable("M1_encoder_b0", [600], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
			self.encoder_hidden_h0 = tf.nn.softplus(tf.matmul(self.x, self.encoder_hidden_w0) + self.encoder_hidden_b0)

			self.encoder_hidden_w1 = tf.get_variable("M1_encoder_w1", [600, 600], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001, dtype=tf.float32))
			self.encoder_hidden_b1 = tf.get_variable("M1_encoder_b1", [600], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
			self.encoder_hidden_h1 = tf.nn.softplus(tf.matmul(self.encoder_hidden_h0, self.encoder_hidden_w1) + self.encoder_hidden_b1)

			# Mean of latent Gaussian posterior
			self.encoder_mu_w = tf.get_variable("M1_encoder_mu_w", [600, z_dim], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001, dtype=tf.float32))
			self.encoder_mu_b = tf.get_variable("M1_encoder_mu_b", [z_dim], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
			self.encoder_mu = tf.matmul(self.encoder_hidden_h1, self.encoder_mu_w) + self.encoder_mu_b

			# Variance of latent Gaussian posterior
			self.encoder_logvar_w = tf.get_variable("M1_encoder_logvar_w", [600, z_dim], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001, dtype=tf.float32))
			self.encoder_logvar_b = tf.get_variable("M1_encoder_logvar_b", [z_dim], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
			self.encoder_logvar = tf.matmul(self.encoder_hidden_h1, self.encoder_logvar_w) + self.encoder_logvar_b

			# Latent variable
			self.z = self.encoder_mu + tf.random_normal(shape=tf.shape(self.encoder_logvar), mean=0.0, stddev=1.0) * tf.exp(0.5 * self.encoder_logvar)

			# decoder with 2 hidden layers, each with 600 hidden units
			self.decoder_hidden_w0 = tf.get_variable("M1_decoder_w0", [z_dim, 600], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001, dtype=tf.float32))
			self.decoder_hidden_b0 = tf.get_variable("M1_decoder_b0", [600], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
			self.decoder_hidden_h0 = tf.nn.softplus(tf.matmul(self.z, self.decoder_hidden_w0) + self.decoder_hidden_b0)

			self.decoder_hidden_w1 = tf.get_variable("M1_decoder_w1", [600, 600], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001, dtype=tf.float32))
			self.decoder_hidden_b1 = tf.get_variable("M1_decoder_b1", [600], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
			self.decoder_hidden_h1 = tf.nn.softplus(tf.matmul(self.decoder_hidden_h0, self.decoder_hidden_w1) + self.decoder_hidden_b1)

			# Reconstructed input
			self.decoder_xhat_w = tf.get_variable("M1_decoder_xhat_w", [600, x_dim], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001, dtype=tf.float32))
			self.decoder_xhat_b = tf.get_variable("M1_decoder_xhat_b", [x_dim], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
			self.decoder_xhat = tf.sigmoid(tf.matmul(self.decoder_hidden_h1, self.decoder_xhat_w) + self.decoder_xhat_b)

			# Objective function
			self.KLD = -0.5 * tf.reduce_sum(1.0 + self.encoder_logvar - tf.pow(self.encoder_mu, 2) - tf.exp(self.encoder_logvar), reduction_indices=1)
			self.likelihood = tf.reduce_sum(self.x * tf.log(tf.clip_by_value(self.decoder_xhat, 1e-10, self.decoder_xhat)) + (1.0 - self.x) * tf.log(tf.clip_by_value(1.0 - self.decoder_xhat, 1e-10, 1.0)) , reduction_indices=1)
			self.loss = tf.reduce_mean(self.KLD - self.likelihood)

			# Evaluate training
			self.eval_likelihood = tf.reduce_mean(self.likelihood)

			self.saver = tf.train.Saver()
			self.session = tf.Session()

	def train(self, x, epochs, batch_size, learning_rate=0.0003, beta1=0.9, beta2=0.999, save_path=None, load_path=None, plot=False):
		self.num_examples = x.shape[0]
		self.batch_size = batch_size
		self.num_batches = self.num_examples // self.batch_size

		if save_path is None:
			self.save_path = "model/VAE.ckpt"
		else:
			self.save_path = save_path

		with self.G.as_default():
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
					_, batch_loss, batch_likelihood = sess.run([self.train_step, self.loss, self.eval_likelihood], feed_dict={self.x: x[batch*self.batch_size:(batch+1)*self.batch_size]})

				if batch_likelihood > best_eval_likelihood:
					best_eval_likelihood = batch_likelihood
					self.saver.save(sess, self.save_path)

				print("Loss at Epoch {}: {}".format(epoch, batch_loss))
				print("Likelihood at Epoch {}: {}".format(epoch, batch_likelihood))

				[sample] = sess.run([self.decoder_xhat], feed_dict={self.x: x[0:25, :]})

				if plot and (epoch % 10 == 0):
					utils.plot_meshgrid(x[0:25, :], sample)