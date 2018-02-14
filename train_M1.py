import data.mnist as mnist  # https://github.com/dpkingma/nips14-ssl
from models import M1
import numpy as np
import utils
import argparse
import tensorflow as tf


def main():
    # Use argparse to decide if user wants to re-train VAE
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action='store_true')
    args = parser.parse_args()

    z_dim = 50
    epochs = 500
    learning_rate = 0.0003
    batch_size = 50

    mnist_path = 'mnist/mnist_28.pkl.gz'
    # Uses anglpy module from original paper (linked at top) to load the dataset
    train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy(mnist_path, binarize_y=True)
    x_all = np.concatenate([train_x.T, valid_x.T, test_x.T], axis=0)
    y_all = np.concatenate([train_y.T, valid_y.T, test_y.T], axis=0)

    x_dim = x_all.shape[1]

    # Visualize how results change over time in the form of a GIF
    utils.create_gif("M1_model.gif")

    # Specify model path and setup VAE object
    M1_model_path = "./model/VAE.ckpt"
    vae = M1(x_dim=x_dim, z_dim=z_dim)

    # Train if user specifies with keyword
    if args.train:
        vae.train(x=x_all, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, plot=True)

    # Visualize the latent space
    tf.reset_default_graph()
    with vae.session:
        vae.saver.restore(vae.session, M1_model_path)
        [sample, latent] = vae.session.run([vae.decoder_xhat, vae.z], feed_dict={vae.x: x_all, vae.phase: True})

        # Plot meshgrid at Epoch 500
        utils.plot_meshgrid(x_all[0:25,:], sample[0:25,:], epochs)


if __name__ == "__main__":
    main()

