import data.mnist as mnist  # https://github.com/dpkingma/nips14-ssl
from models import M1, M2
import argparse
import numpy as np


def main():
    # Use argparse to decide if user wants to re-train VAE
    parser = argparse.ArgumentParser()
    parser.add_argument("-train", action='store_true')
    args = parser.parse_args()

    num_labelled = 100
    num_batches = 100
    z_dim = 50
    epochs = 1000
    learning_rate = 0.0003
    alpha = 0.1

    mnist_path = 'mnist/mnist_28.pkl.gz'
    # Uses anglpy module from original paper (linked at top) to split the dataset for semi-supervised training
    train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy_split(mnist_path, binarize_y=True)
    x_l, y_l, x_u, y_u = mnist.create_semisupervised(train_x, train_y, num_labelled)

    x_lab, y_lab = x_l.T, y_l.T
    x_ulab, y_ulab = x_u.T, y_u.T
    x_valid, y_valid = valid_x.T, valid_y.T
    x_test, y_test = test_x.T, test_y.T

    x_dim = x_lab.shape[1]
    y_dim = y_lab.shape[1]

    # Restore previously trained VAE, M1, and get parameters of encoded latent variable z from image as input for M2
    M1_model_path = "./model_M1/VAE.ckpt"
    M1_vae = M1(x_dim=x_dim, z_dim=z_dim)
    with M1_vae.session:
        M1_vae.saver.restore(M1_vae.session, M1_model_path)

        z1_mu_lab, z1_logvar_lab = M1_vae.session.run([M1_vae.encoder_mu, M1_vae.encoder_logvar],
                                                      feed_dict={M1_vae.x: x_lab, M1_vae.phase: True})
        z1_mu_ulab, z1_logvar_ulab = M1_vae.session.run([M1_vae.encoder_mu, M1_vae.encoder_logvar],
                                                        feed_dict={M1_vae.x: x_ulab, M1_vae.phase: True})
        z1_mu_valid, z1_logvar_valid = M1_vae.session.run([M1_vae.encoder_mu, M1_vae.encoder_logvar],
                                                          feed_dict={M1_vae.x: x_valid, M1_vae.phase: True})

    M2_model_path = "./model_M2/GC.ckpt"
    M2_vae = M2(z1_dim=z_dim, z2_dim=z_dim, y_dim=y_dim, num_examples=x_lab.shape[0] + x_ulab.shape[0],
                num_labelled=num_labelled, num_batches=num_batches, alpha=alpha)

    if args.train:
        M2_vae.train(z1=np.hstack([z1_mu_lab, z1_logvar_lab]), y=y_lab,
                 unlabelled_z1=np.hstack([z1_mu_ulab, z1_logvar_ulab]), epochs=epochs,
                 z1_valid=np.hstack([z1_mu_valid, z1_logvar_valid]), y_valid=y_valid)


if __name__ == "__main__":
    main()
