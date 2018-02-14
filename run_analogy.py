import data.mnist as mnist  # https://github.com/dpkingma/nips14-ssl
from models import M1, M2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def main():
    num_labelled = 100
    num_batches = 100
    z_dim = 50
    alpha = 0.1

    mnist_path = 'mnist/mnist_28.pkl.gz'
    # Uses anglpy module from original paper (linked at top) to split the dataset for semi-supervised training
    train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy_split(mnist_path, binarize_y=True)
    x_l, y_l, x_u, y_u = mnist.create_semisupervised(train_x, train_y, num_labelled)

    x_lab, y_lab = x_l.T, y_l.T
    x_ulab, y_ulab = x_u.T, y_u.T

    x_dim = x_lab.shape[1]
    y_dim = y_lab.shape[1]

    # Get 5 random 50D z vectors
    rand_vec = np.random.normal(scale=1.5, size=(10, 50))
    rand_vec_tile = np.tile(rand_vec, 10)

    y_vec = np.eye(10)
    y_vec_tile = np.tile(y_vec, 10)

    z2_vec = np.zeros((10,500))

    M2_model_path = "./model_M2/GC.cpkt"
    M2_GC = M2(z1_dim=z_dim, z2_dim=z_dim, y_dim=y_dim, num_examples=x_lab.shape[0] + x_ulab.shape[0],
               num_labelled=num_labelled, num_batches=num_batches, alpha=alpha)
    with M2_GC.session:
        M2_GC.saver.restore(M2_GC.session, M2_model_path)

        for i in range(10):
            cur_y = y_vec[i]
            [sample_mu, sample_logvar] = M2_GC.session.run([M2_GC.decoder_z1hat_mu, M2_GC.decoder_z1hat_logvar],
                                                           feed_dict={M2_GC.decoder_combined_input: np.hstack((
                                                               rand_vec_tile[:, i * z_dim:(i + 1) * z_dim],
                                                               y_vec_tile[:, i * y_dim:(i + 1) * y_dim]))})

            sample = sample_mu + np.random.normal(size=(10, 50))*np.exp(0.5*sample_logvar)
            z2_vec[:, i*50:(i+1)*50] = sample

    tf.reset_default_graph()

    # Plot for analogy
    plot_arr = np.zeros((10 * 28, 10 * 28))

    M1_model_path = "./model_M1/VAE.ckpt"
    M1_vae = M1(x_dim=x_dim, z_dim=z_dim)
    with M1_vae.session:
        M1_vae.saver.restore(M1_vae.session, M1_model_path)

        for i in range(10):
            cur_z = z2_vec[:, i*50:(i+1)*50]
            [generated_img] = M1_vae.session.run([M1_vae.decoder_xhat], feed_dict={M1_vae.z: cur_z, M1_vae.phase: True})

            plot_arr[:,i*28:(i+1)*28] = np.reshape(generated_img, (10*28, 28))

    plt.imshow(plot_arr, cmap="gray")
    plt.title("MNIST analogies")
    plt.savefig('temp/analogy.png', format='png')


if __name__ == "__main__":
    main()
