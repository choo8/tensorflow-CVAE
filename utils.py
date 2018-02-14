import imageio
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_meshgrid(x_original, x_reconstructed, epoch):
    f, ax = plt.subplots(1, 2)
    original = np.zeros((28 * 5, 28 * 5))
    reconstructed = np.zeros((28 * 5, 28 * 5))

    for i in range(5):
        for j in range(5):
            original[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_original[i * 5 + j, :].reshape(28, 28)
            reconstructed[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_reconstructed[i * 5 + j, :].reshape(28, 28)

    ax[0].imshow(original, cmap="gray")
    ax[0].set(title="Original Images")
    ax[0].axis('off')
    ax[1].imshow(reconstructed, cmap="gray")
    ax[1].set(title="Reconstructed Images")
    ax[1].axis('off')
    plt.suptitle("Epoch " + str(epoch))

    # Save plot to make GIF
    plt.savefig('temp/Epoch ' + str(epoch))


def create_gif(output_filename):
    filenames = os.listdir(os.path.join(os.getcwd(), 'temp'))
    filenames = sorted(filenames, key=lambda x: x[6])
    images = []

    for filename in filenames:
        images.append(imageio.imread(os.path.join('temp', filename)))

    kwargs = {'fps': 1}
    imageio.mimwrite(output_filename, images, 'GIF-PIL', **kwargs)

