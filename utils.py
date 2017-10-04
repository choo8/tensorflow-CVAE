import matplotlib.pyplot as plt
import numpy as np

def plot_meshgrid(x_original, x_reconstructed):
	f, ax = plt.subplots(1, 2)
	original = np.zeros((28 * 5, 28 * 5))
	reconstructed = np.zeros((28 * 5, 28 * 5))

	for i in range(5):
		for j in range(5):
			original[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_original[i * 5 + j, :].reshape(28, 28)
			reconstructed[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_reconstructed[i * 5 + j, :].reshape(28, 28)
	
	ax[0].imshow(original, cmap="gray")
	ax[0].set(title="Original Images")
	ax[1].imshow(reconstructed, cmap="gray")
	ax[1].set(title="Reconstructed Images")
	plt.show()