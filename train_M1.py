import data.mnist as mnist #https://github.com/dpkingma/nips14-ssl
from models import M1
import matplotlib.pyplot as plt

def main():
	z_dim = 50
	epochs = 100
	learning_rate = 0.0003
	batch_size = 50

	mnist_path = 'mnist/mnist_28.pkl.gz'
	#Uses anglpy module from original paper (linked at top) to load the dataset
	train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy(mnist_path, binarize_y=True)

	x_train, y_train = train_x.T, train_y.T
	x_valid, y_valid = valid_x.T, valid_y.T
	x_test, y_test = test_x.T, test_y.T

	x_dim = x_train.shape[1]
	y_dim = y_train.shape[1]

	print(x_train.shape)
	#plt.imshow(x_train[0, :].reshape((28, 28)), cmap="gray")
	#plt.show()

	vae = M1(x_dim=x_dim, z_dim=z_dim)
	vae.train(x=x_train, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, plot=True)

if __name__ == "__main__":
	main()