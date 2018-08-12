from mlxtend.data import loadlocal_mnist
import cv2

from numpy import *


def load_data(data_address):
     train_images, train_labels = loadlocal_mnist(
               images_path=data_address + '/train-images.idx3-ubyte',
               labels_path=data_address + '/train-labels.idx1-ubyte')

     test_images, test_labels = loadlocal_mnist(
               images_path=data_address + '/t10k-images.idx3-ubyte',
               labels_path=data_address + '/t10k-labels.idx1-ubyte')

     return  train_images, train_labels,  test_images, test_labels



# t = cv2.HuMoments(cv2.moments(train_images)).flatten()

  # x=4
     # first_image = test_images[37]
     # print(test_labels[37])
     # first_image = np.array(first_image, dtype='float')
     # pixels = first_image.reshape((28, 28))
     # plt.imshow(pixels, cmap='gray')
     # plt.show()