from mlxtend.data import loadlocal_mnist
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

train_images, train_labels = loadlocal_mnist(
          images_path='D:/Python Projects/MNIST_With_Moments/mnist_data/train-images.idx3-ubyte',
          labels_path='D:/Python Projects/MNIST_With_Moments/mnist_data/train-labels.idx1-ubyte')

test_images, test_labels = loadlocal_mnist(
          images_path='D:/Python Projects/MNIST_With_Moments/mnist_data/t10k-images.idx3-ubyte',
          labels_path='D:/Python Projects/MNIST_With_Moments/mnist_data/t10k-labels.idx1-ubyte')

M = len(train_labels)
#
# a = 0
# b = 0
# c = 0
# d = 0
# e = 0
# f = 0
# g = 0
# h = 0
# i = 0
# j = 0
#
# for k in range(0, 1000):
#     if(train_labels[k] == 0 ):
#         a = a + 1
#     if (train_labels[k] == 1):
#         b = b + 1
#     if (train_labels[k] == 2):
#         c = c + 1
#     if (train_labels[k] == 3):
#         d = d + 1
#     if (train_labels[k] == 4):
#         e = e + 1
#     if (train_labels[k] == 5):
#         f = f + 1
#     if (train_labels[k] == 6):
#         g = g + 1
#     if (train_labels[k] == 7):
#         h = h + 1
#     if (train_labels[k] == 8):
#         i = i + 1
#     if (train_labels[k] == 9):
#         j = j + 1

# x=4
for k in range(0,99):
    first_image = test_images[k]
    print("iterationtest "+str(k)+" : " +str(test_labels[k]))
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()


    #
    # plt.figure()
    # plt.clf()       #clear figure
    # plt.scatter(x = prediction ,y=labels, s=labels, zorder=10, cmap=plt.cm.Paired,
    #             edgecolor='k',)
    #
    # # # Circle out the test data
    # # plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
    # #             zorder=10, edgecolor='k')
    #
    # plt.axis('tight')
    # # x_min = prediction[,:].min()
    # # x_max = prediction[:, 0].max()
    # # y_min = prediction[:, 1].min()
    # # y_max = prediction[:, 1].max()
    #
    # XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    # Z = plt.clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    #
    # # Put the result into a color plot
    # Z = Z.reshape(XX.shape)
    # plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    # plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
    #             linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    #
    # # plt.title(kernel)
    # plt.show()

    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Plot the surface.
    # surf = ax.plot_surface(C, GAMMA, PERCISION, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    #
    # # Customize the z axis.
    # ax.set_zlim(0, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # plt.show()