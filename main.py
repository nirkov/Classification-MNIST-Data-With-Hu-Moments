import sklearn.metrics
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import importData
import predict_svm_model
import statistic_printer
import train_svm_model
import matplotlib as plt
import numpy as np
from matplotlib import *
from pylab import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

def main():
    #
    train_images, train_labels, test_images, test_labels = importData.load_data(data_address = 'D:/Python Projects/MNIST_With_Moments/mnist_data')
    classifier = train_svm_model.train_model_RBF_kernel(num_train=5000, images=train_images, tag=train_labels, gamma_value=2,
                                                        num_iteretion=-1, c_value=50, log_transform=True, RAM_size=8000)
    prediction, labels = predict_svm_model.predict(clf=classifier, num_test=100, images=test_images, tag=test_labels)

    gamma = 0.5
    c = 1

    print("------------------------------ Training SVM model on MNIST data with HuMomnt only -------------------------------------")
    print()
    print("************************************************************************************************************************")
    print("                                 Training with RBF kernel - gamma is variable")
    print("************************************************************************************************************************")
    print()
    while(gamma < 10 ):
        gamma = gamma + 0.5
        print("gamma value = "+str(gamma))

        gamma_variable_classifier = train_svm_model.train_model_RBF_kernel(num_train=1000, images=train_images, tag=train_labels, gamma_value=gamma,
                                                        num_iteretion=-1, c_value=50, log_transform=True, RAM_size=8000)
        prediction, labels = predict_svm_model.predict(clf=gamma_variable_classifier, num_test=1000, images=test_images,tag=test_labels)
        statistic_printer.print_SVM_HuMoment_statistic(labels, prediction, gamma_variable_classifier)
        statistic_printer.save_SVM_HuMoment_csvFile(labels, prediction, "gamma - variable",'gamma', gamma)
        print("____________________________________________________________________________________________________________________")
        print()

    # print()
    #
    # print("************************************************************************************************************************")
    # print("                                 Training with RBF kernel - C is variable")
    # print("************************************************************************************************************************")
    # print()
    # while(c < 100 ):
    #     c = c + 5
    #     print("C value = " + str(c))
    #     gamma_variable_classifier = train_svm_model.train_model_RBF_kernel(num_train=10000, images=train_images, tag=train_labels, gamma_value=2,
    #                                                     num_iteretion=-1, c_value= c, log_transform=True, RAM_size=8000)
    #     prediction, labels = predict_svm_model.predict(clf=gamma_variable_classifier, num_test=1000, images=test_images,tag=test_labels)
    #     statistic_printer.print_SVM_HuMoment_statistic(labels, prediction, gamma_variable_classifier)
    #     statistic_printer.save_SVM_HuMoment_csvFile(labels, prediction, "C - variable",'C', c)
    #     print("____________________________________________________________________________________________________________________")
    #     print()

    # print()
    # print("************************************************************************************************************************")
    # print("                                Training with RBF kernel - Percision(C,gamma,iteration")
    # print("************************************************************************************************************************")
    # print()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    GAMMA = []
    C = []
    PERCISION = []
    for gamma in range(1,6):
        for c in range(1, 6):

            gamma_variable_classifier = train_svm_model.train_model_RBF_kernel(num_train=1000, images=train_images,
                                                                               tag=train_labels, gamma_value=gamma,
                                                                               num_iteretion=2000, c_value=c,
                                                                               log_transform=True, RAM_size=8000)
            prediction, labels = predict_svm_model.predict(clf=gamma_variable_classifier, num_test=1000,
                                                           images=test_images, tag=test_labels)
            GAMMA.append(gamma)
            C.append(c)
            avr = 0
            for i in range (0,1000):
                if(prediction[i]==labels[i]):
                    avr = avr + 1
            avr = (avr/1000)*100
            PERCISION.append(avr)

    GAMMA = np.array(GAMMA)
    C = np.array(C)
    PERCISION = np.array(PERCISION)

    ax.scatter(C, GAMMA, PERCISION, c='r', marker='o')

    ax.set_xlabel('c')
    ax.set_ylabel('gamma')
    ax.set_zlabel('percision')

    plt.show()


if __name__ == "__main__":
        main()








   # fname = 'output.csv'
    # with open(fname, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     for row in sklearn.metrics.confusion_matrix(labels, prediction):
    #         writer.writerow(row)
