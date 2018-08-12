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

    """"
    Change the path that matches the file location on your computer to import MNIST data.
    """
    train_images, train_labels, test_images, test_labels = importData.load_data(data_address = 'D:/Python Projects/MNIST_With_Moments/mnist_data')


    gamma = 0.1
    c = 1
    iteration = 100
    max_average_gamma = 0
    max_gamma = 0
    max_average_c = 0
    max_c = 0
    max_average_iter = 0
    max_iter = 0

    print("------------------------------ Training SVM model on MNIST data with HuMomnt only -------------------------------------")
    print()
    print("************************************************************************************************************************")
    print("                                 Training with RBF kernel - gamma is variable")
    print("************************************************************************************************************************")
    print()

    while(gamma < 10.2 ):
        gamma = gamma + 0.5
        print("gamma value = "+str(gamma))
        # make svm model's classifier. Algorithm training is done using the features obtained from calculating humoment
        # images.
        gamma_variable_classifier = train_svm_model.train_model_RBF_kernel(num_train=30000, images=train_images, tag=train_labels, gamma_value=gamma,
                                                        num_iteretion=-1, c_value=80, log_transform=True, RAM_size=8000)
        # Predicting the test data.
        prediction, labels = predict_svm_model.predict(clf=gamma_variable_classifier, num_test=3000, images=test_images,tag=test_labels)
        # Print statistic using 'classification_report' function of sklearn library
        statistic_printer.print_SVM_HuMoment_statistic(labels, prediction, gamma_variable_classifier)
        # Save the calculated statistic in csv (exl) file
        statistic_printer.save_SVM_HuMoment_csvFile(labels, prediction, "gamma - variable", 'gamma', gamma)
        # Save the maximum accuracy and the corresponding value of gamma
        max_average_gamma, max_gamma = statistic_printer.make_max_average(prediction, labels, 3000, max_average_gamma, max_gamma, gamma)
        print("_______________________________________________________________________________________________________")
        print()

    print()
    print("max_average_gamma : "+str(max_average_gamma))
    print("max_gamma : "+str(max_gamma))
    print()
    print("***********************************************************************************************************")
    print("                                 Training with RBF kernel - C is variable")
    print("***********************************************************************************************************")
    print()

    while(c < 102 ):
        c = c + 5
        print("C value = " + str(c))
        # make svm model's classifier. Algorithm training is done using the features obtained from calculating humoment
        # images.
        gamma_variable_classifier = train_svm_model.train_model_RBF_kernel(num_train=30000, images=train_images, tag=train_labels, gamma_value=0.5,
                                                        num_iteretion=-1, c_value= c, log_transform=True, RAM_size=8000)
        # Predicting the test data.
        prediction, labels = predict_svm_model.predict(clf=gamma_variable_classifier, num_test=3000, images=test_images,tag=test_labels)
        # Print statistic using 'classification_report' function of sklearn library
        statistic_printer.print_SVM_HuMoment_statistic(labels, prediction, gamma_variable_classifier)
        # Save the calculated statistic in csv (exl) file
        statistic_printer.save_SVM_HuMoment_csvFile(labels, prediction, "C - variable",'C', c)
        # Save the maximum accuracy and the corresponding value of c
        max_average_c, max_c = statistic_printer.make_max_average(prediction, labels, 3000, max_average_c, max_c, c)
        print("_______________________________________________________________________________________________________")
        print()

    print()
    print("max_average_c : " + str(max_average_c))
    print("max_c : " + str(max_c))
    print()
    print("***********************************************************************************************************")
    print("                            Training with RBF kernel - 'number of iteration' is variable")
    print("***********************************************************************************************************")
    print()

    while (iteration < 10000):
        print("C value = " + str(c))
        # make svm model's classifier. Algorithm training is done using the features obtained from calculating humoment
        # images.
        gamma_variable_classifier = train_svm_model.train_model_RBF_kernel(num_train=30000, images=train_images,
                                                                           tag=train_labels, gamma_value=0.5,
                                                                           num_iteretion=iteration, c_value=10,
                                                                           log_transform=True, RAM_size=8000)
        # Predicting the test data.
        prediction, labels = predict_svm_model.predict(clf=gamma_variable_classifier, num_test=3000, images=test_images,
                                                       tag=test_labels)
        # Print statistic using 'classification_report' function of sklearn library
        statistic_printer.print_SVM_HuMoment_statistic(labels, prediction, gamma_variable_classifier)
        # Save the calculated statistic in csv (exl) file
        statistic_printer.save_SVM_HuMoment_csvFile(labels, prediction, "Iteration - variable", 'Iteration', iteration)
        # Save the maximum accuracy and the corresponding number of iteration
        max_average_iter, max_iter = statistic_printer.make_max_average(prediction, labels, 3000, max_average_iter, max_iter, iteration)
        iteration = iteration + 500

        print("_______________________________________________________________________________________________________")
        print()


    print("max_average_iter : " + str(max_average_iter))
    print("max_iter : " + str(max_iter))
    print()
    print()
    print("***********************************************************************************************************")
    print("                                Training with RBF kernel - Percision(C,gamma,iteration")
    print("***********************************************************************************************************")
    print()

    max_percision = 0
    gamma = 0.01
    c = 0.1
    GAMMA = []
    C = []
    PERCISION = []
    while (gamma < 3.2):
        while (c < 1.1 ):
            gamma_variable_classifier = train_svm_model.train_model_RBF_kernel(num_train=30000, images=train_images,
                                                                               tag=train_labels, gamma_value=gamma,
                                                                               num_iteretion=10000, c_value=c,
                                                                               log_transform=True, RAM_size=8000)
            prediction, labels = predict_svm_model.predict(clf=gamma_variable_classifier, num_test=3000,
                                                           images=test_images, tag=test_labels)
            GAMMA.append(gamma)
            C.append(c)
            avr = 0
            for i in range(0,3000):
                if(prediction[i]==labels[i]):
                    avr = avr + 1
            avr = (avr/3000)*100
            PERCISION.append(avr)
            if(max_percision < avr):
                max_percision = avr
            c = c + 0.05
        gamma = gamma + 0.3
        c = 0.1

    statistic_printer.make_3D_graph(GAMMA, 'gamma', C, 'c', PERCISION, 'prediction')




if __name__ == "__main__":
        main()








   # fname = 'output.csv'
    # with open(fname, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     for row in sklearn.metrics.confusion_matrix(labels, prediction):
    #         writer.writerow(row)
