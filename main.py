import Try
import importData
import model_statistics
import numpy as np
import matplotlib.pyplot as plt

import statistic_printer


def main():
    """"
    todo: Change the path that matches the file location on your computer to import MNIST data.
    """
    train_images, train_labels, test_images, test_labels = importData.load_data(data_address = 'D:/Python Projects/MNIST_With_Moments/mnist_data')
    avrage_train_set, avrage_train_tag = Try.make_avrage_data(train_images, train_labels, log=True)

    print(
        "------------------------------ Training SVM model on MNIST data with HuMomnt only -------------------------------------")
    print()

    model_statistics.statistic_for_variable_gamma(gamma_max     =0.1,
                                                  gamma_steps   =0.02,
                                                  num_train     =100,
                                                  num_test      =3000,
                                                  num_iter      =-1,
                                                  c             =5,
                                                  log_trans     =True,
                                                  ram_size      =8000,
                                                  train_images  =train_images,
                                                  train_labels  =train_labels,
                                                  test_images   =test_images,
                                                  test_labels   =test_labels)


    model_statistics.statistic_for_variable_c( c_max          =50,
                                               c_steps       =10,
                                               num_train     =1000,
                                               num_test      =3000,
                                               num_iter      =-1,
                                               gamma         =1,
                                               log_trans     =True,
                                               ram_size      =8000,
                                               train_images  =train_images,
                                               train_labels  =train_labels,
                                               test_images   =test_images,
                                               test_labels   =test_labels)




    model_statistics.statistic_for_variable_iter(iter_max      =401,
                                                 iter_steps    =100,
                                                 num_train     =1000,
                                                 num_test      =3000,
                                                 c             =5,
                                                 gamma         =1,
                                                 log_trans     =True,
                                                 ram_size      =8000,
                                                 train_images  =train_images,
                                                 train_labels  =train_labels,
                                                 test_images   =test_images,
                                                 test_labels   =test_labels)

    model_statistics.percision_gamma_c_3D(iterr         =10000,
                                          num_train     =15000,
                                          num_test      =3000,
                                          c_max         =90,
                                          c_steps       =4,
                                          gamma_max     =5,
                                          gamma_steps   =0.4,
                                          log_trans     =False,
                                          ram_size      =8000,
                                          train_images  =train_images,
                                          train_labels  =train_labels,
                                          test_images   =test_images,
                                          test_labels   =test_labels)

    model_statistics.confusion_matrix_of_gamma_c(gamma_list=[0.1, 0.02],
                                                 c_list=[70, 1001],
                                                 log_trans=True,
                                                 iterr=-1,
                                                 num_train=60000,
                                                 train_images=train_images,
                                                 test_images=test_images,
                                                 train_labels=train_labels,
                                                 test_labels=test_labels,
                                                 num_test=5000,
                                                 ram_size=8000)

if __name__ == "__main__":
        main()







