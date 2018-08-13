import predict_svm_model
import statistic_printer
import train_svm_model


def statistic_for_variable_gamma(gamma_max, gamma_steps, num_train,num_test, num_iter, c, log_trans,ram_size, train_images,train_labels,test_images,test_labels ):
    """
        statistic_for_variable_gamma:
            The function performs iterations in which the gamma is the variable.
            In each iteration:
                1. The gamma has a different value (depending on the gamma_steps parameter you selected) and SVM
                     simulation is performed.
                2. The function prints the classificitaion statistic to the screen and saves it to a .csv file.

            The parameters:
               :param gamma_max    :   The maximum value to which gamma will reach the last iteration.
               :param gamma_steps  :   The value at which the gamma grows at each iteration.
               :param num_train    :   Number of train data.
               :param num_test     :   Number of test data.
               :param num_iter     :   The maximum number of iterations the model will perform.
               :param c            :   C is a constant while gamma is the variable.
               :param log_trans    :   To perform or not the log transform to the vectors which each representing the 7
                                       moments of image.
               :param ram_size     :   Assigning the amount of RAM for use by the model.
               :param train_images :   Train images from MNIST data
               :param train_labels :   Train labels from MNIST data
               :param test_images  :   Test images from MNIST data
               :param test_labels  :   Test labels from MNIST data
          """

    print(
        "*************************************************************************************************************")
    print("                                 Training with RBF kernel - gamma is variable")
    print(
        "*************************************************************************************************************")
    print()
    gamma = 0
    max_average_gamma = 0
    max_gamma = 0
    while (gamma < gamma_max):
        gamma = gamma + gamma_steps
        print("gamma value = " + str(gamma))
        # make svm model's classifier. Algorithm training is done using the features obtained from calculating humoment
        # images.
        gamma_variable_classifier = train_svm_model.train_model_RBF_kernel(num_train=num_train, images=train_images,
                                                                           tag=train_labels, gamma_value=gamma,
                                                                           num_iteretion=num_iter, c_value=c,
                                                                           log_transform=log_trans, RAM_size=ram_size)
        # Predicting the test data.
        prediction, labels = predict_svm_model.predict(clf=gamma_variable_classifier, num_test=num_test, images=test_images
                                                       , tag=test_labels, log_transform=log_trans)
        # Print statistic using 'classification_report' function of sklearn library
        statistic_printer.print_SVM_HuMoment_statistic(labels, prediction, gamma_variable_classifier)
        # Save the calculated statistic in csv (exl) file
        statistic_printer.save_SVM_HuMoment_csvFile(labels, prediction, "gamma - variable", 'gamma', gamma)
        # Save the maximum accuracy and the corresponding value of gamma
        max_average_gamma, max_gamma = statistic_printer.make_max_average(prediction, labels, num_test, max_average_gamma,
                                                                          max_gamma, gamma)
        print("_______________________________________________________________________________________________________")
        print()

    print()
    print("max_average_gamma : " + str(max_average_gamma))
    print("max_gamma : " + str(max_gamma))
    print()



def statistic_for_variable_c(c_max, c_steps, num_train,num_test, num_iter, gamma, log_trans,ram_size, train_images,train_labels,test_images,test_labels ):
    """
        statistic_for_variable_c:

               The function performs iterations in which the C is the variable.
               In each iteration:
                   1. The C has a different value (depending on the c_steps parameter you selected) and SVM simulation
                    is performed.
                   2. The function prints the classificitaion statistic to the screen and saves it to a .csv file.

               The parameters:
                :param c_max        :   The maximum value to which c will reach the last iteration.
                :param c_steps      :   The value at which c grows at each iteration.
                :param num_train    :   Number of train data.
                :param num_test     :   Number of test data.
                :param num_iter     :   The maximum number of iterations the model will perform.
                :param gamma        :   gamma is a constant while gamma is the variable.
                :param log_trans    :   To perform or not the log transform to the vectors which each representing the
                                        7 moments of image.
                :param ram_size     :   Assigning the amount of RAM for use by the model.
                :param train_images :   Train images from MNIST data
                :param train_labels :   Train labels from MNIST data
                :param test_images  :   Test images from MNIST data
                :param  test_labels :   Test labels from MNIST data
           """

    print("***********************************************************************************************************")
    print("                                 Training with RBF kernel - C is variable")
    print("***********************************************************************************************************")
    print()
    c = 0
    max_average_c = 0
    max_c = 0
    while (c < c_max):
        c = c + c_steps
        print("C value = " + str(c))
        # make svm model's classifier. Algorithm training is done using the features obtained from calculating humoment
        # images.
        gamma_variable_classifier = train_svm_model.train_model_RBF_kernel(num_train=num_train, images=train_images,
                                                                           tag=train_labels, gamma_value=gamma,
                                                                           num_iteretion=num_iter, c_value=c,
                                                                           log_transform=log_trans, RAM_size=ram_size)
        # Predicting the test data.
        prediction, labels = predict_svm_model.predict(clf=gamma_variable_classifier, num_test=num_test, images=test_images
                                                       , tag=test_labels, log_transform=log_trans)
        # Print statistic using 'classification_report' function of sklearn library
        statistic_printer.print_SVM_HuMoment_statistic(labels, prediction, gamma_variable_classifier)
        # Save the calculated statistic in csv (exl) file
        statistic_printer.save_SVM_HuMoment_csvFile(labels, prediction, "C - variable", 'C', c)
        # Save the maximum accuracy and the corresponding value of c
        max_average_c, max_c = statistic_printer.make_max_average(prediction, labels, num_test, max_average_c, max_c, c)
        print("_______________________________________________________________________________________________________")
        print()

    print()
    print("max_average_c : " + str(max_average_c))
    print("max_c : " + str(max_c))
    print()


def statistic_for_variable_iter(iter_max, iter_steps, num_train,num_test, c, gamma, log_trans,ram_size, train_images,
                                train_labels,test_images,test_labels):
    """
        statistic_for_variable_iter:

             The function performs iterations in which the number of iteration for training the model is the variable.
             In each iteration:
                1. The number of iteration has a different value (depending on the iter_steps parameter you selected)
                   and SVM simulation is performed.
                2. The function prints the classificitaion statistic to the screen and saves it to a .csv file.

            The parameters:
                :param iter_max     :   The maximum number of iteration                      :type Integer
                :param iter_steps   :   The value in which the number of iterations          :type Integer
                                        increases at each iteration.
                :param num_train    :   Number of train data.                                :type Integer
                :param num_test     :   Number of test data.                                 :type Integer
                :param c            :   C is a constant while number of iteration is         :type Integer
                                        the variable.
                :param gamma        :   gamma is a constant while number of iteration        :type Integer
                                        the variable.
                :param log_trans    :   To perform or not the log transform to the vectors   :type bool
                                        which each representing the 7 moments of image.
                :param ram_size     :   Assigning the amount of RAM for use by the model.    :type Integer
                :param train_images :   Train images from MNIST data                         :type List/numpy.array
                :param train_labels :   Train labels from MNIST data                         :type List/numpy.array
                :param test_images  :   Test images from MNIST data                          :type List/numpy.array
                :param test_labels  :   Test labels from MNIST data                          :type List/numpy.array
               """


    print("***********************************************************************************************************")
    print("                            Training with RBF kernel - 'number of iteration' is variable")
    print("***********************************************************************************************************")
    print()
    iteration = 0
    max_average_iter = 0
    max_iter = 0
    while (iteration < iter_max):
        print("C value = " + str(c))
        # make svm model's classifier. Algorithm training is done using the features obtained from calculating humoment
        # images.
        gamma_variable_classifier = train_svm_model.train_model_RBF_kernel(num_train=num_train, images=train_images,
                                                                           tag=train_labels, gamma_value=gamma,
                                                                           num_iteretion=iteration, c_value=c,
                                                                           log_transform=log_trans, RAM_size=ram_size)
        # Predicting the test data.
        prediction, labels = predict_svm_model.predict(clf=gamma_variable_classifier, num_test=num_test, images=test_images,
                                                       tag=test_labels, log_transform=log_trans)
        # Print statistic using 'classification_report' function of sklearn library
        statistic_printer.print_SVM_HuMoment_statistic(labels, prediction, gamma_variable_classifier)
        # Save the calculated statistic in csv (exl) file
        statistic_printer.save_SVM_HuMoment_csvFile(labels, prediction, "Iteration - variable", 'Iteration', iteration)
        # Save the maximum accuracy and the corresponding number of iteration
        max_average_iter, max_iter = statistic_printer.make_max_average(prediction, labels, num_test, max_average_iter,
                                                                        max_iter, iteration)
        iteration = iteration + iter_steps

        print("_______________________________________________________________________________________________________")
        print()

    print("max_average_iter : " + str(max_average_iter))
    print("max_iter : " + str(max_iter))
    print()

def percision_gamma_c_3D(iterr, num_train,num_test, c_max, c_steps, gamma_max, gamma_steps, log_trans,ram_size,
                         train_images,train_labels,test_images,test_labels):
    """
     --percision_gamma_c_3D
            The function calculates the percentage of accuracy for different gamma and c values and actually
            simulates a function of two variables - percision (gamma, c)

             The parameters:
                :param iterr         :   Number of iteration for training of the model            :type Integer
                :param num_train    :   Number of train data.                                    :type Integer
                :param num_test     :   Number of test data.                                     :type Integer
                :param c_max        :   The maximum value to which c will reach the last         :type Integer
                                        iteration.
                :param c_steps      :   The value at which c grows at each iteration.            :type Integer
                :param gamma_max    :   The maximum value to which gamma will reach the last     :type Integer
                                        iteration.
                :param gamma_steps  :   The value at which the gamma grows at each iteration.    :type Integer
                :param log_trans    :   To perform or not the log transform to the vectors       :type bool
                                        which each representing
                :param ram_size     :   Assigning the amount of RAM for use by the model.        :type Integer
                :param train_images :   Train images from MNIST data                             :type List/numpy.array
                :param train_labels :   Train labels from MNIST data                             :type List/numpy.array
                :param test_images  :   Test images from MNIST data                              :type List/numpy.array
                :param test_labels  :   Test labels from MNIST data                              :type List/numpy.array
    """

    print()
    print("***********************************************************************************************************")
    print("                                Training with RBF kernel - Percision(C,gamma)")
    print("***********************************************************************************************************")
    print()

    max_percision = 0
    gamma = gamma_steps/10
    c = c_steps/10
    gamma_list = []
    c_list = []
    percisio_list = []
    while (gamma < gamma_max):
        while (c < c_max):
            classifierr = train_svm_model.train_model_RBF_kernel(num_train=num_train, images=train_images,
                                                                               tag=train_labels, gamma_value=gamma,
                                                                               num_iteretion=iterr, c_value=c,
                                                                               log_transform=log_trans, RAM_size=ram_size
                                                                               , avrage_data=False)
            prediction, labels = predict_svm_model.predict(clf=classifierr, num_test=num_test,
                                                           images=test_images, tag=test_labels, log_transform=log_trans)
            gamma_list.append(gamma)
            c_list.append(c)
            avr = 0
            for i in range(0, num_test):
                if (prediction[i] == labels[i]):
                    avr = avr + 1
            avr = (avr / num_test) * 100
            percisio_list.append(avr)
            if (max_percision < avr):
                max_percision = avr
            c = c + c_steps
        gamma = gamma + gamma_steps
        print(gamma)
        c = 1

    statistic_printer.make_3D_graph(gamma_list, 'gamma', c_list, 'c', percisio_list, 'prediction')

def confusion_matrix_of_gamma_c(gamma_list, c_list,log_trans, iterr,num_train,train_images,test_images,train_labels,
                                test_labels,num_test,ram_size):
    """
    The function performs a simulation according to the parameters and produces a confusion matrix
    in the directory "confusion matrix"

    :param gamma_list   :   list of gamma value.
    :param c_list       :   list of c value.
    :param log_trans    :   To perform or not the log transform to the vectors.
    :param iterr        :   Number of iteration.
    :param num_train    :   Number of train data.
    :param train_images :   Train images from MNIST data
    :param train_labels :   Train labels from MNIST data
    :param test_images  :   Test images from MNIST data
    :param test_labels  :   Test labels from MNIST data
    :param num_test     :   Number of test data.
    :param ram_size     :   Assigning the amount of RAM for use by the model.
    """

    num_of_couple = len(gamma_list)
    for i in range(0,num_of_couple):
        classifierr = train_svm_model.train_model_RBF_kernel(num_train=num_train, images=train_images,
                                                                           tag=train_labels, gamma_value=gamma_list[i],
                                                                           num_iteretion=iterr, c_value=c_list[i],
                                                                           log_transform=log_trans, RAM_size=ram_size
                                                                           , avrage_data=False)
        prediction, labels = predict_svm_model.predict(clf=classifierr, num_test=num_test,
                                                       images=test_images, tag=test_labels, log_transform=log_trans)
        avr = 0
        for k in range(0, num_test):
            if (prediction[k] == labels[k]):
                avr = avr + 1
        avr = (avr / num_test) * 100
        statistic_printer.confusion_matrix_image(labels, prediction,"gamma_"+str(gamma_list[i])+"_c_"+str(c_list[i])+"_avrage_"+str(avr))



















