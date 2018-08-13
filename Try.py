from mlxtend.data import loadlocal_mnist
import numpy as np
import train_svm_model


def make_avrage_data(train_images,train_labels, log=True):
    """
    The function receives the train_images and return The function returns a set of
    |train_images|/10 vectors where each vector is an average of 10 vectors of class it belongs to

    :param train_images : train images from MNIST
    :param train_labels : train labels from MNIST
    :param log          : enable to calculate log transform
    :return: avrage_train_set, avrage_train_tag
    """

    M = len(train_labels)
    train_images_list, train_label_list = train_svm_model.make_HuMoment_data(num_train=M, images=train_images,
                                                                             tag=train_labels, log_transform=False)
    train_images_list = np.array(train_images_list)
    train_label_list = np.array(train_label_list)

    indexes_0 = np.where(train_label_list == 0)
    indexes_1 = np.where(train_label_list == 1)
    indexes_2 = np.where(train_label_list == 2)
    indexes_3 = np.where(train_label_list == 3)
    indexes_4 = np.where(train_label_list == 4)
    indexes_5 = np.where(train_label_list == 5)
    indexes_6 = np.where(train_label_list == 6)
    indexes_7 = np.where(train_label_list == 7)
    indexes_8 = np.where(train_label_list == 8)
    indexes_9 = np.where(train_label_list == 9)

    image_0 = np.array(train_images_list[indexes_0])
    image_1 = np.array(train_images_list[indexes_1])
    image_2 = np.array(train_images_list[indexes_2])
    image_3 = np.array(train_images_list[indexes_3])
    image_4 = np.array(train_images_list[indexes_4])
    image_5 = np.array(train_images_list[indexes_5])
    image_6 = np.array(train_images_list[indexes_6])
    image_7 = np.array(train_images_list[indexes_7])
    image_8 = np.array(train_images_list[indexes_8])
    image_9 = np.array(train_images_list[indexes_9])

    avrage_train_set = []
    avrage_train_tag = []
    for k in range(0, 5001, 10):
        avrage_train_set.append(make_avrage_vector(image_0, k))
        avrage_train_tag.append(0)
        avrage_train_set.append(make_avrage_vector(image_1, k))
        avrage_train_tag.append(1)
        avrage_train_set.append(make_avrage_vector(image_2, k))
        avrage_train_tag.append(2)
        avrage_train_set.append(make_avrage_vector(image_3, k))
        avrage_train_tag.append(3)
        avrage_train_set.append(make_avrage_vector(image_4, k))
        avrage_train_tag.append(4)
        avrage_train_set.append(make_avrage_vector(image_5, k))
        avrage_train_tag.append(5)
        avrage_train_set.append(make_avrage_vector(image_6, k))
        avrage_train_tag.append(6)
        avrage_train_set.append(make_avrage_vector(image_7, k))
        avrage_train_tag.append(7)
        avrage_train_set.append(make_avrage_vector(image_8, k))
        avrage_train_tag.append(8)
        avrage_train_set.append(make_avrage_vector(image_9, k))
        avrage_train_tag.append(9)

    if(log):
        length = len(avrage_train_set)
        avrage_train_set_log = []
        for m in range(0, length):
            avrage_train_set_log.append(train_svm_model.log_transformation(avrage_train_set[m]))
        avrage_train_set = avrage_train_set_log

    return avrage_train_set, avrage_train_tag



def make_avrage_vector(image_j, k):
    vec = image_j[k]
    for i in range(1, 9):
        vec = vec + image_j[k + i]
    vec = vec / 10
    return vec







