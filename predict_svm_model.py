import numpy as np
import cv2


def predict(clf, num_test, images, tag):
    test_images_list = []
    test_label_list = []

    for i in range(0, num_test):
        temp2 = cv2.HuMoments(cv2.moments(images[i])).flatten()
        temp22 = -np.sign(temp2) * np.log10(np.abs(temp2))
        index = np.isnan(temp22)
        temp22[index] = 0
        test_images_list.append(temp22)
        test_label_list.append(tag[i])

    return clf.predict(test_images_list), test_label_list
