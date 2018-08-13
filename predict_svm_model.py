import cv2
import train_svm_model


def predict(clf, num_test, images, tag, log_transform=True):
    test_images_list = []
    test_label_list = []

    for i in range(0, num_test):
        hu = cv2.HuMoments(cv2.moments(images[i])).flatten()
        if(log_transform):
            hu = train_svm_model.log_transformation(hu)
        test_images_list.append(hu)
        test_label_list.append(tag[i])

    return clf.predict(test_images_list), test_label_list
