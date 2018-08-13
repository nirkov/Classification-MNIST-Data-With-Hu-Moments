import numpy as np
from sklearn import svm
import cv2

#I downloaded the warning flags because the system supposedly calculates log (0) but I handle this case
np.seterr(all='ignore')


def train_model_RBF_kernel(num_train=0, images=None, tag=None, gamma_value=1, num_iteretion=-1, c_value=1, log_transform=False, RAM_size=8000, avrage_data = False):
     if(not avrage_data):
      images, tag = make_HuMoment_data(num_train=num_train, images=images ,tag = tag, log_transform = log_transform)
     classifier = svm.SVC(C=c_value, kernel='rbf', gamma=gamma_value, cache_size=RAM_size, probability=False, max_iter= num_iteretion)
     classifier.fit(images, tag)
     return classifier



def make_HuMoment_data(num_train = 0, images = None ,tag = None, log_transform = False):
     train_label_list = []
     train_images_list = []
     for k in range(0, num_train):
          # Each image in the train data passes through HuMoment - features extraction
          hu = cv2.HuMoments(cv2.moments(images[k])).flatten()
          # If necessary, we will perform log10 transform on the vectors  which contain 7 HuMoments
          if (log_transform):
               hu = log_transformation(hu)
          # These lists contain the new training set which each vector is a 7-size vector
          # containing the seven moments of an image
          train_images_list.append(hu)
          train_label_list.append(tag[k])

     return train_images_list, train_label_list


def log_transformation(hu):
     """
     :param hu : vector size-7 with the 7 HuMoment of an image
     :return   : log transformation to the vector : -np.sign(hu) * np.log10(np.abs(hu))
     """
     try:
          hu = -np.sign(hu) * np.log10(np.abs(hu))
     except ZeroDivisionError as e:
          print(" ")
     NAN_indexes = np.isnan(hu)
     hu[NAN_indexes] = 0

     return hu