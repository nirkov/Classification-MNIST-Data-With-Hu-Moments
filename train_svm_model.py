import numpy as np
from sklearn import svm
import cv2

#I downloaded the warning flags because the system supposedly calculates log (0) but I handle this case
np.seterr(all='ignore')


def train_model_RBF_kernel(num_train = 0, images = None ,tag = None, gamma_value = 1, num_iteretion = -1, c_value = 1, log_transform = False, RAM_size = 8000):

     train_label_list = []
     train_images_list = []

     for k in range(0, num_train):
          Hu = cv2.HuMoments(cv2.moments(images[k])).flatten()
          if(log_transform):
               try:
                    Hu = -np.sign(Hu)*np.log10(np.abs(Hu))
               except ZeroDivisionError as e:
                    print(" ")
               NAN_indexes = np.isnan(Hu)
               Hu[NAN_indexes] = 0
          train_images_list.append(Hu)
          train_label_list.append(tag[k])

     classifier = svm.SVC(C=c_value, kernel='rbf', gamma=gamma_value, cache_size=RAM_size, probability=False, max_iter= num_iteretion, )
     classifier.fit(train_images_list, train_label_list)
     return classifier

