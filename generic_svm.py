import importData
import numpy as np

def main():
    train_images, train_labels, test_images, test_labels = importData.load_data()
    classifier1 = importData.train_model(num_train=5000, gamma_arg=2 ,Ci= 20,images=train_images,tag= train_labels,iter=-1)
    prediction1 , test_label = importData.predict(clf=classifier1, num_test=100, images=test_images, tag=test_labels)
    pre1 = []
    pre1.append(prediction1)
    pre1.append(test_label)
    np.savetxt("predict.txt",prediction1 )
    np.savetxt("tag.txt", test_label)

if __name__ == "__main__":
    main()




# _num_test = 100
    # avg_list = []
    # gamma = []
    # for g in range (3,10):
    #   predict , test_label_list = importData.load_dataset(num_train = 50000, num_test=_num_test, gamma_arg = g/100)
    #   avg = 0
    #   for k in range (0 , _num_test):
    #     if (predict[k] == test_label_list[k] ):
    #       avg = avg + 1
    #   avg = (avg / _num_test) * 100
    #   avg_list.append(avg)
    #   gamma.append(g)