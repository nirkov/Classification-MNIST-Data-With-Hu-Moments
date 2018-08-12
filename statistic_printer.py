import sklearn.metrics
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix_image(labels,prediction):
    confusion_matrix_image = np.array(sklearn.metrics.confusion_matrix(labels, prediction)).reshape(10, 10)
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    cm = pd.DataFrame(confusion_matrix_image, index=classes, columns=classes)
    cm.index.name = 'Actual class'
    cm.columns.name = 'Predicted class'
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=np.empty_like(cm).astype(str), fmt='', ax=ax)
    plt.savefig('confusion matrix image.jpeg')


def print_SVM_HuMoment_statistic(labels,prediction,classifier):
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8','class 9']
    print()
    print("SVM with HuMoment only on MNIST data -\nClassification report for classifier %s:\n\n%s\n"
          % (classifier, sklearn.metrics.classification_report(y_true=labels, y_pred=prediction,
          target_names=target_names,digits=4)))
    print("Confusion matrix: \neach row of the matrix represents the instances in a predicted class \n"
          "end each column represents the instances in an actual class. \n"
          "\n%s" % sklearn.metrics.confusion_matrix(labels, prediction))
    print()
    print()


def save_SVM_HuMoment_csvFile(labels, prediction,fileName,variableName, variableValue):
    svm_stat = sklearn.metrics.precision_recall_fscore_support(labels, prediction)
    out_dict = {
         "precision": svm_stat[0].round(4)
        ,"recall": svm_stat[1].round(4)
        ,"f1-score": svm_stat[2].round(4)
        ,"support": svm_stat[3]}

    out_df = pd.DataFrame.from_dict(out_dict)
    avg_tot = out_df.apply(lambda x: round(x.mean(), 2) if x.name != "support" else round(x.sum(), 2)).to_frame().T
    avg_tot.index = ["avg/total"]
    out_df = out_df.append(avg_tot)

    with open(fileName+'.csv', 'a') as f:
        out_df.to_csv(f, sep=',')

    variable = pd.DataFrame({variableName:[str(variableValue)]},index=['value'])
    with open(fileName+'.csv', 'a') as f:
        variable.to_csv(f, sep=',')


def save_SVM_HuMoment_txtFile(labels, prediction):
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8','class 9']
    cr = sklearn.metrics.classification_report(labels, prediction, target_names=target_names)
    cm = np.array2string(sklearn.metrics.confusion_matrix(labels, prediction))
    f = open('report.txt', 'w')
    f.write('Title\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(cr, cm))


def make_max_average(prediction, labels,num_of_test,current_average,current_var_vul,candidate_var_vul):
    counter = 0
    for i in range(0, num_of_test):
        if (prediction[i] == labels[i]):
            counter = counter + 1
    max_average = (100 * counter) // float(num_of_test)
    if(current_average < max_average):
        current_average = max_average
        current_var_vul = candidate_var_vul
    return current_average, current_var_vul

def make_3D_graph(x,x_name,y,y_name,prediction,prediction_name):
    x = np.array(x)
    y = np.array(y)
    prediction = np.array(prediction)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, prediction, c='r', marker='o')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(prediction_name)
    plt.show()
