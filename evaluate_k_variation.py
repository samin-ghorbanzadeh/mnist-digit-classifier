from mnist_knn_sklearn import run_knn_model
import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import mnist 


def testing_k(X_train, y_train, X_test, y_test, k_values):
    list_accuracy = []
    for k in k_values:
        y_pred, report, accuracy = run_knn_model(X_train, y_train, X_test, y_test, k)
        list_accuracy.append(accuracy)
    return list_accuracy


def line_chart(k_values, list_accuracy):
    plt.plot(k_values, list_accuracy, marker='o')
    plt.title("KNN Accuracy vs k")
    plt.xlabel("k")
    plt.xticks(rotation=45)
    plt.ylabel("accuracy") 
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.show()


(X_train, y_train), (X_test, y_test) = mnist.load_data()
k_values = [1, 3, 5, 7]
list_accuracy = testing_k(X_train, y_train, X_test, y_test, k_values)
line_chart(k_values, list_accuracy)

