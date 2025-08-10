import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist 
import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

def show_confusion_matrix(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def run_knn_model(X_train, y_train, X_test, y_test, k=5):
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_flat, y_train)
    y_pred = knn.predict(X_test_flat)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return y_pred, report, accuracy

def plot_class_distribution(labels):
    counts = pd.Series(labels).value_counts().sort_index()
    counts.plot(kind='bar')
    plt.xlabel("Digits")
    plt.ylabel("Number of Images")
    plt.title("Number of Images per Digit in MNIST Training Set")
    plt.show()

def show_sample_images(num_samples, images, labels):
    plt.figure(figsize=(12, 4))
    for i in range(num_samples):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Digit: {labels[i]}")
        plt.axis('off')
    plt.show()

def show_dataset_shapes(X_train, y_train, X_test, y_test):
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

def compute_average_images(images, labels):
    average_images = []
    for digit in range(10):
        digit_images = images[labels == digit]
        sum_image = np.zeros_like(digit_images[0], dtype=np.float32)
        for img in digit_images:
            sum_image += img.astype(np.float32)
        average_image = sum_image / len(digit_images)
        average_images.append(average_image)
    return average_images

def display_average_images(avg_images):
    plt.figure(figsize=(12, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(avg_images[i], cmap='gray')
        plt.title(f"Digit {i}")
        plt.axis("off")
    plt.suptitle("Average Image of Each Digit", fontsize=16)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
# Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Run basic visualizations and checks
    plot_class_distribution(y_train)
    show_sample_images(int(input("Number of images to show: ")), X_train, y_train)
    show_dataset_shapes(X_train, y_train, X_test, y_test)
    avg_images = compute_average_images(X_train, y_train)
    display_average_images(avg_images)
    
    # Run KNN model
    k_value = int(input("k = "))
    y_pred, report, accuracy = run_knn_model(X_train, y_train, X_test, y_test, k_value)
    
    # Output results
    print("Predicted:", y_pred[:10])
    print("True labels:", y_test[:10])
    print(report)
    print(f"Accuracy on test set: {accuracy:.4f}")
    show_confusion_matrix(y_test, y_pred)
 


























