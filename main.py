# Test of few shot with SVM for detecting 0... More details to be written up.

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load data
    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)

    ### Just 0's and 1's
    # Test train split
    X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist)

    # Train
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Confusion matrix
    y_pred = clf.predict(X_test)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1]).plot(cmap='Blues')
    plt.title("Trained and Tested with 0 and 1")
    plt.show()

    ### Added 8's
    # Test train split
    X_train, y_train, X_test, y_test = get_fewshot_train_test_with_additions(mnist)

    # Train
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Confusion matrix
    y_pred = clf.predict(X_test)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1]).plot(cmap='Blues')
    plt.title("Trained with 0, 1 and 8 | tested with 0 and 1")
    plt.show()


# Hardcoded for specific question, only using first 3 `0`'s in training set
def get_fewshot_train_test(mnist):
    X = mnist.data
    y = mnist.target

    is_zero = (mnist.target == '0')
    is_one = (mnist.target == '1')

    # print (len(y[is_zero])) # 6903
    # print (len(y[is_one])) # 7877

    # 0's
    X_zeros = X[is_zero]
    y_zeros = y[is_zero]
    X_train_zeros = X_zeros[:3]
    y_train_zeros = y_zeros[:3]
    zero_split_index = int(0.7 * len(X_zeros))
    X_test_zeros = X_zeros[zero_split_index:]
    y_test_zeros = y_zeros[zero_split_index:]

    # 1's
    X_ones = X[is_one]
    y_ones = y[is_one]
    one_split_index = int(0.7 * len(X_ones))
    X_train_ones, X_test_ones = X_ones[:one_split_index], X_ones[one_split_index:]
    y_train_ones, y_test_ones = y_ones[:one_split_index], y_ones[one_split_index:]

    # Double check
    print (f"len(y_train_zeros) {len(y_train_zeros)}")
    print (f"len(y_test_zeros) {len(X_test_zeros)}")
    print (f"len(y_train_ones) {len(y_train_ones)}")
    print (f"len(y_test_ones) {len(X_test_ones)}")

    # Combine training and testing sets
    X_train = np.vstack((X_train_zeros, X_train_ones))
    y_train = np.hstack((y_train_zeros, y_train_ones))
    X_test = np.vstack((X_test_zeros, X_test_ones))
    y_test = np.hstack((y_test_zeros, y_test_ones))

    return X_train, y_train, X_test, y_test

# Hardcoded for specific question, only using first 3 `0`'s in training set, but with additional points that won't be seen in test set
def get_fewshot_train_test_with_additions(mnist):
    X = mnist.data
    y = mnist.target

    is_zero = (mnist.target == '0')
    is_one = (mnist.target == '1')

    # print (len(y[is_zero])) # 6903
    # print (len(y[is_one])) # 7877

    # 0's
    X_zeros = X[is_zero]
    y_zeros = y[is_zero]
    X_train_zeros = X_zeros[:3]
    y_train_zeros = y_zeros[:3]
    zero_split_index = int(0.7 * len(X_zeros))
    X_test_zeros = X_zeros[zero_split_index:]
    y_test_zeros = y_zeros[zero_split_index:]

    # 1's
    X_ones = X[is_one]
    y_ones = y[is_one]
    one_split_index = int(0.7 * len(X_ones))
    X_train_ones, X_test_ones = X_ones[:one_split_index], X_ones[one_split_index:]
    y_train_ones, y_test_ones = y_ones[:one_split_index], y_ones[one_split_index:]

    # 8's
    is_eight = (mnist.target == '8')
    X_eights = X[is_eight]
    eight_split_index = int(0.7 * len(X_ones))
    X_train_eights = X_eights[eight_split_index:]
    y_train_eights = list('1' * len(X_train_eights))

    # Double check
    print (f"len(y_train_zeros) {len(y_train_zeros)}")
    print (f"len(y_test_zeros) {len(X_test_zeros)}")
    print (f"len(y_train_ones) {len(y_train_ones)}")
    print (f"len(y_test_ones) {len(X_test_ones)}")
    print (f"len(y_train_eights) {len(y_train_eights)}")

    # Combine training and testing sets
    X_train = np.vstack((X_train_zeros, X_train_ones, X_train_eights))
    y_train = np.hstack((y_train_zeros, y_train_ones, y_train_eights))
    X_test = np.vstack((X_test_zeros, X_test_ones))
    y_test = np.hstack((y_test_zeros, y_test_ones))

    return X_train, y_train, X_test, y_test

# View a 784 long ndarray as image (for debugging)
def view_image(img_array):
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()