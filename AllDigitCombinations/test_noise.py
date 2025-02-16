# Will clean up and document later

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import numpy as np

#### Constants
label1 = '0'
label1_count = 10
label2 = '1'
label2_count = 100

# Returns test train split with `label1_count` of `label1` and `label2_count` of `label2`
def test_train_split_2_fewshot_labels(mnist, label1, label1_count, label2, label2_count):
    X_train_label1, y_train_label1, X_test_label1, y_test_label1 = get_split_with_noise(mnist, label1, label1_count)
    X_train_label2, y_train_label2, X_test_label2, y_test_label2 = get_split(mnist, label2, label2_count)

    X_train = np.vstack((X_train_label1, X_train_label2,))
    y_train = np.hstack((y_train_label1, y_train_label2))

    X_test = np.vstack((X_test_label1, X_test_label2))
    y_test = np.hstack((y_test_label1, y_test_label2))

    return X_train, y_train, X_test, y_test


## THIS IS THE MODIFIED FUNCTION WITH NOISE!
def get_split_with_noise(mnist, label, label_count):
    is_label = (mnist.target == label)
    X = mnist.data[is_label]
    y = mnist.target[is_label]
    X_train = X[:label_count]
    y_train = y[:label_count]
    # NOISE ADDED HERE
    noise = np.random.uniform(-0.10, 0.10, X_train.shape)
    X_train_noise = X_train * (1 + noise)
    X_train = np.vstack((X_train, X_train_noise))
    y_train = np.hstack((y_train, y_train))
    test_split_index = int(0.7 * len(X))
    X_test = X[test_split_index:]
    y_test = y[test_split_index:]
    return X_train, y_train, X_test, y_test

# Split with first `label_count` of label in train and last 30% in test
def get_split(mnist, label, label_count):
    is_label = (mnist.target == label)
    X = mnist.data[is_label]
    y = mnist.target[is_label]
    X_train = X[:label_count]
    y_train = y[:label_count]
    test_split_index = int(0.7 * len(X))
    X_test = X[test_split_index:]
    y_test = y[test_split_index:]
    return X_train, y_train, X_test, y_test




#### Main
# Load data
mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)

# Test train split
X_train, y_train, X_test, y_test = test_train_split_2_fewshot_labels(mnist, label1, label1_count, label2, label2_count)
# Train
clf = svm.SVC(kernel='linear', class_weight='balanced')
clf.fit(X_train, y_train)
# F1 Score
y_pred = clf.predict(X_test)
print (metrics.f1_score(y_test, y_pred, pos_label=label1))