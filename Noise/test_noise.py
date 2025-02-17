# Takes an unbalanced dataset and adds noise to the minority label until dataset is balanced.

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import numpy as np
import statistics

#### Constants
label1 = '0'
label1_count = 10
label2 = '1'
label2_count = 100
####

# Main
def test_noise():
    # Load data
    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)


    ### Without Noise
    # Test train split
    X_train, y_train, X_test, y_test = test_train_split_2_fewshot_labels_noise_option(mnist, label1, label1_count, label2, label2_count, False)
    # Train
    clf = svm.SVC(kernel='linear', class_weight='balanced')
    clf.fit(X_train, y_train)
    # F1 Score
    y_pred = clf.predict(X_test)
    print ('without noise:\t', metrics.f1_score(y_test, y_pred, pos_label=label1))


    ### With Noise
    noise_f1 = []
    for num_tests in range(10):
        X_train, y_train, X_test, y_test = test_train_split_2_fewshot_labels_noise_option(mnist, label1, label1_count, label2, label2_count, True)
        # Train
        clf = svm.SVC(kernel='linear', class_weight='balanced')
        clf.fit(X_train, y_train)
        # F1 Score
        y_pred = clf.predict(X_test)
        f1 = metrics.f1_score(y_test, y_pred, pos_label=label1)
        noise_f1.append(f1)
        print ('with noise:\t', f1)
    # Noise: min max and median
    print ('noise min', min(noise_f1))
    print ('noise max', max(noise_f1))
    print ('noise median', statistics.median(noise_f1))


# Returns test train split with `label1_count` of `label1` and `label2_count` of `label2`
def test_train_split_2_fewshot_labels_noise_option(mnist, label1, label1_count, label2, label2_count, noise):
    if noise:
        max_duplication = label2_count // label1_count # How many times to add noise to balance the dataset
        X_train_label1, y_train_label1, X_test_label1, y_test_label1 = get_split_with_noise(mnist, label1, label1_count, max_duplication)
    else:
        X_train_label1, y_train_label1, X_test_label1, y_test_label1 = get_split(mnist, label1, label1_count)
    X_train_label2, y_train_label2, X_test_label2, y_test_label2 = get_split(mnist, label2, label2_count)

    X_train = np.vstack((X_train_label1, X_train_label2,))
    y_train = np.hstack((y_train_label1, y_train_label2))

    ## Double check dataset
    ##print (f'X_train_label1.shape', X_train_label1.shape)
    ##print (f'X_train_label2.shape', X_train_label2.shape)

    X_test = np.vstack((X_test_label1, X_test_label2))
    y_test = np.hstack((y_test_label1, y_test_label2))

    return X_train, y_train, X_test, y_test


## THIS IS THE MODIFIED FUNCTION WITH NOISE!
# Note that the original data counts as 1 duplication, so really max_duplication = duplications-1
def get_split_with_noise(mnist, label, label_count, max_duplication):
    # Get label
    is_label = (mnist.target == label)
    X = mnist.data[is_label]
    y = mnist.target[is_label]

    # Train
    X_train = X[:label_count]
    y_train = y[:label_count]
    X_train_original = X_train
    y_train_original = y_train
    # NOISE ADDED HERE
    for _ in range(max_duplication-1):
        noise = np.random.uniform(-0.10, 0.10, X_train_original.shape)
        X_train_noise = X_train_original * (1 + noise)
        X_train = np.vstack((X_train, X_train_noise))
        y_train = np.hstack((y_train, y_train_original))

    # Test
    test_split_index = int(0.7 * len(X))
    X_test = X[test_split_index:]
    y_test = y[test_split_index:]

    # Return
    return X_train, y_train, X_test, y_test


# Split with first `label_count` of label in train and last 30% in test
def get_split(mnist, label, label_count):
    # Get label
    is_label = (mnist.target == label)
    X = mnist.data[is_label]
    y = mnist.target[is_label]

    # Train
    X_train = X[:label_count]
    y_train = y[:label_count]
    test_split_index = int(0.7 * len(X))

    # Test
    X_test = X[test_split_index:]
    y_test = y[test_split_index:]

    # Return
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    test_noise()