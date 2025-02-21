# Takes an unbalanced dataset and adds noise to the minority label until dataset is balanced.

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import numpy as np
import statistics
import seaborn as sns

#### Constants
num_noise_tests = 10 # The number of times to randomly generate noise
label1_count = 10
label2_count = 100
####

# Main
def run_test():
    print (f'label1_count:{label1_count}, label2_count:{label2_count}, num_noise_test:{num_noise_tests}')
    print ('\n\n')

    # Iterate over each digit / label
    labels = ['0','1','2','3','4','5','6','7','8','9']
    without_noise_f1_full = []
    with_noise_f1_full = []
    for label1 in labels:
        new_without_noise_f1 = []
        new_with_noise_f1 = []
        for label2 in labels:
            if label1 == label2:
                new_without_noise_f1.append(0)
                new_with_noise_f1.append(0)
                continue
            without_noise_f1, with_noise_median_f1 = test_noise_vs_without_noise_f1(label1, label1_count, label2, label2_count)
            new_without_noise_f1.append(without_noise_f1)
            new_with_noise_f1.append(with_noise_median_f1)
        without_noise_f1_full.append(new_without_noise_f1)
        with_noise_f1_full.append(new_with_noise_f1)

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2)

    sns.heatmap(without_noise_f1_full, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title("Without Noise F1 Score")
    axes[0].set_xlabel("Label 2")
    axes[0].set_ylabel("Label 1")

    sns.heatmap(with_noise_f1_full, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("With Noise F1 Score")
    axes[1].set_xlabel("Label 2")
    axes[1].set_ylabel("Label 1")

    plt.tight_layout()
    plt.show()

# Returns without noise f1 score and median with noise f1 score
def test_noise_vs_without_noise_f1(label1, label1_count, label2, label2_count):
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
    print (f'label1:{label1}, label1_count:{label1_count}, label2:{label2}, label2_count{label2_count}')
    without_noise_f1 = metrics.f1_score(y_test, y_pred, pos_label=label1)
    print ('without noise:\t', without_noise_f1)


    ### With Noise
    noise_f1 = []
    for num_tests in range(num_noise_tests):
        X_train, y_train, X_test, y_test = test_train_split_2_fewshot_labels_noise_option(mnist, label1, label1_count, label2, label2_count, True)
        # Train
        clf = svm.SVC(kernel='linear', class_weight='balanced')
        clf.fit(X_train, y_train)
        # F1 Score
        y_pred = clf.predict(X_test)
        f1 = metrics.f1_score(y_test, y_pred, pos_label=label1)
        noise_f1.append(f1)
        #print ('with noise:\t', f1)
    # Noise: min max and median
    with_noise_median_f1 = statistics.median(noise_f1)
    print ('noise min:\t\t', min(noise_f1))
    print ('noise max:\t\t', max(noise_f1))
    print ('noise median:\t', statistics.median(noise_f1))
    print ('\n\n')

    return (without_noise_f1, with_noise_median_f1)


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
    run_test()