# Test all digit combinations for:
# Holding number of training examples of one digit constant and increasing the other.

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import numpy as np

### Test Variables
max_n = 1000 # The value of n to increase the majority class to (minority class is hardcoded to 3 and 50)
step_n = 10 # The number of n to increase by in each iteration (a higher value runs quicker, but has less test cases)
use_f1_score = True # The default evaluation is F1, but setting this to False will result in using precision instead
labels = ['0','1','2','3','4','5','6','7','8','9']
### End Test Variables

def check(label1, label2, max_n, step_n, use_f1_score):
    # Load data
    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)

    few_shot_examples = range(1, max_n, step_n)
    scores_3 = []
    scores_50 = []

    for i in few_shot_examples:
        # 3 training examples
        # Test train split
        X_train, y_train, X_test, y_test = test_train_split_2_fewshot_labels(mnist, label1, 3, label2, i)
        # Train
        clf = svm.SVC(kernel='linear', class_weight='balanced')
        clf.fit(X_train, y_train)
        # Score
        y_pred = clf.predict(X_test)
        if use_f1_score:
            scores_3.append(metrics.f1_score(y_test, y_pred, pos_label=label1))
        else:
            scores_3.append(metrics.precision_score(y_test, y_pred, pos_label=label1))

        # 50 training examples
        # Test train split
        X_train, y_train, X_test, y_test = test_train_split_2_fewshot_labels(mnist, label1, 50, label2, i)
        # Train
        clf = svm.SVC(kernel='linear', class_weight='balanced')
        clf.fit(X_train, y_train)
        # Score
        y_pred = clf.predict(X_test)
        if use_f1_score:
            scores_50.append(metrics.f1_score(y_test, y_pred, pos_label=label1))
        else:
            scores_50.append(metrics.precision_score(y_test, y_pred, pos_label=label1))

    title = f'constant {label1} examples and increasing {label2} examples class_weight=balanced'
    plt.title(title)
    plt.xlabel(f'number of {label2} training examples')
    if use_f1_score:
        plt.ylabel('f1 score')
    else:
        plt.ylabel('precision')
    plt.plot(few_shot_examples, scores_3, label=f'3 {label1} training examples')
    plt.plot(few_shot_examples, scores_50, label=f'50 {label1} training examples')
    plt.legend()
    plt.grid()
    plt.savefig(f'plots_regression_test/{label1}-{label2}.png')
    #plt.show()
    plt.close()

# Returns test train split with `label1_count` of `label1` and `label2_count` of `label2`
def test_train_split_2_fewshot_labels(mnist, label1, label1_count, label2, label2_count):
    X_train_label1, y_train_label1, X_test_label1, y_test_label1 = get_split(mnist, label1, label1_count)
    X_train_label2, y_train_label2, X_test_label2, y_test_label2 = get_split(mnist, label2, label2_count)

    X_train = np.vstack((X_train_label1, X_train_label2,))
    y_train = np.hstack((y_train_label1, y_train_label2))

    X_test = np.vstack((X_test_label1, X_test_label2))
    y_test = np.hstack((y_test_label1, y_test_label2))

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


for label1 in labels:
    for label2 in labels:
        if label1 == label2:
            continue
        check(label1, label2, max_n, step_n, use_f1_score)