# Test for when there are a constant 0 training examples and an increasing number of 1 training examples.

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import numpy as np

# Keep same 3 0 training examples, but increase number of 1 training examples
def increasing_1_constant_0():
    # Load data
    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)

    few_shot_examples = range(1, 1000)
    f1_scores_3 = []
    f1_scores_50 = []

    for i in few_shot_examples:
        # 3 0 training examples
        # Test train split
        X_train, y_train, X_test, y_test = constant_0_fewshot_1_train_test(mnist, 3, i)
        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_3.append(metrics.f1_score(y_test, y_pred, pos_label='0'))

        # 50 0 training examples
        # Test train split
        X_train, y_train, X_test, y_test = constant_0_fewshot_1_train_test(mnist, 50, i)
        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)
        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_50.append(metrics.f1_score(y_test, y_pred, pos_label='0'))
        
    plt.title('f1 score for constant 0 training examples and increasing 1 training examples')
    plt.xlabel('number of 1 training examples')
    plt.ylabel('f1 score')
    plt.plot(few_shot_examples, f1_scores_3, label='3 0 training examples')
    plt.plot(few_shot_examples, f1_scores_50, label='50 0 training examples')
    plt.legend()
    plt.show()

# Similar to train test split in util, except this onne you pass the number of 0 and number of 1 training examples
def constant_0_fewshot_1_train_test(mnist, few_shot_0_examples, few_shot_1_examples):
    X = mnist.data
    y = mnist.target

    # Add 0
    is_zero = (mnist.target == '0')
    X_zeros = X[is_zero]
    y_zeros = y[is_zero]
    X_train = X_zeros[:few_shot_0_examples]
    y_train = y_zeros[:few_shot_0_examples]
    zero_split_index = int(0.7 * len(X_zeros))
    X_test = X_zeros[zero_split_index:]
    y_test = y_zeros[zero_split_index:]

    # Add 1
    is_one = (mnist.target == '1')
    X_one = X[is_one]
    y_one = y[is_one]
    X_train = np.vstack((X_train, X_one[:few_shot_1_examples]))
    y_train = np.hstack((y_train, y_one[:few_shot_1_examples]))
    one_split_index = int(0.7 * len(X_one))
    X_test = np.vstack((X_test, X_one[one_split_index:]))
    y_test = np.hstack((y_test, y_one[one_split_index:]))

    return X_train, y_train, X_test, y_test
