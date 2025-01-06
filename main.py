# Test of few shot with SVM for detecting 0... More details to be written up.

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from utils import get_fewshot_train_test, augment_flip #, view_image

# Increasing number of few shot examples using 0, 1 and sometimes 8 (commented out).
# Also currently includes some augmentation on 0.
def increasing_few_shots():
    # Load data
    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)

    few_shot_examples = range(1, 200)
    f1_scores_no_extra = []

    # Just 0's and 1's
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, [])

        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_no_extra.append(metrics.f1_score(y_test, y_pred, pos_label='0'))


    #### THIS IS THE SECTION USED TO CHANGE WHAT I'M TESTING ####
    # Compare to 0's augmented and 1's
    ###### compare_to_augmented(f1_scores_no_extra, mnist, few_shot_examples)

    # Compare to trainset using 8's
    ###### compare_to_8s(f1_scores_no_extra, mnist, few_shot_examples)

    # Compare to using all digits
    compare_to_all_digits(f1_scores_no_extra, mnist, few_shot_examples)
    ####                                                    ####


# Trains with augmented 0's and then plots the comparison.
def compare_to_augmented(f1_scores_no_extra, mnist, few_shot_examples):
    f1_scores_extra = []
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, [])
        X_train, y_train = augment_flip(X_train, y_train)

        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_extra.append(metrics.f1_score(y_test, y_pred, pos_label='0'))

    # Plot
    plt.plot(few_shot_examples, f1_scores_no_extra, label='Trained with 0 and 1')
    plt.plot(few_shot_examples, f1_scores_extra, label='Trained with 0 augmented and 1')
    plt.legend()
    plt.grid()
    plt.title('F1 Score for Increasing Number of 0 Training Examples (MNIST 0 vs 1)')
    plt.xlabel('# of 0 training examples')
    plt.ylabel('F1 score')
    plt.show()

# Trains with 8's and then plots the comparison (using just 0's and 1's in test set).
def compare_to_8s(f1_scores_no_extra, mnist, few_shot_examples):
    f1_scores_extra = []
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, ['8'])

        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_extra.append(metrics.f1_score(y_test, y_pred, pos_label='0'))

    # Plot
    plt.plot(few_shot_examples, f1_scores_no_extra, label='Trained with 0 and 1')
    plt.plot(few_shot_examples, f1_scores_extra, label='Trained with 0, 1 and 8')
    plt.legend()
    plt.grid()
    plt.title('F1 Score for Increasing Number of 0 Training Examples (MNIST 0 vs 1)')
    plt.xlabel('# of 0 training examples')
    plt.ylabel('F1 score')
    plt.show()

# Trains with all digits and then plots the comparison (using just 0's and 1's in test set).
def compare_to_all_digits(f1_scores_no_extra, mnist, few_shot_examples):
    f1_scores_extra = []
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, ['2', '3', '4', '5', '6', '7', '8', '9'])

        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_extra.append(metrics.f1_score(y_test, y_pred, pos_label='0'))

    # Plot
    plt.plot(few_shot_examples, f1_scores_no_extra, label='Trained with 0 and 1')
    plt.plot(few_shot_examples, f1_scores_extra, label='Trained with all digits')
    plt.legend()
    plt.grid()
    plt.title('F1 Score for Increasing Number of 0 Training Examples (MNIST 0 vs 1)')
    plt.xlabel('# of 0 training examples')
    plt.ylabel('F1 score')
    plt.show()



if __name__ == "__main__":
    increasing_few_shots()