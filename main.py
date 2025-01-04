# Test of few shot with SVM for detecting 0... More details to be written up.

from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from utils import get_fewshot_train_test, augment_flip #, view_image

# Test when increasing number of few shot examples
def increasing_few_shots():
    # Load data
    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)

    few_shot_examples = range(1, 200)
    f1_scores_no_extra = []
    f1_scores_extra = []

    ### Just 0's and 1's
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, False)

        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_no_extra.append(metrics.f1_score(y_test, y_pred, pos_label='0'))

    ### 0's augmented and 1's
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, False)
        X_train, y_train = augment_flip(X_train, y_train)

        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_extra.append(metrics.f1_score(y_test, y_pred, pos_label='0'))

    '''
    ### Added 8's
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, True)

        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_extra.append(metrics.f1_score(y_test, y_pred, pos_label='0'))
    '''

    # Plot
    plt.plot(few_shot_examples, f1_scores_no_extra, label='Trained with 0 and 1')
    #plt.plot(few_shot_examples, f1_scores_extra, label='Trained with 0, 1 and 8(junk)')
    plt.plot(few_shot_examples, f1_scores_extra, label='Trained with 0 augmented and 1')
    plt.legend()
    plt.grid()
    plt.title('F1 Score for Increasing Number of 0 Training Examples (MNIST)')
    plt.xlabel('# of 0 training examples')
    plt.ylabel('F1 score')
    plt.show()


# Test when hardcoding to 3 few shot examples
def hardcode_3_few_shot():
    # Load data
    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)

    ### Just 0's and 1's
    # Test train split
    X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, 3, False)

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
    X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, 3, True)

    # Train
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Confusion matrix
    y_pred = clf.predict(X_test)
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=[0, 1]).plot(cmap='Blues')
    plt.title("Trained with 0, 1 and 8 | tested with 0 and 1")
    plt.show()


if __name__ == "__main__":
    #hardcode_3_few_shot()
    increasing_few_shots()