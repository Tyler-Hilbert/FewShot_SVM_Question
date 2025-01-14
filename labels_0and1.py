from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from utils import get_fewshot_train_test, augment_flip #, view_image

# Increases the number of 0 few shot training examples
def increasing_few_shots_0and1labels(compare_to):
    # Load data
    mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)

    few_shot_examples = range(1, 1000)
    f1_scores_no_extra = []

    # Just 0's and 1's
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, [])
        y_train[y_train!='0'] = '1' # Replace all non-zero values with 1

        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_no_extra.append(metrics.f1_score(y_test, y_pred, pos_label='0'))


    compare_to(f1_scores_no_extra, mnist, few_shot_examples)
    
# Trains with augmented 0's and then plots the comparison.
def compare_to_augmented_0and1labels(f1_scores_no_extra, mnist, few_shot_examples):
    f1_scores_extra = []
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, [])
        X_train, y_train = augment_flip(X_train, y_train)
        y_train[y_train!='0'] = '1' # Replace all non-zero values with 1

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
    plt.title('Increasing Number of few-shot 0 Training Examples with each 0 augmented with a flip')
    plt.xlabel('# of 0 training examples')
    plt.ylabel('F1 score')
    plt.show()

# Trains with 8's and then plots the comparison (using just 0's and 1's in test set).
def compare_to_8s_0and1labels(f1_scores_no_extra, mnist, few_shot_examples):
    f1_scores_extra = []
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, ['8'])
        y_train[y_train!='0'] = '1' # Replace all non-zero values with 1

        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_extra.append(metrics.f1_score(y_test, y_pred, pos_label='0'))

    # Plot
    plt.plot(few_shot_examples, f1_scores_no_extra, label='Trained with 0 and 1')
    plt.plot(few_shot_examples, f1_scores_extra, label='Trained with 0, 1 and 8(labeled 1)')
    plt.legend()
    plt.grid()
    plt.title('Increasing Number of few-shot 0 Training Examples with additional 8s labeled as 1')
    plt.xlabel('# of 0 training examples')
    plt.ylabel('F1 score')
    plt.show()

# Trains with all digits and then plots the comparison (using just 0's and 1's in test set).
def compare_to_all_digits_0and1labels(f1_scores_no_extra, mnist, few_shot_examples):
    f1_scores_extra = []
    for i in few_shot_examples:
        # Test train split
        X_train, y_train, X_test, y_test = get_fewshot_train_test(mnist, i, ['2', '3', '4', '5', '6', '7', '8', '9'])
        y_train[y_train!='0'] = '1' # Replace all non-zero values with 1

        # Train
        clf = svm.SVC(kernel='linear')
        clf.fit(X_train, y_train)

        # F1 Score
        y_pred = clf.predict(X_test)
        f1_scores_extra.append(metrics.f1_score(y_test, y_pred, pos_label='0'))

    # Plot
    plt.plot(few_shot_examples, f1_scores_no_extra, label='Trained with 0 and 1')
    plt.plot(few_shot_examples, f1_scores_extra, label='Trained with all digits, non-zero labeled as 1')
    plt.legend()
    plt.grid()
    plt.title('Increasing Number of few-shot 0 Training Examples with all non-zero digits labeled as 1')
    plt.xlabel('# of 0 training examples')
    plt.ylabel('F1 score')
    plt.show()