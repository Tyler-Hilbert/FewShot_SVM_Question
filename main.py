# Tests for how training set impacts distinguishing 0 and 1 when there are few 0 training examples.
# SVM
# MNIST

from labels_0and1 import increasing_few_shots_0and1labels
from labels_0and1 import compare_to_augmented_0and1labels, compare_to_8s_0and1labels, compare_to_all_digits_0and1labels

if __name__ == "__main__":
    print ('Using labels 0 and 1, compare to 0 and 1 with augmented 0')
    increasing_few_shots_0and1labels(compare_to_augmented_0and1labels)
    print ('Using labels 0 and 1, compare to 0 and 1 with 8s labeled as 1')
    increasing_few_shots_0and1labels(compare_to_8s_0and1labels)
    print ('Using labels 0 and 1, compare to 0 and 1 with all non-0 digits labeled as 1')
    increasing_few_shots_0and1labels(compare_to_all_digits_0and1labels)