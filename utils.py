import numpy as np
import matplotlib.pyplot as plt

# Creates a test train split with 0 as the few shot example
#   Option to include 8's in dataset (but label them as 1's)
#   Option for verbose printing
def get_fewshot_train_test(mnist, few_shot_samples, extra_train_digits=[], verbose=False):
    if extra_train_digits != [] and extra_train_digits != ['8']:
        assert ("Not finished implementing")


    X = mnist.data
    y = mnist.target

    is_zero = (mnist.target == '0')
    is_one = (mnist.target == '1')

    # print (len(y[is_zero])) # 6903
    # print (len(y[is_one])) # 7877

    # 0's
    X_zeros = X[is_zero]
    y_zeros = y[is_zero]
    X_train_zeros = X_zeros[:few_shot_samples]
    y_train_zeros = y_zeros[:few_shot_samples]
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
    if extra_train_digits == ['8']:
        is_eight = (mnist.target == '8')
        X_eights = X[is_eight]
        eight_split_index = int(0.7 * len(X_ones))
        X_train_eights = X_eights[eight_split_index:]
        y_train_eights = list('1' * len(X_train_eights))

    # Double check
    if verbose:
        print (f"len(y_train_zeros) {len(y_train_zeros)}")
        print (f"len(y_test_zeros) {len(X_test_zeros)}")
        print (f"len(y_train_ones) {len(y_train_ones)}")
        print (f"len(y_test_ones) {len(X_test_ones)}")
        if extra_train_digits == ['8']:
            print (f"len(y_train_eights) {len(y_train_eights)}")

    # Combine training and testing sets
    if extra_train_digits == ['8']:
        X_train = np.vstack((X_train_zeros, X_train_ones, X_train_eights))
        y_train = np.hstack((y_train_zeros, y_train_ones, y_train_eights))
    else:
        X_train = np.vstack((X_train_zeros, X_train_ones))
        y_train = np.hstack((y_train_zeros, y_train_ones))
    X_test = np.vstack((X_test_zeros, X_test_ones))
    y_test = np.hstack((y_test_zeros, y_test_ones))

    return X_train, y_train, X_test, y_test

# Adds a flipped version of 0 to dataset
# Note that this may not get consistent results for letters that aren't symmetrical
def augment_flip(X, y):
    is_zero = (y == '0')
    flipped_images = np.array([np.fliplr(img.reshape(28, 28)).flatten() for img in X[is_zero]])
    X_augmented = np.vstack([X, flipped_images])
    y_augmented = np.hstack([y, ['0'] * len(flipped_images)])
    return X_augmented, y_augmented


# View a 784 long ndarray as image (for debugging)
def view_image(img_array):
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()