import numpy as np
import matplotlib.pyplot as plt

# Return X_train, X_test, y_train, y_test for `few_show_samples` number of 0 and a 70-30 test train split for 1 and all other digits in `extra_train_digits` within `mnist` dataset provided in sklearn datasets.
def get_fewshot_train_test(mnist, few_shot_samples, extra_train_digits=[]):
    X = mnist.data
    y = mnist.target

    # Add 0
    is_zero = (mnist.target == '0')
    X_zeros = X[is_zero]
    y_zeros = y[is_zero]
    X_train = X_zeros[:few_shot_samples]
    y_train = y_zeros[:few_shot_samples]
    zero_split_index = int(0.7 * len(X_zeros))
    X_test = X_zeros[zero_split_index:]
    y_test = y_zeros[zero_split_index:]

    # Add non-zero values
    extra_train_digits.append('1')
    for extra_digit in extra_train_digits: # FIXME - combine all at once instead of 1 digit at a time
        extra_digit_mask = (mnist.target == extra_digit)
        X_extra_digit =  X[extra_digit_mask]
        y_extra_digit = y[extra_digit_mask]
        extra_digit_split = int(0.7 * len(X_extra_digit))
        X_train_extra_digit, X_test_extra_digit = X_extra_digit[:extra_digit_split], X_extra_digit[extra_digit_split:]
        y_train_extra_digit, y_test_digit = y_extra_digit[:extra_digit_split], y_extra_digit[extra_digit_split:]
        X_train = np.vstack((X_train, X_train_extra_digit))
        y_train = np.hstack((y_train, y_train_extra_digit))
        if extra_digit == '1': # Add 1 into the test set
            X_test = np.vstack((X_test, X_train_extra_digit))
            y_test = np.hstack((y_test, y_train_extra_digit))

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