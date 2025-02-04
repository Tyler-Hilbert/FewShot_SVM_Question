from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

n_range = range(10, 100000, 1)
x_plot = list(n_range)
y_plot = []
for i in n_range:
    print (f'i {i}')
    # Generating Class 0
    class_0_count = i
    x_class_0 = np.random.uniform(low=0, high=4, size=class_0_count)
    y_class_0 = np.zeros(class_0_count)

    # Generating Class 1
    class_1_count = 10
    x_class_1 = np.random.uniform(low=6, high=10, size=class_1_count)
    y_class_1 = np.ones(class_1_count)

    # Concatenating the classes
    x = np.concatenate((x_class_0, x_class_1))
    x = x.reshape(-1,1)
    y = np.concatenate((y_class_0, y_class_1))


    # Train
    clf = svm.SVC(kernel='linear', class_weight='balanced')
    clf.fit(x,y)

    # Print out where the hyperlane division is here
    w = clf.coef_[0][0]  # Since it's 1D, we take the first (and only) coefficient
    b = clf.intercept_[0]
    decision_boundary = -b / w
    print(f"Decision boundary is at x = {decision_boundary}")
    y_plot.append(decision_boundary)
plt.plot(x_plot,y_plot)
plt.show()