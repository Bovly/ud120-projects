#
# SVC Kernel and Gamma Parameter Demonstration
#
# This script generates a non-linearly separable dataset and visualizes the
# decision boundaries of a Support Vector Classifier (SVC) using two different
# kernels: 'linear' and 'rbf'.
#
# Instructions:
# 1. To see the difference, run the code as is. The linear kernel will perform poorly,
#    while the rbf kernel will perform well.
# 2. To experiment, change the SVC parameters in the 'Experiment with Parameters' section below.
#
# What to change:
# - kernel: Try 'linear', 'rbf', 'poly', or 'sigmoid'.
# - C: The regularization parameter. A smaller C means more regularization.
# - gamma: The kernel coefficient for 'rbf', 'poly' and 'sigmoid'. A smaller gamma
#          means a wider influence of each training example, resulting in a smoother
#          decision boundary. A larger gamma means a more "local" influence, leading
#          to a more complex, wiggly boundary.
# - degree: The degree of the polynomial kernel function ('poly').
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# ====================
# 1. Data Generation
# ====================
# Create a non-linearly separable dataset to show the power of different kernels.
# The `make_moons` function creates two interlacing half-moon shapes.
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# ============================================
# 2. Experiment with Parameters
# ============================================
# Here are two SVC models to demonstrate the difference.
# The first model uses a simple linear kernel.
model_linear = SVC(kernel='linear', C=1.0)
model_linear.fit(X, y)

# The second model uses a powerful RBF kernel.
# Try changing the `gamma` value here to see its effect!
# gamma = 'scale' is the default and often a good starting point.
# A small gamma (e.g., 0.1) creates a very smooth boundary.
# A large gamma (e.g., 10 or 100) creates a very complex, overfitting boundary.
model_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)
model_rbf.fit(X, y)

# ====================
# 3. Visualization
# ====================

# Helper function to plot the decision boundary
def plot_decision_boundary(ax, model, title):
    # Get min and max values for the axes to create a grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))

    # Predict the class for each point in the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_xticks(())
    ax.set_yticks(())


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the linear model's decision boundary
plot_decision_boundary(ax1, model_linear, 'SVC with linear kernel')

# Plot the RBF model's decision boundary
plot_decision_boundary(ax2, model_rbf, 'SVC with RBF kernel (gamma=scale)')

plt.suptitle('SVC Decision Boundaries with Different Kernels')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("Linear Kernel Accuracy:", model_linear.score(X, y))
print("RBF Kernel Accuracy:", model_rbf.score(X, y))