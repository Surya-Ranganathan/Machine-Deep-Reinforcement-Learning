from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Your dataset
X = [
    [34, 78], [45, 85], [50, 43], [61, 70], [71, 80],
    [60, 52], [75, 89], [55, 42], [80, 90], [52, 65],
    [47, 50], [33, 38], [87, 96], [65, 70], [50, 35]
]
y = [
    0, 0, 0, 1, 1,
    1, 1, 0, 1, 1,
    0, 0, 1, 1, 0
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Use predict_proba to get probability of class 1
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    plt.contourf(xx, yy, probs, 25, cmap="bwr", alpha=0.3)
    plt.contour(xx, yy, probs, levels=[0.5], cmap="Greys", linestyles='dashed')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor='k', s=100)
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1 (Standardized)")
    plt.ylabel("Feature 2 (Standardized)")
    plt.grid(True)
    plt.show()


# Combine train and test data
X_all = np.vstack((X_train, X_test))
y_all = np.concatenate((y_train, y_test))

plot_decision_boundary(X_all, y_all, clf)
