import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Simulate binary labels with 90% chance of class 1
# actual = np.random.binomial(1, 0.9, size=100)
# predicted = np.random.binomial(1, 0.9, size=100)

actual = np.random.randint(0, 2, size=100)
predicted = np.random.randint(0, 2, size=100)

print("Actual:")
print(actual)
print("---------------")
print("Predicted:")
print(predicted)

# Compute confusion matrix
confusion_matrix = metrics.confusion_matrix(actual, predicted)

# Unpack TN, FP, FN, TP from confusion matrix
tn, fp, fn, tp = confusion_matrix.ravel()

# Print values
print("\n--- Confusion Matrix Values ---")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# Plot the confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0, 1])
cm_display.plot()
plt.title("Confusion Matrix")
plt.show()
