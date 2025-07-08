import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk


actual = np.random.randint(0, 2, size=100)
predict = np.random.randint(0, 2, size=100)

print(actual)
print("---------------")
print(predict)

tp = tn = fp = fn = 0
for (x,y) in zip(actual,predict):
    if x == 1 and y == 1:
        tp += 1
    elif x == 0 and y == 0:
        tn += 1
    elif x == 0 and y == 1:
        fp += 1
    elif x == 1 and y == 0:
        fn += 1

print("\n--- Confusion Matrix Values ---")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

#   ✅ High precision = low false positives
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#   ✅ Accuracy = overall correctness
accuracy = tp + tn / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
#   ✅ High recall = low false negatives
recall =  tp / (tp + fn) if (tp + fn) > 0 else 0
#   ✅ High F1 means both precision & recall are high
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# ROC metrics
TPR = recall
FPR = fp / (fp + tn) if (fp + tn) > 0 else 0



print("\n--- Manually Calculated Metrics ---")
print(f"Precision: {precision:.2f}")
print(f"Precision: {accuracy:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1_score:.2f}")