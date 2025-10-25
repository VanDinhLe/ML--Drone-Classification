import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, 
    confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay,
    precision_recall_curve, auc, roc_curve, precision_recall_curve
)


data_dir = '../img/data_64'  # Root folder containing 'well-mixed' and 'un-mixed'
img_size = (320, 320)
# batch_size = 32
batch_size = 8

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "Final Drone RF/test",
    image_size=img_size,
    batch_size=batch_size,
    shuffle = False
)
class_names = test_dataset.class_names


model = keras.models.load_model("best_model.keras")  # or .keras

loss, acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {acc:.4f}")

# evaluate on test set
loss, acc = model.evaluate(test_dataset)
print(f"Test Accuracy (Keras): {acc:.4f}")

# set up true labels & predictions
y_true = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)
y_pred_prob = model.predict(test_dataset)
y_pred = (y_pred_prob > 0.5).astype(int)
y_score = y_pred_prob.ravel()

# calculate metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_prob)

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

#  plot confusion matrix 
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Test Set)")
plt.show()

# plot ROC Curve 
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc_manual = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc_manual:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) - Test Set")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# plot Pre-Recall curve
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_score)
avg_precision = average_precision_score(y_true, y_score)

plt.figure(figsize=(7, 6))
plt.plot(recall_vals, precision_vals, color="purple", lw=2,
         label=f"PR curve (AP = {avg_precision:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Test Set)")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()
