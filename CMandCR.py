# confusion_matrix_only.py

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Load MNIST test set
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)

# 2) Load your trained model
OUTPUT_DIR = "outputs"
model = keras.models.load_model(os.path.join(OUTPUT_DIR, "best_model.h5"))

# 3) Predict
y_pred = np.argmax(model.predict(x_test), axis=1)

# 4) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()

# 5) Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

print("\n✅ Confusion matrix and report saved to 'outputs/confusion_matrix.png'")
