# mnist_gradcam.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# normalize and expand dims
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # shape (N,28,28,1)
x_test  = np.expand_dims(x_test, -1)

num_classes = 10
y_train_c = keras.utils.to_categorical(y_train, num_classes)
y_test_c  = keras.utils.to_categorical(y_test, num_classes)

# 2) Build a simple CNN
def build_model():
    inputs = keras.Input(shape=(28,28,1))
    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation="relu", padding="same", name="last_conv")(x) # named conv for Grad-CAM
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model

model = build_model()
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# 3) Train
callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_DIR,"best_model.h5"), save_best_only=True, monitor="val_accuracy"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
]

history = model.fit(
    x_train, y_train_c,
    validation_split=0.1,
    epochs=20,
    batch_size=128,
    callbacks=callbacks,
    verbose=2
)

# 4) Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test_c, verbose=0)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Save training plot
plt.figure(figsize=(6,4))
plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.title("Accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,"training_accuracy.png"))
plt.close()

# 5) Grad-CAM implementation (Keras + TF2)
last_conv_layer_name = "last_conv"
classifier_layer_names = []  # not needed since we use model to get predictions

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # img_array: (1,28,28,1)
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # compute gradients of top predicted class w.r.t conv outputs
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    return heatmap

def save_and_display_gradcam(img, heatmap, filename=None, alpha=0.4):
    # img: original single-channel 28x28 scaled [0,1]
    # heatmap: small spatial map (e.g., 7x7 or similar)
    heatmap = cv2.resize(heatmap, (28,28))
    heatmap = np.uint8(255 * heatmap)
    # apply colormap
    heatmap_col = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_rgb = np.dstack([img*255]*3).astype(np.uint8)
    superimposed = cv2.addWeighted(heatmap_col, alpha, img_rgb, 1-alpha, 0)
    # Save side-by-side figure using matplotlib
    fig, axes = plt.subplots(1,3, figsize=(8,3))
    axes[0].imshow(img.squeeze(), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Heatmap")
    axes[1].axis("off")
    axes[2].imshow(superimposed)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.close()

# 6) Generate Grad-CAM for a few test images
indices = np.random.choice(len(x_test), 12, replace=False)
for i, idx in enumerate(indices):
    img = x_test[idx]
    img_input = np.expand_dims(img, axis=0)
    preds = model.predict(img_input)
    pred_class = np.argmax(preds[0])
    true_class = y_test[idx]
    heatmap = make_gradcam_heatmap(img_input, model, last_conv_layer_name, pred_index=pred_class)
    filename = os.path.join(OUTPUT_DIR, f"gradcam_{i}_true{true_class}_pred{pred_class}.png")
    save_and_display_gradcam(img[:,:,0], heatmap, filename=filename)
    print(f"Saved {filename} (true={true_class}, pred={pred_class})")

print("Done. Outputs saved to:", OUTPUT_DIR)
