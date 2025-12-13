import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import keras


df = pd.read_csv("labels.csv")

IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CLASSES = 3

print(f"all datapoints: {len(df)}")

image_files_list = df["files"]

def load_and_preprocess_image(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    img = Image.open(filepath).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array

X_data = []
y_labels = df["annotations"]


for i, f in enumerate(image_files_list):
    processed_img = load_and_preprocess_image(f)
    X_data.append(processed_img)



X_data = np.array(X_data[:8])
y_labels = np.array(y_labels[:8])

label_layer = keras.layers.StringLookup(output_mode='int')
label_layer.adapt(y_labels)
y_encoded = label_layer(y_labels)

counts = tf.math.bincount(y_encoded)
majority_index = tf.math.argmax(counts)
num_classes = label_layer.vocabulary_size()

bias_vector = tf.one_hot(majority_index, num_classes)

model = keras.Sequential([
    keras.layers.Conv2D(filters=4, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(units=NUM_CLASSES, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# model.evaluate(X_data, y_encoded)
# model.save("baseline_model.h5")

history = model.fit(X_data, y_encoded, epochs=200, batch_size=len(X_data), verbose=1)

final_loss, final_accuracy = model.evaluate(X_data, y_encoded, verbose=1)
print(f"\nTraining complete after 200 epochs. Final Training Loss: {final_loss:.4f}, Final Training Accuracy: {final_accuracy:.4f}")
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()