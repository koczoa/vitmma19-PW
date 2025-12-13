import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split


df = pd.read_csv("labels.csv")

IMG_HEIGHT = 256
IMG_WIDTH = 256
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



X_data = np.array(X_data)
y_labels = np.array(y_labels)

label_layer = keras.layers.StringLookup(output_mode='int')
label_layer.adapt(y_labels)
y_encoded = np.subtract(label_layer(y_labels), 1)

X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_encoded, test_size=0.3, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

counts = tf.math.bincount(y_encoded)
majority_index = tf.math.argmax(counts)
num_classes = label_layer.vocabulary_size()

bias_vector = tf.one_hot(majority_index, num_classes)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

model = keras.Sequential([
    data_augmentation,
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dense(units=NUM_CLASSES, activation='softmax')
])

model.summary()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# model.evaluate(X_data, y_encoded)
# model.save("baseline_model.h5")
EPOCH_NUM = 20
# history = model.fit(X_data, y_encoded, epochs=EPOCH_NUM, batch_size=8, verbose=1)
history = model.fit(X_train, y_train, epochs=EPOCH_NUM, batch_size=8, verbose=1, validation_data=(X_val, y_val))

# final_loss, final_accuracy = model.evaluate(X_data, y_encoded, verbose=1)
final_loss, final_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTraining complete after {EPOCH_NUM} epochs. Final Training Loss: {final_loss:.4f}, Final Training Accuracy: {final_accuracy:.4f}")
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss over Epochs')
# plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


