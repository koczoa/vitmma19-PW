import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
from collections import Counter
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

# X_data = np.array(X_data)
# y_labels = np.array(y_labels)
#
# print(f"\nShape of image data (X): {X_data.shape}")
# print(f"Shape of labels (y): {y_labels.shape}")
#
#
# label_counts = Counter(y_labels)
#
# most_common_label, most_common_count = label_counts.most_common(1)[0]
#
# baseline_accuracy = most_common_count / len(y_labels)
#
# # print(f"Labels: {y_labels}")
# print(f"Label Counts: {label_counts}")
# print(f"Most common class (baseline prediction): {most_common_label}")
# print(f"Baseline model accuracy: {baseline_accuracy:.4f}")
#


X_data = np.array(X_data)
y_labels = np.array(y_labels)

label_layer = keras.layers.StringLookup(output_mode='int')
label_layer.adapt(y_labels)
y_encoded = label_layer(y_labels)

counts = tf.math.bincount(y_encoded)
majority_index = tf.math.argmax(counts)
num_classes = label_layer.vocabulary_size()

bias_vector = tf.one_hot(majority_index, num_classes)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=X_data.shape[1:]),
    keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='zeros', bias_initializer=bias_vector, trainable=False)
])

model.compile(
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.evaluate(X_data, y_encoded)
model.save("baseline_model.h5")