import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
import wandb

wandb.login()


sweep_config = {
    'method': 'bayes', # 'bayes' for optimization, 'random', or 'grid'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'img_size': {
            'values': [32, 64, 128, 256, 512, 1024]
        },
        'batch_size': {
            'values': [4, 8, 16, 32, 64, 128, 256]
        },
        'epochs': {
            'value': 20
        }
    }
}

df = pd.read_csv("labels.csv")


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



base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = keras.Sequential([
    keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),

    keras.layers.Rescaling(scale=2.0, offset=-1.0),

    base_model,

    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=NUM_CLASSES, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


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
