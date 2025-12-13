import pandas as pd
import numpy as np
from PIL import Image
import keras
from sklearn.model_selection import train_test_split
import wandb
from wandb.integration.keras import WandbCallback
import gc

# 1. Login to Wandb
wandb.login()

# 2. Define the Sweep Configuration
sweep_config = {
    'method': 'bayes',  # 'bayes' for optimization, 'random', or 'grid'
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

# Global constants (immutable across sweeps)
NUM_CLASSES = 3
df = pd.read_csv("labels.csv")
image_files_list = df["files"]
y_labels_raw = df["annotations"]

# Pre-calculate label encoding globally to avoid re-adapting every run
label_layer = keras.layers.StringLookup(output_mode='int')
label_layer.adapt(np.array(y_labels_raw))
y_encoded_global = np.subtract(label_layer(np.array(y_labels_raw)), 1)


def load_and_preprocess_image(filepath, target_size):
    """Helper to load and resize images."""
    img = Image.open(filepath).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return img_array


# 3. Define the Training Function
def train():
    # Initialize wandb for this specific run
    with wandb.init() as run:
        config = wandb.config

        # --- Data Loading (Dynamic based on config.img_size) ---
        print(f"--- Starting run with Img Size: {config.img_size}, Batch Size: {config.batch_size} ---")

        target_size = (config.img_size, config.img_size)
        X_data = []

        # Load and resize images specifically for this hyperparameter setting
        for f in image_files_list:
            processed_img = load_and_preprocess_image(f, target_size)
            X_data.append(processed_img)

        X_data = np.array(X_data)

        # Split Data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_data, y_encoded_global, test_size=0.3, random_state=1
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=1
        )

        # --- Model Building ---
        base_model = keras.applications.MobileNetV2(
            input_shape=(config.img_size, config.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        model = keras.Sequential([
            keras.Input(shape=(config.img_size, config.img_size, 3)),

            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            # MobileNetV2 expects [-1, 1], inputs are [0, 1]. Rescaling handles this.
            keras.layers.Rescaling(scale=2.0, offset=-1.0),

            base_model,

            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(units=NUM_CLASSES, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # --- Training ---
        model.fit(
            X_train, y_train,
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                WandbCallback(save_model=False, log_graph=False)
            ],
            verbose=1
        )

        # Clean up memory for the next run (Important for large image sizes)
        del X_data, X_train, X_val, X_test, model, base_model
        gc.collect()
        keras.backend.clear_session()


# 4. Initialize the Sweep
sweep_id = wandb.sweep(sweep_config, project="mobilenet-optimization")

# 5. Start the Agent
# count=None runs forever, set a number (e.g., count=10) to limit runs
wandb.agent(sweep_id, function=train, count=20)