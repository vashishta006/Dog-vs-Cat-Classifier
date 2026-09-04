import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path

DATASET_DIR = Path("dataset")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    image_size=(180, 180),
    batch_size=32,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="training"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    image_size=(180, 180),
    batch_size=32,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset="validation"
)

print("Classes:", train_ds.class_names)

# MobileNetV2 preprocessing
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(lambda x, y: (preprocess(tf.cast(x, tf.float32)), y))
val_ds = val_ds.map(lambda x, y: (preprocess(tf.cast(x, tf.float32)), y))

# Augmentation
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(180, 180, 3),
    include_top=False,
    weights="imagenet"
)

# First train classifier with base frozen
base_model.trainable = False

model = models.Sequential([
    augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Fine-tune last MobileNetV2 layers
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

model.save("dog_cat_model.keras")

print("FINAL MODEL SAVED!")