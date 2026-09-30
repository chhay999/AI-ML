# Animal image classification: Cat, Buffalo, Elephant, Rhino, Zebra
# Folder layout expected:
#   Dataset/train/<class>/*.jpg
#   Dataset/validation/<class>/*.jpg
#   Dataset/test/<class>/*.jpg
# Each "# %%" marks a notebook cell (works in Jupyter and VS Code).

# %% 1. Imports and settings
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

USE_TRANSFER_LEARNING = True      # True = MobileNetV2 (recommended), False = custom CNN
DATA_DIR = 'Dataset'              # change if your folder is elsewhere
CLASS_NAMES = ['Cat', 'Buffalo', 'Elephant', 'Rhino', 'Zebra']
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 30

print("Working dir:", os.getcwd())
print("Dataset folders:", os.listdir(DATA_DIR))

# %% 2. Data generators
if USE_TRANSFER_LEARNING:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    common = dict(preprocessing_function=preprocess_input)   # scales to [-1, 1]
else:
    common = dict(rescale=1. / 255)                          # scales to [0, 1]

train_gen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.8, 1.2),
    horizontal_flip=True,
    **common,
)
val_gen = ImageDataGenerator(**common)
test_gen = ImageDataGenerator(**common)

options = dict(target_size=IMG_SIZE, classes=CLASS_NAMES,
               batch_size=BATCH_SIZE, class_mode='categorical')

train_dataset = train_gen.flow_from_directory(f'{DATA_DIR}/train/', shuffle=True, **options)
validation_dataset = val_gen.flow_from_directory(f'{DATA_DIR}/validation/', shuffle=False, **options)
test_dataset = test_gen.flow_from_directory(f'{DATA_DIR}/test/', shuffle=False, **options)

print(train_dataset.class_indices)

# %% 3. Sanity check (min/max should be about -1..1 for MobileNet, 0..1 for custom CNN)
x, y = next(train_dataset)
print("Batch:", x.shape, "min:", x.min(), "max:", x.max())
print("Labels sample:", y[:3])

# %% 4. Build the model
def build_transfer_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
    base.trainable = False
    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
    return models.Model(inputs, outputs), base


def build_custom_cnn():
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE + (3,)),

        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(CLASS_NAMES), activation='softmax'),
    ])
    return model, None


model, base = build_transfer_model() if USE_TRANSFER_LEARNING else build_custom_cnn()

model.compile(
    optimizer=optimizers.Adam(1e-3 if USE_TRANSFER_LEARNING else 1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

# %% 5. Train
cbs = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
]

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=cbs,
)

# %% 6. Optional fine-tuning (transfer learning only): improves accuracy a bit more
if USE_TRANSFER_LEARNING:
    base.trainable = True
    for layer in base.layers[:-30]:          # keep early layers frozen
        layer.trainable = False
    for layer in base.layers:                # keep BatchNorm frozen for stability
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    history_ft = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=10,
        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                           restore_best_weights=True)],
    )
    for k in history.history:                # merge histories for plotting
        history.history[k] += history_ft.history[k]

# %% 7. Training curves
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='validation')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()
plt.tight_layout()
plt.show()

# %% 8. Evaluate on the test set
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")

test_dataset.reset()
y_pred = np.argmax(model.predict(test_dataset), axis=1)
y_true = test_dataset.classes               # valid because shuffle=False

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion matrix')
plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.colorbar()
plt.tight_layout()
plt.show()

# %% 9. Save the model
model.save('animal_classifier.keras')

# %% 10. Predict a single image
from tensorflow.keras.preprocessing import image

def predict_image(path):
    img = image.load_img(path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    if USE_TRANSFER_LEARNING:
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr)[0]
    idx = int(np.argmax(probs))
    print(f"{CLASS_NAMES[idx]} ({probs[idx] * 100:.1f}%)")
    plt.imshow(img); plt.axis('off'); plt.show()

# predict_image('Dataset/test/Zebra/example.jpg')