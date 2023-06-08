# -*- coding: utf-8 -*-

#!pip install emnist
from tensorflow import keras
import tensorflow as tf
print(tf.__version__)
import tensorflow_datasets as tfds
from tqdm import tqdm

from keras import layers, models, datasets
from sklearn.metrics import classification_report, confusion_matrix
# from keras.datasets import mnist, emnist
import numpy as np
import matplotlib.pyplot as plt

# Plot the training and validation accuracy and loss curves
def plot_curves(history, metrics_type = "accuracy"):
  plt.figure(figsize=(8, 3))
  plt.subplot(1, 2, 1)
  if metrics_type == "accuracy":
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'], label='Train')
  plt.plot(history.history['val_loss'], label='Val')
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()

  plt.savefig('curves.jpg')
  plt.show()


"""# Download EMNIST"""

# https://www.tensorflow.org/datasets/catalog/emnist#emnistbalanced
# https://www.kaggle.com/datasets/crawford/emnist
# https://paperswithcode.com/sota/image-classification-on-emnist-balanced
dataset_name = "emnist/balanced"
(train_ds, validation_ds) = tfds.load(
    dataset_name,
    split=["train[:85%]", "train[85%:]"],
    as_supervised=True
)

test_ds = tfds.load(
    dataset_name,
    split="test[:25%]",
    as_supervised=True
)

builder = tfds.builder(dataset_name)
builder.info

LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'A', 'B', 'D', 'E', 'F', 'G', 'H', 'N', 'Q', 'R', 'T'] #Modify here to get only uppercase
#          'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
len(LABELS)

# Visualization
plt.figure(figsize=(5, 5))
for i, (image, label) in enumerate(train_ds.take(9)):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(tf.squeeze(image, axis=2), cmap='gray')
  plt.title(LABELS[int(label)])
  plt.axis("off")

## We are transposing to rotate the image by 90 deg clockwise making the images friendly to visualize.
def transpose(image, label):
  image = tf.transpose(image, [1,0,2])
  return image, label

batch_size = 256

# creating loaders for train, valid, test datasets
trainloader = (
    train_ds
    .shuffle(1024)
    .map(transpose, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

valloader = (
    validation_ds
    .shuffle(1024)
    .map(transpose, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

testloader = (
    test_ds
    .map(transpose, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .batch(1)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

import os, cv2
# saving example images
if not os.path.exists('test_data_example'):
   os.makedirs('test_data_example')

imgs, labels = next(iter(valloader))

plt.figure(figsize=(10, 10))
for n in range(25):
    ax = plt.subplot(5, 5, n+1)
    img = tf.squeeze(imgs[n], axis=2)
    plt.imshow(img, cmap='gray')
    plt.imsave(f"test_data_example/{n}.jpg", img, cmap='gray')
    plt.title(LABELS[int(labels[n])])
    plt.axis('off')

"""# Train model"""

# Define the LeNet+ model
out_len = len(LABELS)
tf.compat.v1.reset_default_graph()
model = keras.Sequential(
    [
        layers.Conv2D(12, kernel_size=7, activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=2),
        layers.BatchNormalization(),
        layers.Conv2D(12*2, kernel_size=5, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(32*15, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(32*10, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(out_len, activation="softmax"),
    ]
)

# Compile the model
# https://stackoverflow.com/questions/58565394/what-is-the-difference-between-sparse-categorical-crossentropy-and-categorical-c
model.compile(
    # categorical_crossentropy
    loss="sparse_categorical_crossentropy", 
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.001), 
    metrics=["accuracy"]
    #metrics=tf.keras.metrics.sparse_categorical_accuracy
)

model.summary()

#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=True, expand_nested=True, show_layer_activations=True)

# Train the model
# https://stackoverflow.com/questions/48285129/saving-best-model-in-keras

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
mcp_save = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')

history = model.fit(trainloader, 
                    epochs=10,
                    #batch_size=1024, 
                    validation_data = valloader,
                    #validation_split=0.1,
                    verbose = 1,
                    callbacks = [earlyStopping, mcp_save, reduce_lr_loss]
                    #callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose = 1)]
          ) 

plot_curves(history, "accuracy")
# https://stackoverflow.com/questions/43979449/higher-validation-accuracy-than-training-accurracy-using-tensorflow-and-keras

# Evaluate the model
test_loss, test_acc = model.evaluate(testloader, verbose=1)
print(f"Test accuracy: {test_acc}")

# https://stackoverflow.com/questions/70477631/batchdataset-get-img-array-and-labels
testloader_unbatched = testloader.unbatch()
test_images = list(testloader_unbatched.map(lambda x, y: x))
test_labels = list(testloader_unbatched.map(lambda x, y: y))
print(len(test_images))
print(len(test_labels))

# Generate predictions for the test set
# https://stackoverflow.com/questions/75686441/typeerror-batchdataset-object-is-not-subscriptable
y_pred = model.predict(testloader)
y_pred = np.argmax(y_pred, axis=1)

# Generate classification report and confusion matrix
report = classification_report(test_labels, y_pred)
print('Classification Report:\n', report)

matrix = confusion_matrix(test_labels, y_pred)
print('Confusion Matrix:\n', matrix)

