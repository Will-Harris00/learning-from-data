from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,LambdaCallback
from keras.layers import Input,Dropout, Dense,GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
import numpy as np
import itertools
import datetime

import cv2
import os
import io

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

x_train = [] # training images.
y_train = [] # training labels.
x_test = [] # testing images.
y_test = [] # testing labels.

image_size = 256

train_path = 'cleaned/Training'
test_path = 'cleaned/Testing'

for label in labels:
    train_dir = os.path.join(train_path, label)
    for file in tqdm(os.listdir(train_dir)):
        image = cv2.imread(os.path.join(train_dir, file), 0)  # load images in gray.
        image = cv2.bilateralFilter(image, 2, 50, 50)  # remove images noise.
        # image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)  # produce a pseudocolored image.
        # image = cv2.resize(image, (image_size, image_size))  # resize images.
        image = image[:, :, np.newaxis] # adds the value one representing greyscale
        x_train.append(image)
        y_train.append(labels.index(label))

    test_dir = os.path.join(test_path, label)
    for file in tqdm(os.listdir(test_dir)):
        image = cv2.imread(os.path.join(test_dir, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        # image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
        # image = cv2.resize(image, (image_size, image_size))
        image = image[:, :, np.newaxis]
        x_test.append(image)
        y_test.append(labels.index(label))


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape)

x_train, y_train = shuffle(x_train,y_train, random_state=42)


# images = [x_train[i] for i in range(15)]
# fig, axes = plt.subplots(3, 5, figsize = (10, 10))
# axes = axes.flatten()
# for img, ax in zip(images, axes):
#     ax.imshow(img)
# plt.tight_layout()
# plt.show()


# ImageDataGenerator transforms each image in the batch by a series of random translations, rotations, etc.
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True)

# After you have created and configured your ImageDataGenerator, you must fit it on your data.
datagen.fit(x_train)


# plt.figure(figsize=(20, 16))
#
# images_path = ['/glioma/Tr-glTr_0000.jpg', '/meningioma/Tr-meTr_0000.jpg', '/notumor/Tr-noTr_0000.jpg', '/pituitary/Tr-piTr_0000.jpg']
#
# for i in range(4):
#     ax = plt.subplot(2, 2, i + 1)
#     img = cv2.imread(train_path + images_path[i])
#     plt.imshow(img)
#     plt.title(labels[i])
# plt.show()


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(image_size, image_size, 1)), # B&W images
    tf.keras.layers.Rescaling(1. / 255, input_shape=(image_size, image_size, 1)), # normalize Images into range 0 to 1.

    # Convolutional layer 1
    tf.keras.layers.Conv2D(32,(3,3), input_shape=(64, 64, 1), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    # Convolutional layer 2
    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),

    # Neural network

    tf.keras.layers.Dense(units= 252, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=252, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=4, activation='softmax'),


    # tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    # tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(len(labels))

    # tf.keras.layers.Input(shape=(image_size, image_size, 1)),  # B&W images
    # tf.keras.layers.Dense(128, activation='relu'),
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['sparse_categorical_accuracy'])

model.summary()


# training the model
history = model.fit(x_train,
                    y_train,
                    epochs = 10,
                    batch_size = 32,
                    verbose = 1,
                    validation_data = (x_test, y_test))


# plotting losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()
