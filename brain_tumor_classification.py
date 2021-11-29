from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,LambdaCallback
from keras.layers import Input,Dropout, Dense,GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix, \
    accuracy_score
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

image_size = 50

train_path = 'cleaned/Training'
test_path = 'cleaned/Testing'

for label in labels:
    train_dir = os.path.join(train_path, label)
    for file in tqdm(os.listdir(train_dir)):
        image = cv2.imread(os.path.join(train_dir, file), 0)  # load images in gray.
        image = cv2.bilateralFilter(image, 2, 50, 50)  # remove images noise.
        image = cv2.resize(image, (image_size, image_size))  # resize images.
        image = image[:, :, np.newaxis] # adds the value one representing greyscale
        x_train.append(image)
        y_train.append(labels.index(label))

    test_dir = os.path.join(test_path, label)
    for file in tqdm(os.listdir(test_dir)):
        image = cv2.imread(os.path.join(test_dir, file), 0)
        image = cv2.bilateralFilter(image, 2, 50, 50)
        image = cv2.resize(image, (image_size, image_size))
        image = image[:, :, np.newaxis]
        x_test.append(image)
        y_test.append(labels.index(label))

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=123) # testing dir split into test and validation sets
# research EarlyStopping

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_val = np.array(x_val)
y_val = np.array(y_val)

# 7023 images total
print(x_train.shape) # 81% 5712 images
print(x_test.shape) # 9% 655 images
print(x_val.shape) # 9% 656 images


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


# This callback will stop the training when there is no improvement in the loss for three consecutive epochs.
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)


# Convolutional Neural Network Model
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(image_size, image_size, 1)), # B&W images
#     tf.keras.layers.Rescaling(1. / 255, input_shape=(image_size, image_size, 1)), # normalize Images into range 0 to 1.
#
#     # Convolutional layer 1
#     tf.keras.layers.Conv2D(16, (3,3), input_shape=(64, 64, 1), activation='relu'),
#
#     tf.keras.layers.Flatten(),
#
#     # Neural network
#
#     tf.keras.layers.Dense(units= 16, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(units=4, activation='softmax'),
#
#
#     # tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
#     # tf.keras.layers.MaxPooling2D(),
#     # tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#     # tf.keras.layers.MaxPooling2D(),
#     # tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#     # tf.keras.layers.MaxPooling2D(),
#     # tf.keras.layers.Dense(128, activation='relu'),
#     # tf.keras.layers.Dense(len(labels))
#
#     # tf.keras.layers.Input(shape=(image_size, image_size, 1)),  # B&W images
#     # tf.keras.layers.Dense(128, activation='relu'),
# ])

# Feedforward Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(image_size, image_size, 1)),  # B&W images
    tf.keras.layers.Rescaling(1. / 255, input_shape=(image_size, image_size, 1)), # normalize Images into range 0 to 1.
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax'), # total probability to be one but not necessary for hidden layers
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.summary()

print(x_train.shape, y_train.shape)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5000, reshuffle_each_iteration=True).batch(32)
# training the model
history = model.fit(train_ds,
                    epochs = 250,
                    verbose = 1,
                    validation_data = (x_val, y_val),
                    callbacks=[callback])


# plotting losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()


# validate on val set
predicted_prob = model.predict(x_test)
predicted_class = np.argmax(predicted_prob, axis=-1)


def plt_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout(pad=2)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predicted_class)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plt_confusion_matrix(cnf_matrix, classes=labels,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plt_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
