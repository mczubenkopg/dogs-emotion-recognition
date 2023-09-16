"""
date: 05-01-23
created by Michal Kopczynski
"""

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
from tensorflow.python.keras import layers 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import time
import os
import random

#attention - remember to change the paths!

def show_sample_imgs(data_dir):
    image_count  = len(list(data_dir.glob('*/*.jpg')))
    print("Total images of dataset: ", image_count)
    angry = list(data_dir.glob('angry/*'))
    curious = list(data_dir.glob('curious/*'))
    happy = list(data_dir.glob('happy/*'))
    sad = list(data_dir.glob('sad/*'))
    sleepy= list(data_dir.glob('sleepy/*'))
    photos = [str(angry[0]), str(curious[0]), str(happy[0]), str(sad[0]),
     str(sleepy[0])]
    for photo in photos:
        photo =  PIL.Image.open(photo)
        photo.show()
        time.sleep(1)
    
def visualize_data(train_dataset, class_names):
    plt.figure(figsize=(8, 8))
    for images, labels in train_dataset.take(1):
        for i in range(25):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

def loading_data(data_dir, img_height, img_width, batch_size):

    train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    class_names = train_dataset.class_names
    print(class_names)
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)
    return train_dataset, validation_dataset, class_names, test_dataset

def create_model(num_classes, img_w, img_h, train_dataset):
  #first approach
  #data augment
  # resize_and_rescale = tf.keras.Sequential([
  # tf.keras.layers.experimental.preprocessing.Resizing(img_w, img_h),
  # tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
  # ])
  # data_augmentation = tf.keras.Sequential([
  # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
  # ])
  # net = Sequential([
  # resize_and_rescale,
  # data_augmentation,
  # layers.Conv2D(16, 3, padding='same', activation='relu'),
  # layers.MaxPooling2D(),
  # layers.Conv2D(16, 3, padding='same', activation='relu'),
  # layers.MaxPooling2D(),
  # layers.Conv2D(32, 3, padding='same', activation='relu'),
  # layers.MaxPooling2D(),
  # layers.Conv2D(64, 3, padding='same', activation='relu'),
  # layers.MaxPooling2D(),
  # layers.Dropout(0.2),
  # layers.Flatten(),
  # layers.Dense(128, activation='relu'),
  # layers.Dense(num_classes, name="outputs")
  #   ])

  #VGG16
  #data augment
  # resize_and_rescale = tf.keras.Sequential([
  # tf.keras.layers.experimental.preprocessing.Resizing(img_w, img_h),
  # tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
  # ])
  # data_augmentation = tf.keras.Sequential([
  # tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  # tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
  # ])
  # model = Sequential()
  # model.add(resize_and_rescale)
  # model.add(data_augmentation)
  # model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))
  # model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
  # model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
  # model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
  # model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
  # model.add(Dropout(0.2))
  # model.add(Flatten())
  # model.add(Dense(units=4096,activation="relu"))
  # model.add(Dense(units=4096,activation="relu"))
  # model.add(Dense(num_classes, activation="softmax"))

  #Transfer learning - mobilenet
  data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
  ])
  preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
  rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1)
  IMG_SIZE = (img_w, img_h)
  IMG_SHAPE = IMG_SIZE + (3,)
  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
  image_batch, label_batch = next(iter(train_dataset))
  feature_batch = base_model(image_batch)
  #freezing weights
  base_model.trainable = False
  base_model.summary()
  global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
  feature_batch_average = global_average_layer(feature_batch)
  prediction_layer = tf.keras.layers.Dense(num_classes)
  prediction_batch = prediction_layer(feature_batch_average)
  inputs = tf.keras.Input(shape=(160, 160, 3))
  x = data_augmentation(inputs)
  x = preprocess_input(x)
  x = base_model(x, training=False)
  x = global_average_layer(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)

  return model
  

def compile_model(net):
  base_learning_rate = 0.001
  net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])


def fit(net, train_dataset, validation_dataset, epochs):
  history = net.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=epochs)
  return history

def plot_results(epochs, acc, val_acc, loss, val_loss, review_dir):
  epochs_range = range(epochs)
  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')
  isExist = os.path.exists(review_dir)

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  if not isExist:
    os.makedirs(review_dir)
    print("Review data folder has been created!")
  plt.savefig(str(review_dir) + '/results.png')
  plt.show()

def get_results_of_training(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  return acc, val_acc, loss, val_loss

def load_random_img_to_predict(data_dir):
  photos = []
  for file in os.listdir(data_dir):
    d = os.path.join(data_dir, file)
    if os.path.isdir(d):
        for files in os.listdir(d):
            photos.append(d + "/" +files)
  random_img = random.choice(photos)
  return random_img
  


def saving_model(net):
  # Convert the model.
  converter = tf.lite.TFLiteConverter.from_keras_model(net)
  tflite_model = converter.convert()

  # Save the model.
  with open('dogCNN.tflite', 'wb') as f:
    f.write(tflite_model)

def predict(net, img_path, img_height, img_width, class_names, predict_dir):
  img = tf.keras.utils.load_img(
    img_path, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0)

  predictions = net.predict(img_array)
  score = tf.nn.softmax(predictions[0])
  print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
  )
  plt.imshow(img)
  plt.title(f"Predicted: {class_names[np.argmax(score)]} with {100*np.max(score)} percent confidence.")
  isExist = os.path.exists(predict_dir)
  if not isExist:
    os.makedirs(predict_dir)
    print("Review data folder has been created!")
  plt.savefig(str(predict_dir) + '/predicted.png')
  plt.show()
  




def main():
  data_dir = pathlib.Path("/home/mkopcz/Desktop/deeplab/deeplab/model/dataset")
  predict_dir = pathlib.Path("/home/mkopcz/Desktop/deeplab/deeplab/model/predict_data/")
  review_dir = pathlib.Path("/home/mkopcz/Desktop/deeplab/deeplab/model/review_data/")
  show_sample_imgs(data_dir)

  # parameters dataset
  img_sample_path = load_random_img_to_predict(data_dir)
  batch_size = 32
  img_height = 160
  img_width = 160
  epochs = 10

  train_dataset, validation_dataset, class_names, test_dataset = loading_data(data_dir, img_height, img_width, batch_size)
  AUTOTUNE = tf.data.AUTOTUNE
  train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
  validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
  test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
  visualize_data(train_dataset, class_names)
  num_classes = len(class_names)

  net = create_model(num_classes, img_width, img_height, train_dataset)
  compile_model(net)
  history = fit(net, train_dataset, validation_dataset, epochs)

  net.summary()

  acc, val_acc, loss, val_loss = get_results_of_training(history)

  plot_results(epochs, acc, val_acc, loss, val_loss, review_dir)
  predict(net, img_sample_path, img_height, img_width, class_names, predict_dir)
  saving_model(net)

main()






