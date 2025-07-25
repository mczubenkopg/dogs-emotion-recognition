"""
Title: Grad-CAM class activation visualization
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/26
Last modified: 2021/03/07
Description: How to obtain a class activation heatmap for an image classification model.
Accelerator: GPU
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras

import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

os.environ["KERAS_BACKEND"] = "tensorflow"
last_conv_layer_name = "block14_sepconv2_act"
BASE_DIR = Path('../dataset')

def create_and_train_model():
    train_dataset, val_dataset = keras.utils.image_dataset_from_directory(
        directory=BASE_DIR,
        labels='inferred',
        label_mode='categorical',
        class_names=['angry', 'curious', 'happy', 'sad', 'sleepy'],
        image_size=(256, 256),
        seed=42,
        subset='both',
        validation_split=0.2,
        batch_size=32,
        crop_to_aspect_ratio=True)
    base_model = keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(256, 256, 3), classes=5,
                                             pooling='avg')
    x = base_model.output
    outputs = keras.layers.Dense(5, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=outputs)
    for layer in model.layers[:40]:
        layer.trainable = False
    optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
    history = model.fit(train_dataset, steps_per_epoch=80, epochs=300, validation_data=val_dataset, validation_steps=20)
    # save model architecture
    model_json = model.to_json()
    with open('xception_model.json', 'w') as file:
        file.write(model_json)
    model.save_weights('image_classifier_xception.h5', overwrite=True)
    return model


def get_img_array(img_path, size):
    """
    ## The Grad-CAM algorithm
    """
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    ## Create a superimposed visualization
    """
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)


# """
# ## Let's test-drive it
# """
# # Prepare image
# img_array = preprocess_input(get_img_array(img_path, size=img_size))
# # Make model
# model = model_builder(weights="imagenet")
# # Remove last layer's softmax
# model.layers[-1].activation = None
# # Print what the top predicted class is
# preds = model.predict(img_array)
# print("Predicted:", decode_predictions(preds, top=1)[0])
#
# # Generate class activation heatmap
# heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
# # Display heatmap
# plt.matshow(heatmap)
# plt.show()
#
# save_and_display_gradcam(img_path, heatmap)

if __name__ == '__main__':
    create_and_train_model()