from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import layers, models, losses, optimizers, metrics, preprocessing
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import PIL

BASE_DIR = Path('./Datasets')


def create_model():
    base_model = tf.applications.Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3),
                                          pooling='avg')
    x = base_model.output
    outputs = layers.Dense(5, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    for layer in model.layers[:40]:
        layer.trainable = False
    loss = losses.categorical_crossentropy
    optimizer = optimizers.RMSprop(lr=0.0001)
    metric = [metrics.categorical_accuracy]
    model.compile(optimizer, loss, metric)
    train_generator, val_generator = get_datasets()
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=80,
        epochs=300,
        validation_data=val_generator,
        validation_steps=20)
    # save model architecture
    model_json = model.to_json()
    open('xception_model.json', 'w').write(model_json)

    # save model's learned weights
    model.save_weights('image_classifier_xception.h5', overwrite=True)

    return model


def preprocess_image(x):
    x /= 255.
    x -= 0.5
    x *= 2.

    # 'RGB'->'BGR'
    x = x[..., ::-1]
    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    return x


def get_datasets():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_image)
    train_generator = train_datagen.flow_from_directory(
        os.path.join(BASE_DIR, "imageNet_dataset/train"),
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(BASE_DIR, "imageNet_dataset/validation"),
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical',
        shuffle=True)
    return train_generator, validation_generator


def get_img_array(img_path, size):
    """Formatting img to numpy array"""
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Making heatmaps"""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.5):
    """Use of gradcams"""

    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

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


def main():
    """Main function"""
    img_size = (299, 299)
    last_conv_layer_name = "block14_sepconv2_act"

    # The local path to our target image
    img_path = "/Users/mkopczynski/Desktop/deeplab/deeplab/model/scripts/happy.jpg"

    # Prepare image
    img_array = get_img_array(img_path, size=img_size)  # Add your own preprocess_input function if necessary

    # Load your custom model
    model = keras.models.load_model("/Users/mkopczynski/Desktop/deeplab/deeplab/model/model_fine_final.h5")

    # Remove last layer's softmax
    model.layers[-1].activation = None

    # Predict
    preds = model.predict(img_array)
    # Add your own decode_predictions function if necessary, or print directly the class index
    print("Predicted:", np.argmax(preds[0]))

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    save_and_display_gradcam(img_path, heatmap)


if __name__ == "__main__":
    main()
