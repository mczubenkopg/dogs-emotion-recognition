"""
date: 18-05-23
created by Michal Kopczynski
"""

import os
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

def augment_images(image_path, save_path, augmentations, batches):

    img = load_img(image_path)
    data = img_to_array(img)
    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(**augmentations)
    it = datagen.flow(samples, batch_size=1, save_to_dir=save_path, save_prefix='aug', save_format='jpeg')

    for i in range(batches):
        batch = it.next()

def main():
    
    folder_path = '/dataset/sleepy'
    save_path = '/dataset/sleepy'
    os.makedirs(save_path, exist_ok=True)

    
    augmentations = {
    'rotation_range': 0,  
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}


    
    batches = 3
    for file_name in tqdm.tqdm(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, file_name)
        if os.path.isfile(image_path):
            if image_path.endswith('.jpg') or image_path.endswith('.png') or image_path.endswith('.jpeg'):
                augment_images(image_path, save_path, augmentations, batches)
    print('Image augmentation completed.')

if __name__ == "__main__":
    main()
