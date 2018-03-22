"""PREDICT IMAGE CONTENT"""

import os
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
import numpy as np

def main():
    model = keras.applications.resnet50.ResNet50(include_top=True,
                                                 weights='imagenet',
                                                 input_tensor=None,
                                                 input_shape=None,
                                                 pooling=None,
                                                 classes=1000)

    # images
    path = '../../data/images/tire/'
    files = os.listdir(path)
    image_path = path + files[1]

    # load image
    img = load_img(image_path, target_size=(224, 224))
    image = img_to_array(img)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # predict
    prediction = model.predict(image)
    n = 10
    top_n = np.argsort(prediction)[:, -n:][0, :][::-1]

    # translate classes to labels
    with open('imagenet_classes.txt', 'r') as handle:
        labels = eval(handle.read())
    predictions = [labels[i] for i in top_n]
    print('top predictions for image:')
    print(predictions)

if __name__ == '__main__':
    main()
