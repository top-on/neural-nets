"""TRANSFER LEARNING TO DISTINGUISH CAR AND BIKE TIRES."""

import os
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
import numpy as np
import PIL

def build_model() -> Model:
    """Build Keras model, based on ResNet50, trained on imagenet."""
    # download link: https://github.com/fchollet/deep-learning-models/releases/
    model = keras.applications.resnet50.ResNet50(include_top=False,
                                                 weights='imagenet',
                                                 input_tensor=None,
                                                 input_shape=None,
                                                 pooling=None,
                                                 classes=1000)
    # make existing layers not trainable
    for layer in model.layers:
        layer.trainable = False
    # add layer
    x = model.output
    x = Dense(2048, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    # create new model
    model_final = Model(inputs=model.input, outputs=predictions)
    model_final.compile(loss="categorical_crossentropy",
                        optimizer="Adam",
                        metrics=["accuracy"])
    return model_final

def preprocess_image(image_path : PIL.Image.Image) -> np.ndarray:
    """Preprocess image for ResNet50."""
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    return img

if __name__ == '__main__':
    # build model
    model = build_model()

    # path to image
    path = 'data/images/bicycle tire profile closeup -car/'
    files = os.listdir(path)
    image_path = path + files[7]

    # load image
    image = preprocess_image(image_path)

    # fit on one image
    labels = {'bike': 0,
              'car': 1}
    model.fit(x=image, y=np.array([[[[1, 0]]]]), epochs=10)

    # predict
    prediction = model.predict(image)
    print('Prediction for trained image: {}'.format(prediction))
