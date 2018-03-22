"""TRANSFER LEARNING TO DISTINGUISH CAR AND BIKE TIRES."""

import os
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense
import numpy as np
import PIL


def build_model() -> Model:
    """Build Keras model, based on ResNet50, trained on imagenet."""
    # download link: https://github.com/fchollet/deep-learning-models/releases/
    model_trained = keras.applications.resnet50.ResNet50(include_top=False,
                                                         weights='imagenet',
                                                         input_tensor=None,
                                                         input_shape=None,
                                                         pooling=None,
                                                         classes=1000)
    # make existing layers not trainable
    for layer in model_trained.layers:
        layer.trainable = False
    # add layer
    x = model_trained.output
    x = Dense(2048, activation="relu")(x)
    predictions = Dense(2, activation="softmax")(x)
    # create new model
    model_final = Model(inputs=model_trained.input, outputs=predictions)
    model_final.compile(loss="categorical_crossentropy",
                        optimizer="Adam",
                        metrics=["accuracy"])
    return model_final


def preprocess_image(img_path: PIL.Image.Image) -> np.ndarray:
    """Preprocess image for ResNet50."""
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    return img


if __name__ == '__main__':
    # build model
    model = build_model()

    # training data
    images = np.array([])
    labels = []
    label_encoding = {'bike': np.array([[[1, 0]]]),
                      'car': np.array([[[0, 1]]])}

    # add bike images
    path = 'data/images/bicycle tire profile closeup -car/'
    files = os.listdir(path)
    bike_images = [preprocess_image(path + file) for file in files]
    bike_images = np.array(bike_images)
    bike_labels = [label_encoding['bike'] for i in range(len(bike_images))]
    bike_labels = np.array(bike_labels)

    # add car images (TODO: DRY pattern)
    path = 'data/images/car tire profile closeup -bicycle/'
    files = os.listdir(path)
    car_images = [preprocess_image(path + file) for file in files]
    car_images = np.array(car_images)
    car_labels = [label_encoding['car'] for i in range(len(car_images))]
    car_labels = np.array(car_labels)

    # combine numpy arrays
    images = np.concatenate((bike_images, car_images), axis=0)
    labels = np.concatenate((bike_labels, car_labels), axis=0)

    # fit on one image
    images = preprocess_input(images, mode='tf')
    history = model.fit(x=np.array(images),
                        y=np.array(labels),
                        epochs=3,
                        validation_split=0.2,
                        verbose=2,
                        batch_size=20)
    print(history.history)

    # predict
    img = images[-1]
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    prediction = model.predict(img)
    print('Prediction for trained image: {}'.format(prediction))
