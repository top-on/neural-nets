"""Classify faces, and get feature extraction for faces."""

from keras_vggface.utils import preprocess_input, decode_predictions
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
import numpy as np


def load_preprocess(img_path, shape=(224, 224)):
    """Load and preprocess image."""
    x = image.load_img(img_path, target_size=shape)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, version=1)  # or version=2
    return x


# load model (with person classifier)
vggface = VGGFace(model='resnet50')

# classify image
img = load_preprocess('src/faces/chen2.jpg')
preds = vggface.predict(x)
print('Predicted: {};\nConfidence: {}'.format(
    decode_predictions(preds)[0][0][0],
    decode_predictions(preds)[0][0][1]))

# load model (feature extraction)
vggface_topless = VGGFace(model='resnet50', include_top=False)
face_chen1 = 'src/faces/chen.jpg'
face_chen2 = 'src/faces/chen2.jpg'
face_sheldon1 = 'src/faces/sheldon.jpg'
face_sheldon2 = 'src/faces/sheldon2.jpg'
chen1 = vggface_topless.predict(load_preprocess(face_chen1))
chen2 = vggface_topless.predict(load_preprocess(face_chen2))
sheldon1 = vggface_topless.predict(load_preprocess(face_sheldon1))
sheldon2 = vggface_topless.predict(load_preprocess(face_sheldon2))

# check correlation
np.correlate(chen1.flatten(), chen2.flatten())
np.correlate(chen1.flatten(), sheldon1.flatten())
np.correlate(chen2.flatten(), sheldon2.flatten())
np.correlate(chen1.flatten(), sheldon2.flatten())
np.correlate(sheldon1.flatten(), sheldon2.flatten())
# Result: correlation between different face: ~3000, between same persons ~9000
