"""Example for augmenting image."""

from keras.preprocessing.image import ImageDataGenerator, load_img, \
    img_to_array

# load img to augment
x = load_img('src/image_augmentation/input/bike/bike.png')
x = img_to_array(x)
x = x.reshape((1,) + x.shape)


datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

i = 0
for batch in datagen.flow(x,
                          batch_size=1,
                          save_to_dir='src/image_augmentation/output/',
                          save_prefix='bike',
                          save_format='png'):
    i += 1
    if i > 20:
        break
