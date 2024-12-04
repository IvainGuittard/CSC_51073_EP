from data_extraction_test import *
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Activation, Input
import numpy as np
import matplotlib.pyplot as plt
from utils import lab_to_rgb, imshow

# Initialize the model
model_a = Sequential()
model_b = Sequential()

layer_size = 64
kernel_shape = 4,4


def setup_model(model):
    # Input layer
    model.add(Input(shape=(shapeX,shapeY,1)))

    # Convolutional layers
    model.add(Conv2D(layer_size, kernel_shape, activation='relu', padding='same'))
    model.add(Conv2D(layer_size, kernel_shape, activation='relu', padding='same'))
    model.add(Conv2D(layer_size, kernel_shape, activation='relu', padding='same'))

    # Upsampling layers (transpose convolutions)
    model.add(Conv2DTranspose(layer_size, kernel_shape, activation='relu', padding='same'))
    model.add(Conv2DTranspose(layer_size, kernel_shape, activation='relu', padding='same'))

    # Final output layer with 1 chanel
    model.add(Conv2D(1, kernel_shape, activation='linear', padding='same'))

    # Compile the model
    model.compile(loss="mean_squared_error", optimizer='adam')

# A
setup_model(model_a)
new_test_image_a = model_a.predict(np.array([test_image_bnw]))[0]

# B
setup_model(model_b)
new_test_image_b = model_b.predict(np.array([test_image_bnw]))[0]

new_test_image_lab = np.concatenate((
    test_image_bnw,
    new_test_image_a,
    new_test_image_b
), axis=-1)
new_test_image = lab_to_rgb(new_test_image_lab)


# Prepare the data (convert to float32 for consistency)
entrees = np.array(images_array_black_n_white, dtype=np.float32)
sorties_a = np.array(images_array_lab, dtype=np.float32)[:,:,:,1:2]
sorties_b = np.array(images_array_lab, dtype=np.float32)[:,:,:,2:3]

test_image_bnw_as_rgba = np.array([ [ 3 * [pixel[0]] for pixel in row] for row in test_image_bnw])
imshow(test_image_bnw_as_rgba)
imshow(test_image)
imshow(new_test_image,False)

def train(n=1):
    model_a.fit(x=entrees, y=sorties_a, epochs=n)

    model_b.fit(x=entrees, y=sorties_b, epochs=n)
    new_test_image_a = model_a.predict(np.array([test_image_bnw]))[0]
    new_test_image_b = model_b.predict(np.array([test_image_bnw]))[0]
    
    
    new_test_image_lab = np.concatenate((
        test_image_bnw,
        new_test_image_a,
        new_test_image_b,
    ), axis=-1)
    new_test_image = new_test_image_lab - 128
    new_test_image[:,:,0] += 128
    new_test_image[:,:,0] /= 2.55
    
    imshow(lab_to_rgb(new_test_image))
    imshow(new_test_image)
    imshow(lab_to_rgb(new_test_image),False)
    

