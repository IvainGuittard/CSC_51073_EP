import os
import numpy as np
from skimage.io import imread
from PIL import Image  # Importer PIL pour la compression
from utils import rgba_to_lab, lab_to_rgb, imshow
import cv2

# Define the path to the training set
base_folder_path = 'dataset/landscape_folder'

# Initialize an empty list to store all image file paths
all_image_files = []

# Walk through all directories and subdirectories in the base folder
for root, _, files in os.walk(base_folder_path):
    # Filter and collect .jpg files
    image_files = [os.path.join(root, f) for f in files if f.endswith('.jpg')]
    all_image_files.extend(image_files)

print(len(all_image_files))

# Trier les fichiers si l'ordre est important (optionnel)
image_files.sort()

shapeX, shapeY = 200, 300

# Lire et redimensionner les images
images = []
for image_path in all_image_files[:4001]:
    image = Image.open(image_path)  # Ouvrir l'image avec PIL
    image = image.resize((shapeY, shapeX))  # Redimensionner l'image à (32, 32)
    image = np.array(image)  # Convertir l'image redimensionnée en tableau NumPy
    if len(image.shape) == 3 : 
        if image.shape[2] == 4:  # Si l'image a un canal alpha, il faut le supprimer
            image = image[:, :, :3]
        images.append(image)

# Convertir la liste d'images en un tableau NumPy
images_array = np.stack(images, axis=0)

# images_array_back_n_white = np.array([[[np.mean(pixel[:3]) for pixel in row] for row in image] for image in images_array])
 
images_array_black_n_white = np.zeros((len(images_array),shapeX, shapeY))
images_array_l = np.zeros((len(images_array),shapeX, shapeY,1))
images_array_lab = np.array([rgba_to_lab(images_array[n]) for n in range (len(images_array))])

for n,image in enumerate(images_array):
    images_array_black_n_white[n] = cv2.cvtColor(images_array[n, :, :, :3].astype('uint8'), cv2.COLOR_RGB2GRAY)
    images_array_l[n] = images_array_lab[n, :, :, 0:1].astype('uint8')


       
imshow(images_array_black_n_white[-1])
imshow(images_array_lab[-1])


test_image = np.array(images[-1])
test_image_l = rgba_to_lab(test_image)[:,:,0:1]
test_image_bnw = cv2.cvtColor(test_image[:, :, :3].astype('uint8'), cv2.COLOR_RGB2GRAY)


new_test_image_lab = np.concatenate((
    test_image_l.astype('float32'),
    np.zeros((shapeX, shapeY, 1), dtype='float32'),
    np.zeros((shapeX, shapeY, 1), dtype='float32'),
), axis=-1)

new_test_image = lab_to_rgb(new_test_image_lab)



# Prepare the data (convert to float32 for consistency)
entrees = np.array(images_array_black_n_white, dtype=np.float32)
sorties_a = np.array(images_array_lab, dtype=np.float32)[:,:,:,1]
sorties_b = np.array(images_array_lab, dtype=np.float32)[:,:,:,2]

test_image_bnw_as_rgba = np.array([ [ 3 * [pixel] for pixel in row] for row in test_image_bnw])
imshow(test_image_bnw_as_rgba)
imshow(test_image)
imshow(new_test_image,False)

