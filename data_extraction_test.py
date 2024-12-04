import os
import numpy as np
from skimage.io import imread
from utils import rgba_to_lab, imshow

# Définir le chemin du dossier contenant les images
folder_path = 'dataset/flowers'

# Récupérer tous les fichiers PNG du dossier
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')][:99]

print(len(image_files))

# Trier les fichiers si l'ordre est important (optionnel)
image_files.sort()

shapeX, shapeY = 200, 200
# Lire et empiler les images
images = []
for file in image_files:
    image_path = os.path.join(folder_path, file)
    image = imread(image_path)
    
    n,m,_ = image.shape
    
    if (shapeX < n and shapeY < m) :
        images.append(image[:shapeX,:shapeY,:])
    elif (shapeX < n):
        new_image = np.zeros((shapeX,shapeY,3))
        new_image[:,:m,:] = image[:shapeX,:,:]
        images.append(new_image)
    elif (shapeY < m):
        new_image = np.zeros((shapeX,shapeY,3))
        new_image[:n,:,:] = image[:,:shapeY,:]
        images.append(new_image)
    else :
        new_image = np.zeros((shapeX,shapeY,3))
        new_image[:n,:m,:] = image
        images.append(new_image)

# Convertir la liste d'images en un tableau NumPy
images_array = np.stack(images[:-1], axis=0)

# images_array_back_n_white = np.array([[[np.mean(pixel[:3]) for pixel in row] for row in image] for image in images_array])
 
images_array_black_n_white = np.zeros((len(images_array),shapeX, shapeY,1))
images_array_lab = np.array([rgba_to_lab(images_array[n]) for n in range (len(images_array))])

for n,image in enumerate(images_array):
    images_array_black_n_white[n] = images_array_lab[n,:,:,0:1]
       
imshow(images_array_black_n_white[-1])
imshow(images_array_lab[-1])


test_image = np.array(images[-1])
test_image_lab = rgba_to_lab(test_image)
test_image_bnw = test_image_lab[:,:,0:1]

