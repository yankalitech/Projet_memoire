import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open("../path_img/00000.ppm")
image_array = np.array(image)
print('image :', image_array)
print('classe :', type(image_array))
print('type :', image_array.dtype)
print('taille :', image_array.shape)
print('modifiable :', image_array.flags.writeable)
print('image area', image_array[0][0])
# plt.imshow(image_array)
# plt.show()

def determine_k_grayscale(tab_rgb):
    k = 0
    gray_tab_rgb = []
    print('int', tab_rgb)
    if tab_rgb[0] != 0 or tab_rgb[1] != 0 or tab_rgb[2] != 0:
        k = (tab_rgb[0] + tab_rgb[1] + tab_rgb[2])/3

        for elt in tab_rgb:
            if elt != 0:
                b = k/elt
                gray_tab_rgb.append(b * elt)
            else:
                gray_tab_rgb.append(elt)
    print('out', gray_tab_rgb)
    return gray_tab_rgb

def new_numpy_img(old_image):
    image = old_image
    print(len(old_image))
    for i in range(len(old_image)):
        for j in range(len(old_image[i])):
            image[i][j]= determine_k_grayscale(old_image[i][j])
    return image


imag = new_numpy_img(image_array)
plt.imshow(imag)
plt.show()


