import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from skimage import data, filters, color, morphology
from skimage.segmentation import flood, flood_fill
from PIL import Image

# source ~/anaconda3/bin/deactivate base

image_array = skimage.io.imread(fname="../path_img/00000.ppm")
print(type(image_array))

cat = data.chelsea()

# Change the cameraman's coat from dark to light (255).  The seed point is
# chosen as (155, 150)
print(cat.shape, type(cat))
cat_sobel = filters.sobel(cat[..., 0])
cat_nose = flood(cat_sobel, (240, 265), tolerance=0.03)
print(cat_nose.shape, type(cat_nose))
fig, ax = plt.subplots(nrows=3, figsize=(10, 20))

ax[0].imshow(cat)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(cat_sobel)
ax[1].set_title('Sobel filtered')
ax[1].axis('off')

# ax[2].imshow(image_array)
ax[2].imshow(cat_nose)
ax[2].plot(265, 240, 'wo')  # seed point
ax[2].set_title('Nose segmented with `flood`')
ax[2].axis('off')

fig.tight_layout()
plt.show()