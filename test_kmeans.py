from PIL import Image
import numpy as np
from kmeans import *
import pandas as pd


im = Image.open('north-africa-1940s-grey.png')
h, w = im.size
X = np.asarray(im)

k=4
centroids, labels = kmeans(X, k=k, centroids='kmeans++', tolerance=.01)
centroids = centroids.astype(np.uint8)
X = centroids[labels] # reassign all points

print('done')











