# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import imageio
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import euclidean

#Define number of pixels of the image
basewidth = 100
baseheight = 100
img = Image.open('sample.jpg','r')
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), Image.ANTIALIAS)
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), Image.ANTIALIAS)
img.save('resized_image.jpg')

#Shape the image to the second dimension
imgred = imageio.imread(r'\\campus.eur.nl\users\students\408055ww\Documents\resized_image.jpg')
X = imgred.reshape((imgred.shape[0]*imgred.shape[1]), imgred.shape[2])

#number of clusters
K = 16

#Spectral clustering algorithm
model = SpectralClustering(n_clusters=K, affinity='nearest_neighbors',
                         assign_labels='kmeans')
labels = model.fit_predict(X)
centroid = np.zeros((3,K))

#Find the centroid for each cluster
Xlabel = np.c_[X, labels]
for i in range(0, K):
    df = pd.DataFrame(Xlabel)
    df0 = df[df[3]==i]
    df1 = df0.drop(3,axis=1)
    df2 = df1.mean(axis=0)
    centroid[:,i] = np.array(df2)
centroid = np.transpose(centroid)

#Shape back into third dimension
X_compressed = centroid[labels]
X_compressed = np.clip(X_compressed.astype('uint8'), 0, 255)
X_compressed = X_compressed.reshape(imgred.shape[0], imgred.shape[1], imgred.shape[2])

#Plot the images
fig, ax = plt.subplots(1, 2, figsize = (12, 8))
ax[0].imshow(imgred)
ax[0].set_title('Original Image')
ax[1].imshow(X_compressed)
ax[1].set_title('Compressed Image')
for ax in fig.axes:
    ax.axis('off')













