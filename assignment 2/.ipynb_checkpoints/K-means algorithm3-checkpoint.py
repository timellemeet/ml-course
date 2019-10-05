

import imageio
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.image import imread

from sklearn.cluster import KMeans

#Read the image
img = imageio.imread(r'\\campus.eur.nl\users\students\408055ww\Documents\resized_image.jpg')
plt.imshow(img)

img_size = img.shape

# Reshape it to be 2-dimension
X = img.reshape(img_size[0] * img_size[1], img_size[2])


rng = np.random.RandomState(seed=12345)
seeds = np.arange(10**5)
rng.shuffle(seeds)
seeds = seeds[:20] # Select the first 20 seeds
split_ratios = [0.1, 0.5, 0.9]

for count, split_ratio in enumerate(split_ratios):
 ssd = []
 cen= []
 n_inter = []
 for i in seeds:
    
    k=2
    km = KMeans(n_clusters= k)
    km.fit(X)
    
    # Getting the cluster labels
    labels = km.predict(X)
    
    # Centroid values
    centroids = km.cluster_centers_
    cen.append(centroids)
    
    #Sum of squared distances of samples to their closest cluster center.
    sum_sqdist = km.inertia_
    ssd.append(sum_sqdist)
  
    #Number of iterations run the k-mean algorithm
    iterations = km.n_iter_
    n_inter.append(iterations)
    
#    print('The value of the objective function is' , sum_sqdist)

# Find the minimum of the sum of squared distances of samples to their closest cluster center.
min_sum_sqdist= min(ssd) 

#Find the position of the minimum value of the objective function of the 
index = ssd.index(min_sum_sqdist)

#The respective centroids of the minimun value of the objective function
best_cent = cen[index]

#The respective iterations of the minimun value of the objective function
iter = n_inter[index]

# Use the rational centroids to compress the image
X_compressed = best_cent[km.labels_]
X_compressed = np.clip(X_compressed.astype('uint8'), 0, 255)
    
# Reshape X_recovered to have the same dimension as the original image
X_compressed = X_compressed.reshape(img_size[0], img_size[1], img_size[2])
    
#Plot the original and the compressed image next to each other
fig, ax = plt.subplots(1, 2, figsize = (12, 8))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[1].imshow(X_compressed)
ax[1].set_title('Compressed Image' )
for ax in fig.axes:
    ax.axis('off')  


print('The minimum value of the objective function is' , min_sum_sqdist )
print('The respective centroids are' ,cen[index])
print('The respective iterations are' ,n_inter[index])
    