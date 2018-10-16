import os
import os.path
import math
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
from sklearn.manifold import TSNE
import h5py
import sklearn
from sklearn.svm import LinearSVC

data_file='data_4.h5'
with h5py.File(data_file, 'r') as hf:
		X = hf['x'][:]
		Y = hf['y'][:]
#print(list(hf.keys()))
        
  data set 1 # X=X**2
# data set 2 #X= TSNE(n_components=2, perplexity=20, verbose=2).fit_transform(X)
# data set 3 #X= TSNE(n_components=2, perplexity=10, verbose=2).fit_transform(X)



X1=X_embedded[:,0]
Y1=X_embedded[:,1]
#Z1=X_embedded[:,2]
arr=[]
for j in range(0,len(Y)):
    #for j in range(0,len(Y)):
    if 1==1:
        if Y[j]==1:
            arr.append('black')
        elif Y[j] ==2:
            arr.append('blue')
        else:
            arr.append('red')


print("values of x and Y")
print (X1)
print("values of x and Y")
print(Y1)


#2d plot........................................
plt.figure()
#X1_root=np.sqrt(X1)
plt.scatter(X1,Y1,s=25,color=arr[:])
#plt.scatter(X1,-(X1**2),s=25,color=arr[:])
#plt.scatter(X1*X1,np.abs(Y1),s=25,color=arr[:])
plt.savefig('.\\Plot\\Visualize_Data_'+data_file.split("\\")[1]+'.png')
plt.show()


#3D plot--------------------------------------------

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X1,Y1,Z1,s=25, color=arr[:])
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')




# plt.show()