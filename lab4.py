import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import csv
from PIL import Image


parser = argparse.ArgumentParser(description='Predict.')
parser.add_argument('inputtrain', type=str, help='File with the input train data')


args = parser.parse_args()
print(args.inputtrain)


train_dataset = h5py.File(args.inputtrain, "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # train set imagenes

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)

train_set_x_flatten = train_set_x_flatten/255.


print('Imagenes de entrenamiento:', train_set_x_flatten.shape)


def distance(a1,a2):
    return np.sqrt(np.sum(np.square(a1-a2)))
class KMeans():
    def __init__(self,k=5,max_iter=100):
        self.k= k
        self.max_iter=max_iter
        self.clusters = {}
        self.label = []
        
    def initialization(self,X):
        for i in range(self.k):
            j = np.random.randint(0,X.shape[0])
            center = X[j]
            points = []
            cluster = {
                'center':center,
                'points':points,
                'id'    :i
            }
            self.clusters[i]=cluster
        self.label = np.zeros((X.shape[0],1))
    
    def assignPointTOClusters(self,X):
        for i in range(X.shape[0]):
            dist = []
            curr_x = X[i]
            for ki in range(self.k):
                d = distance(curr_x,self.clusters[ki]['center'])
                dist.append(d)
            
            current_cluster = np.argmin(dist)
            self.clusters[current_cluster]['points'].append(curr_x)
            self.label[i]=(self.clusters[current_cluster]['id'])
            
    def check(self,old_c,new_c):
        distances = [distance(old_c[i], new_c[i]) for i in range(self.k)]
        return sum(distances) == 0
        
    def updateClusters(self):
        for kx in range(self.k):
            pts = np.array(self.clusters[kx]['points'])
            
            if pts.shape[0]>0: # If cluster has some nonzero points
                new_u = pts.mean(axis=0)
                self.clusters[kx]['center'] = new_u
                # Clear the list
                self.clusters[kx]['points'] = []
    
    def plotClusters(self):
        for kx in range(self.k):
            #print(len(self.clusters[kx]['points']))
            pts = np.array(self.clusters[kx]['points'])
            # plot points , cluster center
            try:
                plt.scatter(pts[:,0],pts[:,1])
            except:
                pass
            uk = self.clusters[kx]['center']
        #plt.show()
            
    def fit(self,X):
       # print(self.k)
        self.initialization(X)
        for i in range(self.max_iter):
            
            self.assignPointTOClusters(X)
            self.plotClusters()
            old_c = [self.clusters[i]['center'] for i in range(self.k)]
            self.updateClusters()
            new_c = [self.clusters[i]['center'] for i in range(self.k)]
            if self.check(old_c,new_c):
                break
            
k = 2
cats = KMeans(k)
cats.fit(train_set_x_flatten)

with open('cats_clustering.csv', 'w') as csvfile:
    pwriter = csv.writer(csvfile)
    for i in range(train_set_x_flatten.shape[0]):
        pwriter.writerow([i, int(cats.label[i][0])])
