
'''
20CS60R40: AKASH SINGH SANT
# Project Code: VC4
# Project Title: Virus Clustering using Complete Linkage Hierarchical Clustering Technique
# '''

import numpy as np
from numpy.linalg import norm
import pandas as pd
import random as rd
from sklearn import preprocessing
import math
import requests

class kMeanAndHeirarch():
    def __init__(self, dataset, RowCount, ColCount):
        self.k = 0
        self.dataset = dataset
        self.NoOfPoints = RowCount
        self.NoOfFeature = ColCount
        self.Centroids = np.array([])
        self.intraMean = {}
        self.clsIntrMean = {}
        self.silhouteCS = []
        self.EucDistance = np.array([])
        self.InterCDis = np.zeros((RowCount, 0))
        self.optimalSillhoute = float('-inf')
        self.ClusSet1 = {}
        self.DistMatrix = np.array([])
        self.ClusSet2 = {}

    # Resetting values

    def ResetVals(self):
        self.Centroids = np.array([])
        self.intraMean = {}
        self.clsIntrMean = {}
        self.EucDistance = np.array([])

    # Normalising the dataset
    def Normalize(self):
        # Normalize the data
        for col in self.dataset.columns[:]:
            mean = self.dataset[col].mean()
            std = self.dataset[col].std()
            self.dataset[col] = (self.dataset[col] - mean)/std

    # intialising random Centroid at beginning
    def InitCentroid(self):
        self.Centroids = np.array([])
        self.Centroids = self.Centroids.reshape(data.shape[1], 0)
        klp = []
        count = self.k
        while(count):
            rand = rd.randint(0, self.NoOfPoints-1)
            if(rand not in klp):
                self.Centroids = np.c_[self.Centroids,
                                       self.dataset.iloc[rand].values]
                count = count-1
                klp.append(rand)
    # Performing the kmeans ALgorithm
    def Kmeans(self, X):
        for n in range(20):
            self.EucDistance = np.array([])
            self.EucDistance = self.EucDistance.reshape(self.NoOfPoints, 0)
            for j in range(self.k):
                dist = np.linalg.norm((X - self.Centroids[:, j]), axis=1)
                self.EucDistance = np.c_[self.EucDistance, dist]
            # C keeps status of points -> Clusters
            C = np.argmin(self.EucDistance, axis=1)
            # Y keeps Points of Respective CLuster
            Y = {}
            for zpx in range(self.k):
                Y[zpx] = np.array([]).reshape(self.NoOfFeature, 0)
            for i in range(self.NoOfPoints):
                Y[C[i]] = np.c_[Y[C[i]], X[i]]
            for l in range(self.k):
                Y[l] = Y[l].T
            # Taking the new Centroid as the mean of the given Centroid for clusters
            for p in range(self.k):
                self.Centroids[:, p] = np.mean(Y[p], axis=0)

        # 2d array keeping the mean distance between sample and other cluster points
        self.InterCDis = np.zeros((self.NoOfPoints, self.k))
        for i in range(self.NoOfPoints):
            # setting the distance between the point and its own cluster as infinite to skip it during selection of b
            self.InterCDis[i][C[i]] = float('inf')
        for i in range(self.NoOfPoints):
            for j in range(self.k):
                if (C[i] != j):
                    # distance between the sample and points of other cluster
                    diff = sum(np.linalg.norm(
                        (Y[j] - X[i]), axis=1))/Y[j].shape[0]  # finding out the mean distance between the sample and other cluster
                    # storing at the respective cluster index for a sample
                    self.InterCDis[i][j] = diff
        # this is for distance bwteen the sample and all points within its own cluster
        intraDis = {}
        for k in range(self.NoOfPoints):
            intraDis[k] = np.array([]).reshape(Y[C[k]].shape[0], 0)
        for kp in range(self.NoOfPoints):
            diff = np.linalg.norm(
                (Y[C[kp]] - X[kp]), axis=1)  # distance between the sample and and other points within its cluster
            intraDis[kp] = np.c_[intraDis[kp], diff]
            # a values for sillhoute coefficient
            self.intraMean[kp] = intraDis[kp].mean()
        li = []
        for jk in range(self.NoOfPoints):
            # taking only the closest cluster mean distance
            b = min(self.InterCDis[jk])
            a = self.intraMean[jk]
            li.append((b-a)/max(b, a))
        tempsill = sum(li)/self.NoOfPoints
        print("Silhoutte Coefficient for k =", self.k, "==>", tempsill)
        if(tempsill > self.optimalSillhoute):  # only Updating when a better k value found
            self.optimalSillhoute = tempsill
            for i in range(self.k):
                result = np.where(C == i)
                self.ClusSet1[i] = result[0]
                tempdiolist = []
                for x in range(len(result[0])):
                    tempdiolist.append(result[0][x])
                self.ClusSet1[i] = tempdiolist
            return self.k
        else:
            return -1

    def Heirach(self):
        for i in range(self.NoOfPoints):
            tempList = []
            tempList.append(i)
            # Dictionary to keep the clusters
            # taking every element as a cluster in the beginning
            self.ClusSet2[i] = tempList
        self.DistMatrix = np.array([])
        self.DistMatrix = self.DistMatrix.reshape(self.NoOfPoints, 0)
        for j in range(self.NoOfPoints):
            dist = np.linalg.norm((X - X[j]), axis=1)
            self.DistMatrix = np.c_[self.DistMatrix, dist]
        # print(self.DistMatrix)
        for i in range(self.NoOfPoints):
            self.DistMatrix[i][i] = float('inf')
        # print(self.DistMatrix)
        for num in range(self.NoOfPoints-self.k):
            minIndexes = np.unravel_index(
                self.DistMatrix.argmin(), self.DistMatrix.shape)  # finding the row and column indices for min element
            if(minIndexes[0] < minIndexes[1]):
                minRow = minIndexes[0]
                maxRow = minIndexes[1]
            else:
                minRow = minIndexes[1]
                maxRow = minIndexes[0]
            for i in range(self.NoOfPoints):
                if(i != maxRow and i != minRow):
                    self.DistMatrix[minRow][i] = max(
                        self.DistMatrix[minRow][i], self.DistMatrix[maxRow][i])
                    self.DistMatrix[i][minRow] = self.DistMatrix[minRow][i]
                self.DistMatrix[maxRow][i] = float(
                    'inf')  # making the rows useless equivalent to deleting
                self.DistMatrix[i][maxRow] = float(
                    'inf')  # making the columns useless equivalent to deleting
            for k in self.ClusSet2[maxRow]:
                self.ClusSet2[minRow].append(k)  # merging the cluster
            # deleting the dictionary item after merging the cluster
            self.ClusSet2.pop(maxRow)

    # For Writing kmeans.txt and reporting Optimal K
    def reportKmeans(self):
        print("\nOptimal K =", self.k)
        print("")
        minstartvalue=[]
        clusterorder=[]
        for i in range(self.k):
            self.ClusSet1[i].sort()
            minstartvalue.append(self.ClusSet1[i][0])
        for i in range(self.k):
            minv=min(minstartvalue)
            minind=minstartvalue.index(minv)
            clusterorder.append(minind)
            minstartvalue[minind]=math.inf
        with open("kmeans.txt", "w") as f:
            for i in clusterorder:
                f.write(str(self.ClusSet1[i])[1:-1])
                f.write("\n")
        print("kmeans.txt generated")
    # for Writing Agglormerative.txt
    def reportAgglomerative(self):
        with open("agglomerative.txt", "w") as w:
            for key in self.ClusSet2.keys():
                self.ClusSet2[key].sort()
                w.write(str(self.ClusSet2[key])[1:-1])
                w.write("\n")
        print("agglomerative.txt generated\n")

    # For Printing Jaccard similarity
    def reportJacc(self):
        j = 0
        for i in range(self.k):
            for key in self.ClusSet2.keys():
                s1 = set(self.ClusSet1[i])
                s2 = set(self.ClusSet2[key])
                print("Jaccard similarity, kmeans Cluster", i+1, "<==> Heirarchical Cluster",
                      j+1, "=", float(len(s1.intersection(s2)) / len(s1.union(s2))))
                j += 1
            j = 0
            print("")
def downloadcsv(DATA_PATH):
    with open(DATA_PATH,"wb") as fil:
        response=requests.get("http://cse.iitkgp.ac.in/~aritrah/course/theory/ML/Spring2021/projects/Project3/VC4/virus_4_unlabelled.csv")
        fil.write(response.content)



def PreProcess(data):
    num = preprocessing.LabelEncoder()
    for column in data.columns:
        if (data[column].dtypes == 'object'):
            data[column] = num.fit_transform(data[column])
    data = data.fillna(-999)
    return data

DATA_PATH = 'virus_4_unlabelled.csv'
downloadcsv(DATA_PATH)
data = pd.read_csv(DATA_PATH)
data = data.iloc[:, 1:]
data = PreProcess(data)
PrObj = kMeanAndHeirarch(data, data.shape[0], data.shape[1])
PrObj.Normalize()
X = data.iloc[:].values
optimal = 0
for z in range(3, 7):
    PrObj.ResetVals()
    PrObj.k = z
    PrObj.InitCentroid()
    tempvar = PrObj.Kmeans(X)
    if(tempvar > optimal):
        optimal = tempvar
PrObj.k = optimal
PrObj.reportKmeans()
PrObj.Heirach()
PrObj.reportAgglomerative()
PrObj.reportJacc()
