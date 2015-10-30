import csv
import numpy as np
import time
import random
from math import sqrt

inputfile=[]
temp=[]
topics=[]

def get_distance(u,v,type):
    u=u[1:]
    v=v[1:]
    u=np.array(map(int, u))
    v=np.array(map(int, v))
    if(type==1):    #euclidean
        diff = u - v
        return sqrt(np.dot(diff, diff))
    if(type==2):    #cosine
        return 1 - (np.dot(u, v) / (sqrt(np.dot(u, u)) * sqrt(np.dot(v, v))))


def mean_point(list_of_vectors):
    list_of_vectors=np.array(list_of_vectors)
    return np.mean(list_of_vectors,axis=0)

id=1

with open('dtm_idf_small.csv','rb') as csvfile:
    csvfile.next()
    reader = csv.reader(csvfile)
    for row in reader:
        temp = row[2:]
        temp=map(int, temp)
        temp.insert(0,id)
        inputfile.append(temp)
        id=id+1
        temp=[]

cluster_track={}

for i in range(1,inputfile.__len__()+1):
    cluster_track[i]=-1

k=2
dist_type=2
max_iterations=10

clusters=random.sample(inputfile,k)

count=len(clusters)

total_updates = 0
no_of_iterations=0
start_time = time.clock()
while True:
    no_of_iterations=no_of_iterations+1
    cluster_lists = [ [] for c in clusters]
    total_updates=0
    for row in inputfile:
        smallest_distance=get_distance(row, clusters[0],dist_type)
        tempIndex=0
        for i in range(count-1):
            distance = get_distance(row,clusters[i+1],dist_type)

            if distance<smallest_distance:
                smallest_distance=distance
                tempIndex=i+1
        cluster_lists[tempIndex].append(row)

        if cluster_track[row[0]]!=tempIndex:
            total_updates=total_updates+1
            cluster_track[row[0]]=tempIndex

    if total_updates==0 or no_of_iterations>max_iterations:
        break;
    else:
        for k in range(0,cluster_lists.__len__()):
            clusters[k]=mean_point(cluster_lists[k])

end_time = time.clock()
print "Total Offline cost is ", end_time-start_time,"s"

print(no_of_iterations)
