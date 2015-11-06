from __future__ import division
import csv
import numpy as np
import time
import random
from math import sqrt, log
#from collections import Counter

inputfile=[]
temp=[]
topics=[]

def calc_variance(clusters,count):
    variance=[]
    for i in range(0,count):
        variance.append(clusters[i].__len__())
    print "variance ", np.var(variance)

def get_distance(u,v,type):
    u=u[1:]
    v=v[1:]
    u=np.array(map(float, u))
    v=np.array(map(float, v))
    if(type==1):    #euclidean
        diff = u - v
        return sqrt(np.dot(diff, diff))
    if(type==2):    #cosine
        return 1 - (np.dot(u, v) / (sqrt(np.dot(u, u)) * sqrt(np.dot(v, v))))


def get_entropy(cluster_dict,size):
    ans=0
    for values in cluster_dict.itervalues():
        P=values/size;
        ans= ans - (P*log(P if P>0 else 1)/log(2))
    return ans

def mean_point(list_of_vectors):
    list_of_vectors=np.array(list_of_vectors)
    return np.mean(list_of_vectors,axis=0)



no = raw_input("Enter the file you want to work with:\n"
               "1. Feature Vector using IDF \n"
               "2. Small data set\n") or '3'
if no=='1':
    filename='dtm_large_data_set.csv'
elif no=='2':
    filename='dtm_small_data_set.csv'
else:
    print "Error: enter valid input"
    exit()

max_iterations=10

nc = (raw_input("Enter the no of clusters(default 10):\n")) or '10'

if not nc.isdigit() or (int(nc)>120 and int(nc)<0):
    print("Enter valid clusters")
    exit()
else:
    nc=int(nc)


dist_type=(raw_input("Enter the distance metric:(default 1) \n"
                        "1. Euclidean\n2. Cosine \n")) or '1'

if not(int(dist_type)==1 or int(dist_type)==2):
    print("Using Euclidean")
    dist_type=1
else:
    dist_type=int(dist_type)

max_iterations=(raw_input("No.of iterations: (default until k-means converges)")) or '3'

if not max_iterations.isdigit() or int(max_iterations)<0:
    print("Enter valid input")
    exit()
max_iterations=int(max_iterations)
#for seperating topics and adding ID values
id=1
with open(filename,'rb') as csvfile:
    csvfile.next()
    reader = csv.reader(csvfile)
    for row in reader:
        topics_temp = row[0:1]
        temp = row[2:]
        temp=map(float, temp)
        for i in range(0,temp.__len__()):
            if temp[i]>0:
                temp[i]=1
            else:
                temp[i]=0
        temp.insert(0,id)
        topics_temp.insert(0,id)
        inputfile.append(temp)
        topics.append(topics_temp)
        id=id+1
        temp=[]
        topics_temp=[]


cluster_track={}

for i in range(1,inputfile.__len__()+1):
    cluster_track[i]=-1

clusters=random.sample(inputfile,nc)

count=len(clusters)

total_updates = 0
no_of_iterations=0

#k-means algorithm starts here
start_time = time.clock()
while True:
    no_of_iterations=no_of_iterations+1
    cluster_lists = [ [] for c in clusters]
    total_updates=0
    #for i in range(0,count):
     #   cluster_lists[i].append(clusters[i])
    for row in inputfile:
        if len(clusters[0])!=0:
            smallest_distance=get_distance(row, clusters[0],dist_type)
        tempIndex=0
        for i in range(count-1):
            if len(clusters[i+1])!=0:
                distance = get_distance(row,clusters[i+1],dist_type)

            if distance<smallest_distance:
                smallest_distance=distance
                tempIndex=i+1
        cluster_lists[tempIndex].append(row)

        if cluster_track[row[0]]!=tempIndex:
            total_updates=total_updates+1
            cluster_track[row[0]]=tempIndex

    if total_updates<=count or no_of_iterations>max_iterations:
        break;
    else:
        t=0
        for k in range(0,cluster_lists.__len__()):
            if cluster_lists[k].__len__()>1:
                clusters[k]=mean_point(cluster_lists[k])
            else:
                max=0
                for l in range(0,count):
                    if cluster_lists[l].__len__()>max:
                        max=cluster_lists[l].__len__()
                        test=l
                clu=random.sample(cluster_lists[test],1)
                clusters[k]=clu[0]
                clu=[]
        count=len(clusters)
    #print no_of_iterations
end_time = time.clock()

#calc_variance(cluster_lists,count)

#cluster validation
majority_count={}
entropies=[]
for i in range (0,cluster_lists.__len__()):
    for j in range (0,cluster_lists[i].__len__()):
        if isinstance(cluster_lists[i][j][0], int):
            key=(topics[int(cluster_lists[i][j][0])-1][1])
            if majority_count.has_key(key):
              majority_count[key]=majority_count.get(key)+1
            else:
                majority_count[key]=1
    entropy=get_entropy(majority_count,cluster_lists[i].__len__())
    entropies.append(entropy)
    majority_count.clear()

#final weighted entropy
final_entropy=0
for i in range(0,cluster_lists.__len__()):
    final_entropy=final_entropy+(entropies[i]*cluster_lists[i].__len__()/inputfile.__len__())


#output
#print entropies
print "Total Offline cost is ", end_time-start_time,"s"
print "Entropy " , final_entropy
