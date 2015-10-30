import csv
import numpy as np
import time
from textblob.classifiers import NaiveBayesClassifier
#sample input form for NB classifier
#train = [
#     ('I love this sandwich.', 'pos'),
#     ('this is an amazing place!', 'pos')]
# input format for Naive Bayes

inputfile=[]
temp=[]
topics=[]

no = raw_input("Enter the file you want to work with:\n"
               "1. Feature Vector \n"
               "2. Feature Vector using IDF \n"
               "3. Small dataset")

if no=='1':
    filename='dtm_sparse_term_removed.csv'
elif no=='2':
    filename='dtm_idf.csv'
elif no=='3':
    filename='dtm_idf_small.csv'
else:
    print "Error: enter valid input"
    exit()
with open(filename,'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        temp = row
        inputfile.append(temp)
        temp=[]

inputfile = np.array(inputfile)
len = inputfile[0].__len__()

topics = inputfile[:,0]
inputfile = inputfile[:,2:len]

final_topic_feature=[]

sentence=""
temp=[]
id=0
for i in range (1,inputfile.__len__()):
    for j in range (0,inputfile[0].__len__()):
        if(float(inputfile[i][j]) != 0.0):
            sentence = sentence+inputfile[0][j] + " "
    temp=[sentence,topics[i]]
    final_topic_feature.append(temp)
    sentence=""
    temp=[]
  #  id=id+1
  #  if(id == 100):
  #      break

total_len = final_topic_feature.__len__();

test_perc = raw_input("Enter the test percentage in number eg) 20 , 30 ..")

train_len = total_len * (100 - int(test_perc))*0.01

index = 0
trainset=[]
testset=[]
for data in final_topic_feature:
    if index < train_len:
        trainset.append(data)
    else:
        testset.append(data)
    index=index+1

print "***************Naive Bayes Classification******************"
starttime = time.clock()
cl = NaiveBayesClassifier(trainset)
endtime = time.clock()

print "Total Offline cost is ", endtime-starttime,"s"

for i in range(0,10):
    starttime = time.clock()
    cl.classify(testset[i][0])
    endtime = time.clock()
    print "Total Online cost of ", i+1, "is ",endtime-starttime,"s"

startime = time.clock()
print "Accuracy of the model", cl.accuracy(testset)*100
endtime = time.clock()

print "Total test time is ", endtime-starttime,"s"