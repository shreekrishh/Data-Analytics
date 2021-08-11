import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from numpy import exp              #importing all necessary modules

test1 = pd.read_csv('C:\KADG task\kdag2\ds1_test.csv')
data1 = pd.read_csv('C:\KADG task\kdag2\ds1_train.csv')
test2 = pd.read_csv('C:\KADG task\kdag2\ds2_test.csv')
data2 = pd.read_csv('C:\KADG task\kdag2\ds2_train.csv')

weights = np.array([0.01, 0.001])   #theta
bias = 0                            #thetanot
iterations = 15000
inputs = np.array(list(zip(data1['x_1'], data1['x_2'])))
ys = np.array(list(zip(data1['y'])))

alpha = 0.000001
x1arr = np.array(list(zip(data1['x_1'])))
x2arr = np.array(list(zip(data1['x_2'])))
# feature scaling start
meanx1 = np.sum(x1arr) / len(ys)
meanx2 = np.sum(x2arr) / len(ys)
sum1 = 0
for i in range(len(ys)):
    sum1 += (x1arr[i] - meanx1) ** 2
SD1 = np.sqrt(sum1 / len(ys))
sum2 = 0
for i in range(len(ys)):
    sum2 += (x2arr[i] - meanx2) ** 2
SD2 = np.sqrt(sum2 / len(ys))
x1fes = np.empty(len(ys))
x2fes = np.empty(len(ys))
for i in range(len(ys)):
    t = (x1arr[i] - meanx1) / SD1
    x1fes[i] = t
for i in range(len(ys)):
    t = (x2arr[i] - meanx2) / SD2
    x2fes[i] = t
inputsfes = []
for i in range(len(ys)):
    inputsfes.append([x1fes[i], x2fes[i]])

#define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

#optimization by gradient descent algo
for i in range(iterations):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(len(ys)):
        sum1 += (sigmoid((np.dot(np.transpose(weights), inputsfes[i])+bias)-ys[i]) * inputsfes[i][0])
        sum2 += (sigmoid((np.dot(np.transpose(weights), inputsfes[i])+bias)-ys[i]) * inputsfes[i][1])
        sum3 += (sigmoid((np.dot(np.transpose(weights), inputsfes[i])+bias)-ys[i])*1)
    weights[0] = weights[0] - (alpha * sum1) / len(ys)
    weights[1] = weights[1] - (alpha * sum2) / len(ys)
    bias = bias - (alpha * sum3) / len(ys)
print(weights)
print(bias)
#testing
testinputs = np.array(list(zip(test1['x_1'], test1['x_2'])))
ystests = np.array(list(zip(test1['y'])))

testx1arr = np.array(list(zip(test1['x_1'])))
testx2arr = np.array(list(zip(test1['x_2'])))
#feature scaling
testmeanx1 = np.sum(testx1arr) / len(ystests)
testmeanx2 = np.sum(testx2arr) / len(ystests)
sum1 = 0
for i in range(len(ystests)):
    sum1 += (testx1arr[i] - testmeanx1) ** 2
SD1 = np.sqrt(sum1 / len(ystests))
sum2 = 0
for i in range(len(ystests)):
    sum2 += (testx2arr[i] - testmeanx2) ** 2
SD2 = np.sqrt(sum2 / len(ystests))
testx1fes = np.empty(len(ystests))
testx2fes = np.empty(len(ystests))
for i in range(len(ystests)):
    t = (testx1arr[i] - testmeanx1) / SD1
    testx1fes[i] = t
for i in range(len(ystests)):
    t = (testx2arr[i] - testmeanx2) / SD2
    testx2fes[i] = t
testinputsfes = []

for i in range(len(ystests)):
    testinputsfes.append([testx1fes[i], testx2fes[i]])
#test
accu=0
for i in range(len(ystests)):
    if sigmoid(np.dot(np.transpose(weights),testinputsfes[i])+bias) >= 0.5 and ystests[i]==1:
        accu+=1
    elif sigmoid(np.dot(np.transpose(weights),testinputsfes[i])+bias) < 0.5 and ystests[i]==0:
        accu+=1


accuper= (accu/len(ystests))*100
print(accuper)


