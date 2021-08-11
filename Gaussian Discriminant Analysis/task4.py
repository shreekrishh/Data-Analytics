import np as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

test1 = pd.read_csv('C:\KADG task\kdag4\ds1_test.csv')
data1 = pd.read_csv('C:\KADG task\kdag4\ds1_train.csv')
test2 = pd.read_csv('C:\KADG task\kdag4\ds2_test.csv')
data2 = pd.read_csv('C:\KADG task\kdag4\ds2_train.csv')

inputs = np.array(list(zip(data1['x_1'], data1['x_2'])))
ys = np.array(list(zip(data1['y'])))

# calculation of phi
prob1count = 0
prob0count = 0

for i in range(len(ys)):
    if ys[i] == 1:
        prob1count += 1
    else:
        prob0count += 1
phi = prob1count / len(ys)
print(phi)

mu0 = np.zeros(2, dtype=float)
mu1 = np.zeros(2, dtype=float)
for i in range(len(ys)):
    if ys[i] == 0:
        mu0 += inputs[i].T
    else:
        mu1 += inputs[i].T
mu0 /= len(ys)
mu1 /= len(ys)
print(mu0)
print(mu1)

mu0.reshape(2, 1)
mu1.reshape(2, 1)

sigma = np.zeros([2, 2])
for i in range(len(ys)):
    if ys[i] == 1:
        sigma += np.dot(np.transpose(inputs[i].reshape(1, 2)) - mu1.reshape(2,1), (np.transpose((inputs[i].reshape(1, 2)).T - mu1.reshape(2,1))))
    else:
        sigma += np.dot(np.transpose(inputs[i].reshape(1, 2)) - mu0.reshape(2,1), (np.transpose((inputs[i].reshape(1, 2)).T - mu0.reshape(2,1))))
sigma /= len(ys)
print(sigma)
'''sigma = np.zeros((2, 2))
for i in range(len(ys)):
    if ys[i] == 1:
        sigma += np.dot((inputs[i].reshape(1, 2)).T - mu1, ((inputs[i].reshape(1, 2)).T - mu1).T)
    else:
        sigma += np.dot((inputs[i].reshape(1, 2)).T - mu0, ((inputs[i].reshape(1, 2)).T - mu0).T)

sigma /= len(ys)
print(sigma)'''

theta = np.dot(np.linalg.inv(sigma), (mu1 - mu0))
print(theta)
theta0 = np.log(phi / (1 - phi)) + ((np.dot(np.dot(np.transpose(mu1), np.linalg.inv(sigma)), mu0)) / 2) - (
            (np.dot(np.dot(np.transpose(mu1), np.linalg.inv(sigma)), mu1)) / 2)
print(theta0)

testinputs = np.array(list(zip(test1['x_1'], test1['x_2'])))
testys = np.array(list(zip(test1['y'])))

probablities=[]
classify=[]
for i in range(len(testys)):
    k=(1/(1+np.exp(-1*(np.dot(np.transpose(theta),testinputs[i])+theta0))))
    probablities.append(k)

for i in range(len(testys)):
    if probablities[i] >= 0.5 :
        classify.append(1)
    else:
        classify.append(0)

count=0
for i in range(len(testys)):
    if classify[i]==testys[i]:
        count+=1

print(classify)
accuracy=(count/len(testys))*100
print(accuracy)



'''xprob1 = np.zeros([1,2])
xprob0 = np.zeros([1,2])
for i in range(len(ys)):
    if ys[i] == 1:
        prob1count += 1
        xprob1[0][0] += inputs[i][0]
        xprob1[0][1] += inputs[i][1]
    else:
        prob0count += 1
        xprob0[0][0] += inputs[i][0]
        xprob0[0][1] += inputs[i][1]
phi=prob1count/len(ys)
print(phi)
munot=np.zeros([1,2])
muone=np.zeros([1,2])
for i in range(2):
    munot[0][i]=xprob0[0][i]/prob0count
    muone[0][i]=xprob1[0][i]/prob1count
print(munot)
print(muone)
sigsum=np.zeros([2,2])
munot.reshape(2,1)
muone.reshape(2,1)
print(sigsum)
for i in range(len(ys)):
    if ys[i]==0:
        sigsum+=np.dot((np.subtract(inputs[i].reshape(1,2),munot)), (np.transpose(np.subtract(inputs[i].reshape(1,2),munot))))
    if ys[i]==1:
        sigsum+=np.dot((np.subtract(inputs[i].reshape(1,2),muone)), (np.transpose(np.subtract(inputs[i].reshape(1,2),muone))))
print(sigsum)
sigma = sigsum/len(ys)
''''''sigma=np.zeros([2,2])
for i in range(2):
    for j in range(2):
        sigma[i][j]=sigsum[i][j]/len(ys)
print(sigma)'''

'''theta=np.linalg.inv(sigma).dot(muone-munot)
thetanot= np.log(phi/(1-phi)) + ((np.dot(np.dot(np.transpose(munot),np.linalg.inv(sigma)),munot))/2) - ((np.dot(np.dot(np.transpose(muone),np.linalg.inv(sigma)),muone))/2)

testinputs = np.array(list(zip(test1['x_1'], test1['x_2'])))
testys = np.array(list(zip(test1['y'])))

probablities=[]
classify=np.zeros(len(testys))
for i in range(len(testys)):
    k=(1/(1+np.exp(-1*(np.dot(np.transpose(theta),testinputs[i])+thetanot))))
    probablities.append(k)

for i in range(len(testys)):
    if probablities[i] >= 0.5 :
        classify[i]= 1

count=0
for i in range(len(testys)):
    if classify[i]==testys[i]:
        count+=1

print(classify)
accuracy=(count/len(testys))*100
print(accuracy)
'''
