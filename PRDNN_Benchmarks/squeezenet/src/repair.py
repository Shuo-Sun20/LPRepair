import gurobipy
import argparse
import time
from math import exp
import numpy as np
from sklearn.decomposition import PCA
from sys import argv
from keras.models import load_model
from keras.models import Model
import pandas as pd
from ImageHelper import read_imagenet_images
from pysyrenn import Network
import random
from pysyrenn import FullyConnectedLayer, NormalizeLayer
import json
import resource
import psutil


def limit_memory(maxsize):
        soft,hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS,(maxsize,hard))

limit_memory(1024*1024*1024*64)



nodeRate = 0.98
epsilon = 0.05
maxSample = 752

#load network
network = Network.from_file("../squeezenet1.1.onnx")
normalize = NormalizeLayer(
            means=np.array([0.485, 0.456, 0.406]),
            standard_deviations=np.array([0.229, 0.224, 0.225]))
network = Network([normalize] + network.layers)
	
#load neg examples
negSamples,negLabels = read_imagenet_images("../data/imagenet-a",n_labels=9)
#load train examples
#posSamples,posLabels = read_imagenet_images("../data/imgs",n_labels=9,maxNum=200)
#posl = len(posSamples)


#concate the two datasets
samples = negSamples #np.concatenate((posSamples,negSamples))
labels = negLabels #np.concatenate((posLabels,negLabels))
#negl = 100
print(f"{samples.shape},{labels.shape}")

#modify the network
sorted_labels = sorted(set(labels))
outLen = 9
labels = np.array(list(map(sorted_labels.index, labels)))
final_weights = np.zeros((1000, 9))
final_biases = np.zeros(9)
for new_label, old_label in enumerate(sorted_labels):
	final_weights[old_label, new_label] = 1.
final_layer = FullyConnectedLayer(final_weights, final_biases)
newnetwork = Network(network.layers + [final_layer])
#PCA operation
samples = samples[:maxSample]
labels = labels[:maxSample]
negl = maxSample
innerData = network.compute(samples)
posinnerData = np.load("../data/preimg_exp.npy")
poslabels = np.load("../data/preimg_label.npy")
posl = len(poslabels)
innerData = np.concatenate((posinnerData,innerData))
labels = np.concatenate((poslabels,labels))
print(f"{innerData.shape},{posinnerData.shape},{poslabels.shape},{labels.shape}")
originLen = innerData.shape[1]
pca = PCA(nodeRate)
m = np.mean(innerData,axis=0).reshape(1,-1)
innerData = pca.fit_transform(innerData)

#split Hard and Soft Examples
pred = np.argmax(newnetwork.compute(samples),axis=1)
pospred = np.load("../data/preimg_pred.npy")
print(pred.shape,pospred.shape)
pred = np.concatenate((pospred,pred))
print(pred.shape)
posData=[]
posLabels=[]
negData=[]
negLabels=[]
for i in range(posl):
	if pred[i] == labels[i]:
		posData.append(innerData[i])
		posLabels.append(labels[i])
for i in range(posl,len(pred)):
		negData.append(innerData[i])
		negLabels.append(labels[i])
posLen = len(posLabels)
negLen = len(negLabels)

#Build Model
print("Building Model\n")
start = time.time()
MODEL = gurobipy.Model()
#variables
nodeLen = innerData.shape[1]
xLen = outLen*nodeLen+outLen
x = MODEL.addMVar(xLen,lb= -1,ub = 1)
MODEL.update()

#Constr
#Part1 OriginData HardConstr
posParam = []
for ind in range(posLen):
	cate = posLabels[ind]
	eles = posData[ind]
	for outInd in range(outLen):
		if outInd == cate:
			continue
		oneConstr = np.zeros(xLen)
		for i in range(nodeLen):
			oneConstr[cate*nodeLen+i] = eles[i]
			oneConstr[outInd*nodeLen+i] = -eles[i]
		oneConstr[outLen*nodeLen+cate] = 1.0
		oneConstr[outLen*nodeLen+outInd] = -1.0
		posParam.append(oneConstr)
posParam=np.array(posParam)

b = np.full((len(posParam),1),-10)
MODEL.addMConstr( posParam, x, '>', b)

negParam = []
for ind in range(negLen):
	cate = negLabels[ind]
	eles = negData[ind]
	for outInd in range(outLen):
		if outInd == cate:
			continue
		oneConstr = np.zeros(xLen)
		for i in range(nodeLen):
			oneConstr[cate*nodeLen+i] = eles[i]
			oneConstr[outInd*nodeLen+i] = -eles[i]
		oneConstr[outLen*nodeLen+cate] = 1.0
		oneConstr[outLen*nodeLen+outInd] = -1.0
		negParam.append(oneConstr)
negParam=np.array(negParam)

b = np.full((len(negParam),1),epsilon)
MODEL.addMConstr( negParam, x, '>', b)

#OA = negLen*posParam.sum(axis=0)+posLen*negParam.sum(axis=0)
OA = posParam.sum(axis=0)+negParam.sum(axis=0)
#OA = negParam.sum(axis=0)
MODEL.setObjective(OA@x,gurobipy.GRB.MAXIMIZE)

buildTime = time.time()-start
print(f"Building time: {buildTime}")
#solve
#MODEL.write("Prob.lp")
print("Solving\n")
MODEL.optimize()	

totalTime=time.time()-start
print(f"Total time: {totalTime}")

W = []
for i in range(nodeLen):
	newW = []
	for k in range(outLen):
		newW.append(x[k*nodeLen+i].x)
	W.append(newW)	
W = np.array(W)
W = W.reshape(W.shape[0],W.shape[1])
print(W.shape)
param = np.dot(pca.components_.transpose(),W)
b = []
for i in range(outLen):
	b.append(x[outLen*nodeLen+i].x)
b = np.array(b).reshape(1,-1)
nb = np.dot(m,param)
print (W.shape,b.shape,m.shape,param.shape,nb.shape)
b = np.subtract(b,nb)
print(param,b)
print(param[0][1],b[0][1])

fout = open("../results/squeezenet.txt","w")
for i in range(outLen):
	for j in range(originLen):
		fout.write(f"{param[j][i]},")
	fout.write("\n")
for i in range(outLen):
	fout.write(f"{b[0][i]}\n")
fout.close()






