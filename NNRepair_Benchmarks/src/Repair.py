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

nodeRate = 0.98
epsilon = 0.00001
maxSample = 6000

def load_data(shape,dirName,part='train'):
	data = pd.read_csv(f"../data/{dirName}/{part}_samples.csv",header=None)
	result = []
	for i in range(len(data)):
		sample = data.loc[i].values.reshape(shape)
		result.append(sample)
	samples = np.array(result,dtype=np.float32)
	
	labels = pd.read_csv(f"../data/{dirName}/{part}_labels.csv",header=None)
	n = []
	for i in range(len(labels)):
		n.append(labels.loc[i].iloc[0])
	labels = np.array(n)
	return samples,labels

model = load_model(argv[1])
newModel = Model(inputs=model.input, outputs=model.layers[-3].output)
shape = model.layers[0].input.shape[1:]
negSamples,negLabels = load_data(shape,argv[2])
posSamples,posLabels = load_data(shape,f"{argv[2].split('_')[0]}_normal")
posl = len(posSamples)
negl = len(negSamples)
samples = np.concatenate((posSamples,negSamples))
labels = np.concatenate((posLabels,negLabels))
print(f"{samples.shape},{labels.shape}")
pred = np.argmax(model.predict(samples),axis=1)
innerData = newModel.predict(samples)
originLen = innerData.shape[1]
pca = PCA(nodeRate)
m = np.mean(innerData,axis=0).reshape(1,-1)
innerData = pca.fit_transform(innerData)


posData=[]
posLabels=[]
negData=[]
negLabels=[]
for i in range(posl):
	if pred[i] == labels[i]:
		posData.append(innerData[i])
		posLabels.append(labels[i])
for i in range(posl,len(pred)):
	if pred[i] !=labels[i]:
		negData.append(innerData[i])
		negLabels.append(labels[i])

posLen = len(posLabels)
negLen = len(negLabels)

"""
if posLen > negLen:
	posLen = negLen
if negLen > posLen:
	negLen = posLen
"""
"""
if (posLen > maxSample):
	posLen = maxSample
if (negLen > maxSample):
	negLen = maxSample*0.8
"""
nodeLen = innerData.shape[1]
outLen = model.layers[-1].output.shape[1]
print(f"\n\n\n\n\n\n\n\n\n\n\nposLen : {posLen}\nnegLen : {negLen}\noriginLen:{originLen}\nnodeLen : {nodeLen}\noutLen : {outLen}\n\n\n\n")


#Build Model
print("Building Model\n")
start = time.time()
MODEL = gurobipy.Model()
#variables
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

b = np.full((len(posParam),1),epsilon)
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

b = np.full((len(negParam),1),-5)
MODEL.addMConstr( negParam, x, '>', b)

#OA = negLen*posParam.sum(axis=0)+posLen*negParam.sum(axis=0)
OA = posParam.sum(axis=0)+negParam.sum(axis=0)
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

fout = open(argv[4],"w")
for i in range(outLen):
	for j in range(originLen):
		fout.write(f"{param[j][i]},")
	fout.write("\n")
for i in range(outLen):
	fout.write(f"{b[0][i]}\n")
fout.close()
	
	
	
	
	
	
	
	
	
	
	
	
