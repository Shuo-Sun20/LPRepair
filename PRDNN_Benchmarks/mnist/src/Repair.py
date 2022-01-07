from sys import argv
import gurobipy
import numpy as np
from pysyrenn import Network
import random
from sklearn.decomposition import PCA
import time

n_lines = 50
nodeRate = 0.98
epsilon = 0.05
outLen = 10
lb = -5.0

def get_corrupted(split, max_count, only_correct_on=None, corruption="fog"):
        """Returns the desired dataset."""
        random.seed(24)
        np.random.seed(24)

        all_images = [
            np
            .load(f"../mnist_data/{corruption}/{split}_images.npy")
            .reshape((-1, 28 * 28))
            for corruption in ("identity", corruption)
        ]
        labels = np.load(f"../mnist_data/identity/{split}_labels.npy")

        indices = list(range(len(labels)))
        random.shuffle(indices)
        labels = labels[indices]
        all_images = [images[indices] / 255. for images in all_images]

        if only_correct_on is not None:
            outputs = only_correct_on.compute(all_images[0])
            outputs = np.argmax(outputs, axis=1)

            correctly_labelled = (outputs == labels)

            all_images = [images[correctly_labelled] for images in all_images]
            labels = labels[correctly_labelled]

        lines = list(zip(*all_images))
        if max_count is not None:
            lines = lines[:max_count]
            labels = labels[:max_count]
        return lines, labels
	
def sample_like_syrenn(train_syrenn, train_labels):
        points, labels = [], []
        for line, label in zip(train_syrenn, train_labels):
            start, end = line[0], line[-1]
            points.extend([start, end])
            # We always want to include the start/end
            alphas = np.random.uniform(low=0.0, high=1.0, size=(100))
            interpolated = start + np.outer(alphas, end - start)
            points.extend(interpolated)
            labels.extend(label for _ in range(len(interpolated) + 2))
        return points, labels

#load network
network = Network.from_file("model.eran" )
network = Network(network.layers[:-1])
	
#load negSamples
lines,labels = get_corrupted("train",n_lines,only_correct_on=network, corruption="fog")
points,labels = sample_like_syrenn(lines,labels)
print(np.array(points).shape)
	
#load posSamples
poslines,poslabels = get_corrupted("train",None,only_correct_on=network, corruption="fog")


#load neg examples
negSamples=np.array(points)
negLabels = labels
negl = len(negSamples)
#load train examples
posSamples = np.array([line[0] for line in poslines])
posLabels = poslabels
posl = len(posSamples)
print(negSamples.shape,posSamples.shape)
#concate the two datasets
samples = np.concatenate((posSamples,negSamples))
labels = np.concatenate((posLabels,negLabels))
print(f"{samples.shape},{labels.shape}")

#modify the network
newnetwork = network
network = Network(network.layers[:-1])
start = time.time()
#PCA operation
innerData = network.compute(samples)
originLen = innerData.shape[1]
pca = PCA(nodeRate)
m = np.mean(innerData,axis=0).reshape(1,-1)
innerData = pca.fit_transform(innerData)
nodeLen = innerData.shape[1]
#split Hard and Soft Examples
pred = np.argmax(newnetwork.compute(samples),axis=1)
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

#Build Model
print("Building Model\n")
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

b = np.full((len(posParam),1),lb)
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
OA = negLen*posParam.sum(axis=0)+posLen*negParam.sum(axis=0)
#OA = posParam.sum(axis=0)+negParam.sum(axis=0)
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

fout = open(argv[1],"w")
for i in range(outLen):
	for j in range(originLen):
		fout.write(f"{param[j][i]},")
	fout.write("\n")
for i in range(outLen):
	fout.write(f"{b[0][i]}\n")
fout.close()
