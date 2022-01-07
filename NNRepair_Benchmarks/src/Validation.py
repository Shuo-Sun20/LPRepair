from keras.models import Model
from keras.models import load_model
import numpy as np
from sys import argv
import pandas as pd

#args : 1.patch 2.network 3. data

def load_data(shape,dirName,part='validation'):
	data = pd.read_csv(f"../data/{dirName}/{part}_samples.csv",header=None)
	result = []
	for i in range(len(data)-1):
		sample = data.loc[i+1].values.reshape(shape)
		result.append(sample)
	samples = np.array(result,dtype=np.float32)
	
	labels = pd.read_csv(f"../data/{dirName}/{part}_labels.csv")
	n = []
	for i in range(len(labels)):
		n.append(labels.loc[i].iloc[0])
	labels = np.array(n)
	return samples,labels

fin = open(argv[3])
W = []
for i in range(10):
	line = fin.readline()
	eles = line.strip().split(',')[:-1]
	eles = list(map(float,eles))
	W.append(eles)
W=np.array(W)
B = []
for i in range(10):
	line=fin.readline()
	eles = float(line.strip())
	B.append(eles)
B=np.array(B)

model = load_model(argv[1])
shape = model.layers[0].input.shape[1:]


data,label = load_data(shape,argv[2])
len1 = len(data)
newModel = Model(inputs=model.input, outputs=model.layers[-3].output)
inner = newModel.predict(data)
oo = model.predict(data)
oi = np.argmax(oo,axis=1)
ov = np.count_nonzero(np.equal(oi,label)) / len(label)
print("OriginValidation:",ov)
result = np.dot(inner,W.transpose())+B.transpose()
out = np.argmax(result,axis=1)
rv = np.count_nonzero(np.equal(out,label)) / len(label)
print("RepairValidation:", rv)

data,label = load_data(shape,f"{argv[2].split('_')[0]}_normal")
len2 = len(data)
inner = newModel.predict(data)
oo = model.predict(data)
oi = np.argmax(oo,axis=1)
ot = np.count_nonzero(np.equal(oi, label)) / len(label)
print("OriginTrain:",np.count_nonzero(np.equal(oi, label)) / len(label))
result = np.dot(inner,W.transpose())+B.transpose()
out = np.argmax(result,axis=1)
rt = np.count_nonzero(np.equal(out, label)) / len(label)
print("RepairTrain:",np.count_nonzero(np.equal(out, label)) / len(label))

print("OriginOverall:",(ot*len2+ov*len1)/(len1+len2))
