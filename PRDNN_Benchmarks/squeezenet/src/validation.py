from keras.models import Model
from keras.models import load_model
import numpy as np
from sys import argv
import pandas as pd
from ImageHelper import read_imagenet_images
from pysyrenn import FullyConnectedLayer, NormalizeLayer
from pysyrenn import Network
#args : 1.patch 2.network 3. data

fin = open("../results/squeezenet.txt")
W = []
for i in range(9):
	line = fin.readline()
	eles = line.strip().split(',')[:-1]
	eles = list(map(float,eles))
	W.append(eles)
W=np.array(W)
B = []
for i in range(9):
	line=fin.readline()
	eles = float(line.strip())
	B.append(eles)
B=np.array(B)

#load network/ half
network = Network.from_file("../squeezenet1.1.onnx")
normalize = NormalizeLayer(
            means=np.array([0.485, 0.456, 0.406]),
            standard_deviations=np.array([0.229, 0.224, 0.225]))
newModel = Network([normalize] + network.layers)

#load data
data,labels= read_imagenet_images("../data/vimage",n_labels=9)
len1 = len(data)

#modify the network
sorted_labels = sorted(set(labels))
labels = list(map(sorted_labels.index, labels))
final_weights = np.zeros((1000, len(sorted_labels)))
final_biases = np.zeros(len(sorted_labels))
for new_label, old_label in enumerate(sorted_labels):
	final_weights[old_label, new_label] = 1.
final_layer = FullyConnectedLayer(final_weights, final_biases)
model = Network(newModel.layers + [final_layer])


#compute
inner = newModel.compute(data)
oo = model.compute(data)
oi = np.argmax(oo,axis=1)
ov = np.count_nonzero(np.equal(oi,labels)) / len(labels)
print("OriginValidation:",ov)
result = np.dot(inner,W.transpose())+B.transpose()
out = np.argmax(result,axis=1)
rv = np.count_nonzero(np.equal(out,labels)) / len(labels)
print("RepairValidation:", rv)

#load data
data,labels= read_imagenet_images("../data/imagenet-a",n_labels=9)
len1 = len(data)

#modify the network
sorted_labels = sorted(set(labels))
labels = list(map(sorted_labels.index, labels))
final_weights = np.zeros((1000, len(sorted_labels)))
final_biases = np.zeros(len(sorted_labels))
for new_label, old_label in enumerate(sorted_labels):
        final_weights[old_label, new_label] = 1.
final_layer = FullyConnectedLayer(final_weights, final_biases)
model = Network(newModel.layers + [final_layer])

#compute
inner = newModel.compute(data)
oo = model.compute(data)
oi = np.argmax(oo,axis=1)
ov = np.count_nonzero(np.equal(oi,labels)) / len(labels)
print("OriginValidation:",ov)
result = np.dot(inner,W.transpose())+B.transpose()
out = np.argmax(result,axis=1)
rv = np.count_nonzero(np.equal(out,labels)) / len(labels)
print("RepairValidation:", rv)
