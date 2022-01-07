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

#load network
network = Network.from_file("../squeezenet1.1.onnx")
normalize = NormalizeLayer(
            means=np.array([0.485, 0.456, 0.406]),
            standard_deviations=np.array([0.229, 0.224, 0.225]))
network = Network([normalize] + network.layers)

#load neg examples
negSamples,negLabels = read_imagenet_images("../data/imgs",n_labels=9)
negl = len(negSamples)

#modify the network
labels = negLabels
sorted_labels = sorted(set(labels))
labels = list(map(sorted_labels.index, labels))
final_weights = np.zeros((1000, len(sorted_labels)))
final_biases = np.zeros(len(sorted_labels))
for new_label, old_label in enumerate(sorted_labels):
	final_weights[old_label, new_label] = 1.
final_layer = FullyConnectedLayer(final_weights, final_biases)
newnetwork = Network(network.layers + [final_layer])

start = time.time()
innerData = []
pred = []
for i in range(len(negLabels)):
	if len(innerData) == 0:
		innerData = network.compute(negSamples[i])
		pred = np.argmax(newnetwork.compute(negSamples[i]),axis=1)
	else:
		innerData = np.concatenate((innerData,network.compute(negSamples[i])))
		pred = np.concatenate((pred,np.argmax(newnetwork.compute(negSamples[i]),axis=1)))
print(time.time()-start)


np.save("../data/preimg_exp.npy",innerData)
np.save("../data/preimg_label.npy",labels)
np.save("../data/preimg_pred.npy",pred)

print(np.count_nonzero(np.equal(pred,labels)) / len(labels))
