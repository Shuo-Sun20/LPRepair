import numpy as np
from sys import argv
import pandas as pd
from pysyrenn import Network
import random
#args : 1.patch 2.network 3. data
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
	

fin = open(f"{argv[1]}")
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

#load network/ half
network = Network.from_file("../scripts/model.eran")
newModel = Network(network.layers[:-2])

#load data
test_lines, label = get_corrupted("test", None)
test_images = list(map(np.array, zip(*test_lines)))

#modify the network
model = network

#compute
inner = newModel.compute(test_images[0])
oo = model.compute(test_images[0])
oi = np.argmax(oo,axis=1)
ov = np.count_nonzero(np.equal(oi,label)) / len(label)
print("OriginValidation:",ov)
result = np.dot(inner,W.transpose())+B.transpose()
out = np.argmax(result,axis=1)
rv = np.count_nonzero(np.equal(out,label)) / len(label)
print("RepairValidation:", rv)

#compute
inner = newModel.compute(test_images[1])
oo = model.compute(test_images[1])
oi = np.argmax(oo,axis=1)
ov = np.count_nonzero(np.equal(oi,label)) / len(label)
print("OriginValidation:",ov)
result = np.dot(inner,W.transpose())+B.transpose()
out = np.argmax(result,axis=1)
rv = np.count_nonzero(np.equal(out,label)) / len(label)
print("RepairValidation:", rv)
