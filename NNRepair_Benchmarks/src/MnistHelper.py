import numpy as np
import pandas as pd

test_data = ""
test_label = "../../../data/data/test_label.txt"
CAV_val_label = "Datasets/MNIST_Dataset/mnist-data-adversarial/val-data/mnist_adv_val_label.txt"
CAV_val_data = "Datasets/MNIST_Dataset/mnist-data-adversarial/val-data/mnist_val_csv_fgsm_epsilon0.01.txt"
CAV_train_data = "Datasets/MNIST_Dataset/mnist-data-adversarial/data/mnist_test_csv_fgsm_epsilon0.01.txt"
CAV_train_label = "Datasets/MNIST_Dataset/mnist-data-adversarial/data/mnist_test_label_csv.txt"

def load_MNIST_Sample(part = "test"):
	assert (part in ["test","train"])
	f = open(f"../../../data/data/{part}_data.txt")
	line = f.readline()
	result = []
	while(line):
		eles = line.strip().split()
		eles = list(map(float,eles))
		eles = np.array(eles)
		eles = eles.reshape(28,28,1)
		result.append(eles)
		line = f.readline()
	return np.array(result)

def load_MNIST_Label(part = "test"):
	assert (part in ["test","train"])
	f = open(f"../../../data/data/{part}_label.txt")
	line = f.readline()
	result = []
	while(line):
		eles=int(line.strip())
		result.append(eles)
		line = f.readline()
	return np.array(result)

def load_CSV_MNIST_Label(part="val"):
	assert(part in ["val","test"]), "param part wrong in Label"
	
	if part == "val":
		CAV_label =  "Datasets/MNIST_Dataset/mnist-data-adversarial/val-data/mnist_adv_val_label.txt"
	else:
		CAV_label =  "Datasets/MNIST_Dataset/mnist-data-adversarial/data/mnist_test_label_csv.txt"
	data = pd.read_csv(CAV_label)
	result = []
	for i in range(len(data)):
		label = data.loc[i].values
		result  += list(label)
	return np.array(result)

def load_CSV_MNIST_Sample(part="val",epsilon=0.1):
	assert(part in ["val","test"]), "param part wrong in Label"
	assert(epsilon in [0.1,0.01,0.2,0.3,0.05]), "param part wrong in Label"
	
	dirname = 'val-data' if part=="val" else "data"
	CAV_data = f"Datasets/MNIST_Dataset/mnist-data-adversarial/{dirname}/mnist_{part}_csv_fgsm_epsilon{epsilon}.txt"
	data = pd.read_csv(CAV_data)
	result = []
	for i in range(len(data)):
		inputData = data.loc[i].values
		result.append(inputData.reshape(28,28,1))
	return np.array(result)

if __name__ == '__main__':
	print (load_CSV_MNIST_Train_Sample().shape)
	print(load_CSV_MNIST_Train_Label())
	print(load_MNIST_Train_Data().shape)
	print(load_MNIST_Train_Label())