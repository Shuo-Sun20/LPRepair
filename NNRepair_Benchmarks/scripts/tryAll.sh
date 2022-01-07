#!/bin/sh
for network in cifar10_normal cifar10_adv cifar10_poisoned mnist_normal mnist_poisoned normmnist
do
python ../src/Repair.py ../networks/${network}.h5  ${network} train ../results/${network}.txt   >> ../results/${network}.proc
python ../src/Validation.py ../networks/${network}.h5 ${network} ../results/${network}.txt   >> ../results/${network}.res
done
