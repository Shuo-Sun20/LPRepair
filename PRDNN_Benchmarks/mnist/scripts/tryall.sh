#!/bin/sh
python ../src/Repair.py ../results/mnist.txt >> ../results/mnist.proc
python ../src/Validation.py ../results/mnist.txt >> ../results/mnist.res







