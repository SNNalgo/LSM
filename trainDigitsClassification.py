# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:21:50 2020

@author: anmol
"""

import numpy as np
from NetworkClasses import LSMNetwork
from sklearn.datasets import load_digits
from sklearn import linear_model

digits = load_digits()
train_data = digits.data[:1600,:]/16.0
test_data = digits.data[1600:,:]/16.0

train_labels = digits.target[:1600]
test_labels = digits.target[1600:]

steps = 1000
ch = 64
dims = (5,5,5)
frac_inhibitory = 0.2
w_mat = 4*np.array([[3, 6],[-2, -2]])
fanout = 7
reservoir_network = LSMNetwork(dims, frac_inhibitory, w_mat, fanout, steps, ch, ignore_frac=0.5)
train_in_spikes = np.zeros((ch, steps))
X = np.zeros((train_labels.shape[0],reservoir_network.n_nodes))
for i in range(train_labels.shape[0]):
    print(i)
    for j in range(ch):
        spike_interval = np.int32(steps/(train_data[i,j]*50 + 0.0001))
        jitter = np.random.randint(20)
        spikes = np.zeros(steps-20)
        spikes[jitter::spike_interval] = 60
        train_in_spikes[j,:-20] = spikes
    reservoir_network.add_input(train_in_spikes)
    rate_coding = reservoir_network.simulate()
    #X[i,:] = rate_coding
    X[i,:] = rate_coding/(np.max(rate_coding) + 0.0001)

#maxX = (np.max(X) + 0.0001)
#X = X/maxX
print("training linear model")
clf = linear_model.SGDClassifier(max_iter=100000, tol=1e-3)
clf.fit(X, train_labels)

X_test = np.zeros((test_labels.shape[0],reservoir_network.n_nodes))
test_in_spikes = np.zeros((ch, steps))
initial_activities = np.zeros(test_labels.shape[0])
extra_activities = np.zeros(test_labels.shape[0])
for i in range(test_labels.shape[0]):
    print(i)
    for j in range(ch):
        spike_interval = np.int32(steps/(test_data[i,j]*50 + 0.0001))
        jitter = np.random.randint(20)
        spikes = np.zeros(steps-20)
        spikes[jitter::spike_interval] = 60
        test_in_spikes[j,:-20] = spikes
    reservoir_network.add_input(test_in_spikes)
    rate_coding = reservoir_network.simulate()
    for k in range(len(reservoir_network.action_buffer)):
        for l in range(len(reservoir_network.action_buffer[k])):
            if reservoir_network.action_buffer[k][l][0] != 60:
                extra_activities[i] += 1
            else:
                initial_activities[i] += 1
    #X_test[i,:] = rate_coding
    X_test[i,:] = rate_coding/(np.max(rate_coding) + 0.0001)

#X_test = X_test/maxX
score = clf.score(X_test, test_labels)
print("test score = " + str(score))
print("train score = " + str(clf.score(X, train_labels)))
