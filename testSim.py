# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:37:28 2020

@author: anmol
"""

import numpy as np
from NetworkClasses import LSMNetwork

steps = 1000
ch = 780
dims = (10,10,10)
frac_inhibitory = 0.2
w_mat = np.array([[10, 10],[-5, -5]])
fanout = 9
in_spikes = np.random.choice([0, 20], size=(ch,steps), replace=True, p = [0.8, 0.2])
print("generate Network")
reservoir_network = LSMNetwork(dims, frac_inhibitory, w_mat, fanout, steps, ch)
print("initialize input")
reservoir_network.add_input(in_spikes)

total_initial_activity = 0
for i in range(len(reservoir_network.action_buffer)):
    total_initial_activity = total_initial_activity + len(reservoir_network.action_buffer[i])

print("simulate")
rate_coding = reservoir_network.simulate()

total_final_activity = 0
for i in range(len(reservoir_network.action_buffer)):
    total_final_activity = total_final_activity + len(reservoir_network.action_buffer[i])

#reservoir_network.add_input(in_spikes)
#
#total_initial_activity2 = 0
#for i in range(len(reservoir_network.action_buffer)):
#    total_initial_activity2 = total_initial_activity2 + len(reservoir_network.action_buffer[i])