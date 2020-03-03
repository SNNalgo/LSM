# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:41:16 2020

@author: anmol
"""
import numpy as np
from ObjectClasses import Neuron, Spike
from ReservoirDefinitions import create_random_reservoir

class LSMNetwork:
    def __init__(self, dims, frac_inhibitory, w_matrix, fanout, simulation_steps, num_in_ch):
        #simulation_steps : total number of simulation steps to simulate time T in steps of dt = T/dt
        self.dims = dims
        self.n_nodes = dims[0]*dims[1]*dims[2]
        self.num_in_ch = num_in_ch
        if num_in_ch<=self.n_nodes:
            mapped_nodes = np.random.choice(self.n_nodes, size=num_in_ch, replace=False)
        else:
            mapped_nodes = np.random.choice(self.n_nodes, size=num_in_ch, replace=True)
        self.mapped_nodes = mapped_nodes
        self.frac_inibitory = frac_inhibitory
        self.w_matrix = w_matrix
        self.fanout = fanout
        adj_mat, all_connections, all_weights = create_random_reservoir(dims, frac_inhibitory, w_matrix, fanout)
        #self.adj_mat = adj_mat
        self.all_connections = all_connections
        self.all_weights = all_weights
        self.neuronList = [Neuron(i, all_connections[i], all_weights[i], fanout) for i in range(len(all_connections))]
        self.simulation_steps = simulation_steps
        self.current_time_step = 0
        self.action_buffer = []
        for i in range(simulation_steps):
            self.action_buffer.append([])
        return
    
    def add_input(self, input_spike_train):
        #input_spike_train : num_channels x simulation_steps matrix of all channels of the input spike train
        for i in range(len(self.neuronList)):
            self.neuronList[i] = Neuron(i, self.all_connections[i], self.all_weights[i], self.fanout)
        for t_step in range(input_spike_train.shape[1]):
            self.action_buffer[t_step] = []
            for ch in range(self.num_in_ch):
                if input_spike_train[ch,t_step] > 0:
                    self.action_buffer[t_step].append((input_spike_train[ch,t_step], self.mapped_nodes[ch]))
        return
    
    def simulate(self):
        rate_coding = np.zeros(self.n_nodes)
        frac = 0.0
        for t_step in range(self.simulation_steps):
            #print(t_step)
            if len(self.action_buffer[t_step])>0:
                for action in self.action_buffer[t_step]:
                    spike_val = action[0]
                    target_node = action[1]
                    spike_produced = self.neuronList[target_node].receive_spike(t_step, spike_val)
                    if spike_produced != None:
                        if t_step > frac*self.simulation_steps:
                            rate_coding[target_node] += 1
                        receiver_nodes = spike_produced.receiver_nodes
                        spike_values = spike_produced.spike_values
                        receive_times = spike_produced.receive_times
                        for node in range(len(receiver_nodes)):
                            if(receive_times[node]<self.simulation_steps):
                                self.action_buffer[receive_times[node]].append((spike_values[node], receiver_nodes[node]))
        return rate_coding