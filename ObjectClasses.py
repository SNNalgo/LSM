# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:47:19 2020

@author: anmol
"""
import numpy as np

class Spike:
    
    def __init__(self, spike_values, receiver_nodes, receive_times):
        self.receiver_nodes = receiver_nodes
        self.receive_times = receive_times
        self.spike_values = spike_values

class Neuron:
    V_rest = 0
    tau = 20
    t_ref = 10
    V_th = 50
    
    def __init__(self, sid, out_conn, out_weights, fanout, spike_propagation_time=None):
        self.sid = sid
        self.Vp = self.V_rest
        self.out_conn = out_conn
        self.fanout = fanout
        self.last_spike = -15
        self.last_receive_spike = 0
        self.out_weights = out_weights
        if spike_propagation_time == None:
            self.spike_propagation_time = [10 - np.random.randint(5) for i in range(fanout)] #time taken for spike to propaate 
        else:
            self.spike_propagation_time = spike_propagation_time
    
    def send_spike(self, time):
        send_times = [time + self.spike_propagation_time[i] for i in range(len(self.spike_propagation_time))]
        spike = Spike(self.out_weights, self.out_conn, send_times)
        self.last_spike = time
        self.Vp = self.V_rest
        return spike
    
    def receive_spike(self, time, value):
        spiked = False
        if (time - self.last_spike) > self.t_ref:
            V = self.Vp * np.exp(-(time-self.last_receive_spike)/self.tau)
            V = V + value
            self.last_receive_spike = time
            if V>self.V_th:
                spiked = True
                spike = self.send_spike(time)
            else:
                self.Vp = V
        if spiked:
            return spike
        else:
            return None