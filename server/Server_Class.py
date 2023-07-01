import copy
from collections import OrderedDict

import torch
import numpy as np
from numpy import random

class Server():

    def __init__(self, args, model):
        self.clients_list = np.arange(args.num_clients)
        self.args = args
        self.global_model = copy.deepcopy(model)

    def sample_clients(self):
        """
        Return: array of integers, which corresponds to the indices of sampled deviecs
        """
        sampling_set = np.random.choice(self.args.num_clients, self.args.schedulingsize, replace = False)

        return sampling_set
    
    def broadcast(self, Clients_list, Clients_list_idx = None):
        """
        Input: a list of Client class
        Flow: Set the current global model to sampled clients
        """
        for client_idx in Clients_list_idx:
            with torch.no_grad():
                Clients_list[client_idx].model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))

    def aggregation(self, Clients_list, sampling_set):
        """
        Input: sampling_set: array of integers, which corresponds to the indices of sampled devices and a list of Client class
        Flow: aggregate the updated threholds in the sampling set
        """
        #You can change the weights of clients arbitrarily 
        #For simplicy, we use 1/args.schedulingsize here
    

        weight_dict = OrderedDict()

        weight_difference_dict = OrderedDict()
        for i, client in enumerate(sampling_set):
            local_difference = Clients_list[client].model_difference
            if i == 0:
                for key in local_difference.keys():
                    weight_difference_dict[key] = local_difference[key] * 1/self.args.schedulingsize
            else:
                for key in local_difference.keys():
                    weight_difference_dict[key] += local_difference[key] *1/self.args.schedulingsize

        for key in weight_difference_dict.keys():
            weight_dict[key] = self.global_model.state_dict()[key] + weight_difference_dict[key]
        self.global_model.load_state_dict(weight_dict)

