import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
from collections import OrderedDict
import copy

path = os.getcwd() #current path
sys.path.append(os.path.abspath(os.path.join(path, os.pardir))) #import the parent directory

from model import quantization


class Client():
    def __init__(self, args, model, loss, client_id, tr_loader, te_loader, device, scheduler = None):
        self.args = args
        self.model = model
        self.loss = loss
        self.scheduler = scheduler
        self.client_id = client_id
        self.tr_loader = tr_loader
        self.te_loader = te_loader
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr= self.args.learning_rate, 
                            momentum=self.args.momentum, weight_decay=self.args.weight_decay)
        self.model_difference = OrderedDict()
    
    def local_training(self, comm_rounds):
        initial = copy.deepcopy(self.model)
        for epoch in range(1, self.args.local_epoch+1):
            for data, label in self.tr_loader:
                data.to(self.device), label.to(self.device)
                self.model.train()
                output = self.model(data)
                loss_val = self.loss(output, label)

                self.optimizer.zero_grad()
                loss_val.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()
        for name in self.model.state_dict():
            foo = self.model.state_dict()[name] - initial.state_dict()[name]
            quantized_foo = self.uniform_quantize(foo)
            self.model_difference[name] = quantized_foo
            
    def local_test(self):

        total_acc = 0.0
        num = 0
        self.model.eval()
        std_loss = 0. 
        iteration = 0.
        with torch.no_grad():
            for data, label in self.te_loader:
                data, label = data.to(self.device), label.to(self.device)
                output = self.model(data)
                pred = torch.max(output, dim=1)[1]
                te_acc = (pred.cpu().numpy()== label.cpu().numpy()).astype(np.float32).sum()

                total_acc += te_acc
                num += output.shape[0]

                std_loss += self.loss(output, label)
                iteration += 1
        std_acc = total_acc/num*100.
        std_loss /= iteration

        
        return std_acc, std_loss

    def uniform_quantize(self, x):
        if self.args.m_bit == 32:
            return x
        elif self.args.m_bit == 1:
            return torch.sign(x)
        else:
            m = float(2 ** (self.args.m_bit - 1))
            out = torch.round(x * m) / m
            return out