import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
from torchvision.transforms import ToTensor
from time import time
import numpy as np
import copy
from model import CNN_model
from server import Server_Class
from Split_Data import Non_iid_split
from client import Client_Class
from utils import*

class Simulator():
    def __init__(self, args, logger, local_tr_data_loaders, local_te_data_loaders, device):
        self.args = args
        self.logger = logger
        self.Clients_list = None
        self.Clients_list = None
        self.Server = None
        self.local_tr_data_loaders = local_tr_data_loaders
        self.local_te_data_loaders = local_te_data_loaders
        self.device = device


    def initialization(self, model):

        loss = nn.CrossEntropyLoss()

        self.Server = Server_Class.Server(self.args, model)
        
        self.Clients_list = [Client_Class.Client(self.args, copy.deepcopy(self.Server.global_model), loss, 
                                        client_id, tr_loader, te_loader, self.device, scheduler=None)
                                        for (client_id, (tr_loader, te_loader)) in enumerate(zip(self.local_tr_data_loaders, self.local_te_data_loaders))]

    def FedAvg(self):

        best_acc = 0
        acc_history = []

        for rounds in np.arange(self.args.comm_rounds):
            begin_time = time()
            avg_acc =[]
            avg_loss =[]
            self.logger.info("-"*30 + "Epoch start" + "-"*30)

            sampled_clients = self.Server.sample_clients()

            self.Server.broadcast(self.Clients_list, sampled_clients)
            for client_idx in sampled_clients:
                acc, loss = self.Clients_list[client_idx].local_test()
                avg_acc.append(acc), avg_loss.append(loss)

            for client_idx in sampled_clients:
                self.Clients_list[client_idx].local_training(rounds)        


            self.Server.aggregation(self.Clients_list, sampled_clients)


            avg_acc_round = np.mean(avg_acc)

            acc_history.append(avg_acc_round) #save the current average accuracy to the history

            self.logger.info('round: %d, avg_acc: %.3f, spent: %.2f' %(rounds, avg_acc_round,
                                                                                                         time()-begin_time))

            cur_acc = avg_acc_round
            if cur_acc > best_acc:
                best_acc =cur_acc

        #####Check final accuracy
        self.Server.broadcast(self.Clients_list, range(0, self.args.num_clients))
        final_acc =[]
        for client_idx, client in enumerate(self.Clients_list):
            acc, loss = client.local_test()
            final_acc.append(acc)
            self.logger.info('client_id: %d , final acc: %.3f' %(
                             client_idx, loss))
        final_avg_acc = np.mean(final_acc)

        self.logger.info(">>>>> Training process finish")
        self.logger.info("Best test accuracy {:.4f}".format(best_acc))  
        self.logger.info("Final test accuracy {:.4f}".format(final_avg_acc))
        self.logger.info(">>>>> Accuracy history during training")
        self.logger.info(acc_history)

