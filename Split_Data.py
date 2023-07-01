from torch.utils.data import Dataset
import numpy as np
import torch
import random

class Non_iid(Dataset):
    def __init__(self, x, y):
        self.x_data = x.unsqueeze(1).to(torch.float32)
        # self.x_data = x.reshape(x.shape[0], 28, 28, 1)
        self.y_data = y.to(torch.int64)
        self.cuda_available = torch.cuda.is_available()
    
    #Return the number of data
    def __len__(self):
        return len(self.x_data)
    
    #Sampling
    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        if self.cuda_available:
            return x.cuda(), y.cuda()
        else:
            return x, y


def data_stats(non_iid_datasets, num_classes, num_clients):

    client_data_counts = {client:{} for client in range(num_clients)}
    client_total_samples = []
    for client, data in enumerate(non_iid_datasets):
        total_sample = 0
        for label in range(num_classes):
            idx_label = len(np.where(data.y_data == label)[0])
            # client_data_counts[client].append(idx_label/data.__len__() * 100)
            label_sum = np.sum(idx_label)
            client_data_counts[client][label] = label_sum
            total_sample += label_sum
        client_total_samples.append(total_sample)

    return client_data_counts, client_total_samples

def Non_iid_split(num_classes, num_clients, tr_datasets, te_datasets, alpha):
    """
    Input: num_classes, num_clients, datasets, alpha
    Output: Dataset classes of the number of num_clients 
    """
    tr_idx_batch = [[] for _ in range(num_clients)]
    tr_data_index_map = {}
    te_idx_batch = [[] for _ in range(num_clients)]
    te_data_index_map = {}

    #for each calss in the training/test dataset
    for label in range(num_classes):
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients)) #It generates dirichichlet random variable with alpha over num_clients

        tr_idx_label = np.where(tr_datasets.targets == label)[0] #np.where returns corresponding indices and datatype
        np.random.shuffle(tr_idx_label)
        tr_proportions = (np.cumsum(proportions) * len(tr_idx_label)).astype(int)[:-1]

        tr_idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(tr_idx_batch, np.split(tr_idx_label, tr_proportions))]
        
        te_idx_label = np.where(te_datasets.targets == label)[0]
        np.random.shuffle(te_idx_label)
        te_proportions = (np.cumsum(proportions) * len(te_idx_label)).astype(int)[:-1]

        te_idx_batch = [idx_j + idx.tolist() for idx_j, idx in
                         zip(te_idx_batch, np.split(te_idx_label, te_proportions))]
        
    for client in range(num_clients):
        np.random.shuffle(tr_idx_batch[client])
        tr_data_index_map[client] = tr_idx_batch[client]
        te_data_index_map[client] = te_idx_batch[client]

    Non_iid_tr_datasets = []
    Non_iid_te_datasets = []

    for client in range(num_clients):
        tr_x_data = tr_datasets.data[tr_data_index_map[client]]
        tr_y_data = tr_datasets.targets[tr_data_index_map[client]]
        Non_iid_tr_datasets.append(Non_iid(tr_x_data, tr_y_data))

        te_x_data = te_datasets.data[te_data_index_map[client]]
        te_y_data = te_datasets.targets[te_data_index_map[client]]
        Non_iid_te_datasets.append(Non_iid(te_x_data, te_y_data))

    return Non_iid_tr_datasets, Non_iid_te_datasets

