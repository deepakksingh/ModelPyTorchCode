import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    'Implementation of a custom dataset in PyTorch'

    def __init__(self, list_IDs, labels):
        'Initialization'

        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'

        ID = self.list_IDs[index]

        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y
        