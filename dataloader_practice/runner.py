import torch
from torch.utils.data import DataLoader
from myDatasetClass import CustomDataset

#CUDA for pytorch

# cuda_availability_flag = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# parameters
params = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 6
}

max_epochs = 100

"""
Notations
Before getting started, let's go through a few organizational tips that are particularly useful when dealing with large datasets.

Let ID be the Python string that identifies a given sample of the dataset. A good way to keep track of samples and their labels is to adopt the following framework:

1) Create a dictionary called partition where you gather:

- in partition['train'] a list of training IDs
- in partition['validation'] a list of validation IDs

2) Create a dictionary called labels where for each ID of the dataset, the associated label is given by labels[ID]

For example, let's say that our training set contains id-1, id-2 and id-3 with respective labels 0, 1 and 2, with a validation set containing id-4 with label 1. In that case, the Python variables partition and labels look like
>>> partition
{'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}

>>> labels
{'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1}

"""

# Dataset
partition = # IDs
labels = # Labels

training_set = CustomDataset(partition['train'],labels)
training_generator = DataLoader(training_set, **params)

validation_set = CustomDataset(partition['validation'],labels)
validation_generator = DataLoader(validation_set, **params)

# loop over epochs

for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:

        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computation steps go below

        # Enter model computation steps


    # Validation
    with torch.no_grad():
        for local_batch, local_labels in validation_generator:

            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computation


