import torch
from torch.utils.data import Dataset

class AviationDataset(Dataset):
    def __init__(self, sequences, metadata, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.metadata = torch.FloatTensor(metadata)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.metadata[idx], self.labels[idx]