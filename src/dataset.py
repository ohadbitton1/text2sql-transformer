import torch
from torch.utils.data import Dataset

class SQLDataset (Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length


    def __len__(self):
        return len(self.dataset)