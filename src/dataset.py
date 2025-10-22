import torch
from torch.utils.data import Dataset

class SQLDataset (Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        question =  self.dataset['question'][index]
        query = self.dataset['sql'][index]
        T5_prefix = "translate English to SQL: "
        full_request = T5_prefix + question
        source_incoding = self.tokenizer(full_request, max_length= self.max_length, truncation=True, padding= 'max_length', return_tensors='pt')
        target_incoding = self.tokenizer(query, max_length= self.max_length, truncation=True, padding= 'max_length', return_tensors='pt')
        
        dict_data = {
        'input_ids': source_incoding['input_ids'].squeeze(),
        'labels': target_incoding['input_ids'].squeeze(),
        'attention_mask': source_incoding['attention_mask'].squeeze()}

        return dict_data