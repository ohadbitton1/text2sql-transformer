from torch.utils.data import DataLoader
from src.dataset import SQLDataset
import pandas as pd


def get_dataloaders(train_data_path, target_data_path, batch_size, tokenizer, max_length):
    df_val = pd.read_csv(target_data_path)
    df_train = pd.read_csv(train_data_path)
    train_data = SQLDataset(df_train, tokenizer, max_length)
    target_data = SQLDataset(df_val, tokenizer, max_length)
    train_dataloader = DataLoader(train_data, batch_size, shuffle = True)
    target_dataloader = DataLoader(target_data, batch_size, shuffle = False)
    return train_dataloader, target_dataloader
