from torch.utils.data import DataLoader
from src.dataset import SQLDataset
import pandas as pd


def get_dataloaders(train_data_path, val_data_path, test_data_path, batch_size, tokenizer, max_length):
    df_train = pd.read_csv(train_data_path)
    df_val = pd.read_csv(val_data_path)
    df_test = pd.read_csv(test_data_path)
    train_data = SQLDataset(df_train, tokenizer, max_length)
    val_data = SQLDataset(df_val, tokenizer, max_length)
    test_data = SQLDataset(df_test, tokenizer, max_length)
    train_dataloader = DataLoader(train_data, batch_size, shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size, shuffle = False)
    test_dataloader = DataLoader(test_data, batch_size, shuffle= False)
    return train_dataloader, val_dataloader, test_dataloader
