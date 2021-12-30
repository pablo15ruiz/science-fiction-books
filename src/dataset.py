from hyperparameters import LABEL_COLUMNS, N_WORKERS

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl


class ScienceFictionDataset(Dataset):

    def __init__(self, df, tokenizer, max_token_len=100):
        self.df = df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        description = row.Description
        labels = row[LABEL_COLUMNS]

        encoding = self.tokenizer.encode_plus(
            description,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return dict(
            labels=torch.FloatTensor(labels),
            input_ids=encoding['input_ids'].flatten(),
            attention_mask=encoding['attention_mask'].flatten()
        )


class ScienceFictionDataModule(pl.LightningDataModule):

    def __init__(self, train_df, test_df, tokenizer,
                 batch_size=8, max_token_len=128):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = ScienceFictionDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )
        self.test_dataset = ScienceFictionDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=N_WORKERS
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=N_WORKERS
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=N_WORKERS
        )
