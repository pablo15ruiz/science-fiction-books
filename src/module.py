from hyperparameters import BERT_MODEL_NAME
import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel, AdamW


class ScienceFictionClassifier(pl.LightningModule):

    def __init__(self, n_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.criterion = nn.BCELoss()

    def forward(self, input_ids, attention_mask):
        print(input_ids, attention_mask)
        output = self.bert(input_ids, attention_mask=attention_mask)
        print(output.pooler_output)
        output = self.classifier(output.pooler_output)
        print(output)
        output = torch.sigmoid(output)
        print(output)
        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self(input_ids, attention_mask)
        loss = self.criterion(labels, output)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self(input_ids, attention_mask)
        loss = self.criterion(labels, output)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self(input_ids, attention_mask)
        loss = self.criterion(labels, output)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return optimizer
