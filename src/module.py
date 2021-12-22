from hyperparameters import BERT_MODEL_NAME
import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import BertModel, AdamW

class ScienceFictionClassifier(pl.LightningModule):
    
    def __init__(self, n_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.criterion = nn.BCELoss()
    
    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        return output
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        print(input_ids, attention_mask, labels)
        output = self.forward(input_ids, attention_mask)
        loss = self.criterion(labels, output)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self.forward(input_ids, attention_mask)
        loss = self.criterion(labels, output)
        return loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return optimizer