from dataset import ScienceFictionDataModule
from hyperparameters import BERT_MODEL_NAME, LABEL_COLUMNS
from module import ScienceFictionClassifier

import pandas as pd

from sklearn.model_selection import train_test_split

from pytorch_lightning import Trainer

from transformers import BertTokenizer

def main():
    df = pd.read_csv('../data/science_fiction_books.csv')[:100]

    train_df, test_df = train_test_split(df, test_size=0.1)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    data_module = ScienceFictionDataModule(train_df, test_df, tokenizer, batch_size=8,
        max_token_len=512)
    model = ScienceFictionClassifier(len(LABEL_COLUMNS))

    trainer = Trainer(log_every_n_steps=2, max_epochs=1)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()