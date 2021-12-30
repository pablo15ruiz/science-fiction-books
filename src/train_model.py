from dataset import ScienceFictionDataModule
from hyperparameters import BERT_MODEL_NAME, LABEL_COLUMNS
from module import ScienceFictionClassifier

import pandas as pd

from sklearn.model_selection import train_test_split

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import BertTokenizer

BATCH_SIZE = 12
MAX_TOKEN_LEN = 512


def main():
    df = pd.read_csv('../data/science_fiction_books.csv')
    train_df, test_df = train_test_split(df, test_size=0.2)

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    data_module = ScienceFictionDataModule(
        train_df,
        test_df,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_token_len=MAX_TOKEN_LEN
    )
    model = ScienceFictionClassifier(len(LABEL_COLUMNS))

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint',
        monitor='val_loss',
        verbose=True
    )
    logger = TensorBoardLogger('lightning_logs', name='science-fiction-books')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_train_batches=2
    )
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
