from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

DATA_PATH = Path('/mnt/large/data/guitar/')
MANIFEST_PATH = DATA_PATH / 'manifest-2023-10-08.csv'
AUDIO_SAMPLE_RATE = 16000
VALID_COLUMNS = {'onset', 'contour', 'note'}


def load_manifest():
    return pd.read_csv(MANIFEST_PATH, index_col=0).reset_index(drop=True)


def load_datasets():
    df = load_manifest()
    train_df, val_df = train_test_split(df, test_size=0.2)
    train_dataset = EmbeddingDataset(train_df)
    val_dataset = EmbeddingDataset(val_df)
    return train_dataset, val_dataset


class EmbeddingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(str(row.audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
        embedding_data = np.load(row.embedding_path, allow_pickle=True)
        target_data = {
            k: torch.from_numpy(v)
            for k, v in embedding_data.item().items()
            if k in VALID_COLUMNS
        }
        return {'audio': audio, **target_data}
