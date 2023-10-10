from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from basic_pitch.constants import AUDIO_SAMPLE_RATE, AUDIO_WINDOW_LENGTH, ANNOT_N_FRAMES
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

DATA_PATH = Path('/mnt/large/data/guitar/')
MANIFEST_PATH = DATA_PATH / 'manifest-2023-10-08.csv'
VALID_COLUMNS = {'onset', 'contour', 'note'}


def load_manifest():
    return pd.read_csv(MANIFEST_PATH, index_col=0).reset_index(drop=True)


def load_datasets():
    df = load_manifest()
    train_df, val_df = train_test_split(df, test_size=0.2)
    train_dataset = EmbeddingDataset(train_df)
    val_dataset = EmbeddingDataset(val_df)
    return train_dataset, val_dataset


def estimate_annotations(n_samples):
    total_frames = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH
    frames_per_annot = total_frames / ANNOT_N_FRAMES
    n = (n_samples // frames_per_annot).astype(int)
    n += ((n_samples % total_frames) == 0).astype(int)
    return n


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
            k: self._prepare_target(v)
            for k, v in embedding_data.item().items()
            if k in VALID_COLUMNS
        }
        return {'audio': self._prepare_audio(audio), **target_data}

    def _prepare_audio(self, x):
        # TODO: how much adio do we actually want?
        return np.expand_dims(x[:(AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH)], axis=0)

    def _prepare_target(self, x):
        # TODO: How many targets can we handle?
        return x[:ANNOT_N_FRAMES]
