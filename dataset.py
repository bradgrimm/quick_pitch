from pathlib import Path
from random import randrange

import librosa
import numpy as np
import pandas as pd
from librosa import cqt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from constants import AUDIO_SAMPLE_RATE, ANNOT_N_FRAMES, AUDIO_WINDOW_LENGTH, AUDIO_N_SAMPLES, ANNOTATIONS_N_SEMITONES, \
    MAX_N_SEMITONES, CONTOURS_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, FFT_HOP

DATA_PATH = Path('/mnt/large/data/guitar/')
MANIFEST_PATH = DATA_PATH / 'manifest-2023-10-08.csv'
VALID_COLUMNS = {'onset', 'contour', 'note'}
FRAMES_PER_ANNOT = (AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH) / ANNOT_N_FRAMES


def load_manifest():
    return pd.read_csv(MANIFEST_PATH, index_col=0).reset_index(drop=True)


def load_datasets():
    df = load_manifest()
    df = df[df.audio_path.apply(lambda x: Path(x).parent.name == 'audio_mono-mic')]
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
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx not in self._cache:
            self._cache[idx] = self._load(idx)

        data = self._cache[idx]
        audio = data['audio']
        a_idx = randrange(0, audio.shape[0] - AUDIO_N_SAMPLES)
        audio = np.expand_dims(audio[a_idx:a_idx+AUDIO_N_SAMPLES], axis=0)
        t_idx = self._safe_target_idx(data, a_idx)
        target_data = {
            k: v[t_idx:t_idx+ANNOT_N_FRAMES]
            for k, v in data.items() if k in VALID_COLUMNS
        }
        return {'audio': audio, **target_data}

    def _safe_target_idx(self, data, a_idx):
        t_idx = int(a_idx / FRAMES_PER_ANNOT)
        total = data['contour'].shape[0]
        return np.clip(t_idx, 0, total - ANNOT_N_FRAMES)

    def _load(self, idx):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(str(row.audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)
        all_data = np.load(row.embedding_path, allow_pickle=True)
        embeddings = {k: v for k, v in all_data.item().items() if k in VALID_COLUMNS}
        return {
            'audio': audio,
            **embeddings,
        }
