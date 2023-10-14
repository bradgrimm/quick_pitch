import os
import signal
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import typer
from quick_pitch import ICASSP_2022_MODEL_PATH
from quick_pitch.inference import predict
from tqdm.auto import tqdm

app = typer.Typer()
DATA_PATH = Path('/mnt/large/data/guitar/')


@app.command()
def process_embeddings():
    basic_pitch_model = None
    os.makedirs(DATA_PATH / 'embeddings', exist_ok=True)

    manifest = []
    glob_exp = str(DATA_PATH / '*' / '*.wav')
    for path in tqdm(glob(glob_exp)):
        path = Path(path)
        source_name = Path(path.parent.name) / path.name
        out_path = path.parent.parent / 'embeddings' / path.with_suffix('').name
        manifest.append({
            "audio_path": path,
            "embedding_path": out_path.with_suffix('.npy'),
        })
        if os.path.exists(out_path.with_suffix('.npy')):
            continue

        if basic_pitch_model is None:
            basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))

        model_output, midi_data, note_events = predict(path, basic_pitch_model)
        model_output['source'] = source_name
        with DelayedKeyboardInterrupt():
            np.save(out_path, model_output)

    formatted_date = datetime.now().strftime("%Y-%m-%d")
    df = pd.DataFrame(manifest)
    df.to_csv(DATA_PATH / f'manifest-{formatted_date}.csv')


class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


if __name__ == '__main__':
    process_embeddings()
