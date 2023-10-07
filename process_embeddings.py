import os
import signal
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
import typer
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from tqdm.auto import tqdm

app = typer.Typer()
DATA_PATH = '/mnt/large/data/guitar/'


@app.command()
def process_embeddings():
    basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
    os.makedirs(Path(DATA_PATH) / 'embeddings', exist_ok=True)
    for path in tqdm(glob('/mnt/large/data/guitar/*/*.wav')):
        path = Path(path)
        source_name = Path(path.parent.name) / path.name
        out_path = path.parent.parent / 'embeddings' / path.with_suffix('').name
        if os.path.exists(out_path.with_suffix('.npy')):
            continue

        model_output, midi_data, note_events = predict(path, basic_pitch_model)
        model_output['source'] = source_name
        with DelayedKeyboardInterrupt():
            np.save(out_path, model_output)


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