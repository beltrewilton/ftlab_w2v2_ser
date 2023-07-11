import os
from pathlib import Path
import json

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import soundfile as sf
from tqdm import tqdm
from vad.vad_lab import VAD

vad = VAD(mapping="Ekman")

RAVDESS_PATH = Path("/Users/beltre.wilton/Downloads/SER-Datasets/RAVDESS")
AUDIO_TARGET = "/Users/beltre.wilton/apps/ftlab_w2v2_ser/rawdata/ravdess_16k"
LABELS = "/Users/beltre.wilton/apps/ftlab_w2v2_ser/rawdata/labels_ravdess"
CSV_FILE = "ravdess.csv"

if not os.path.exists(AUDIO_TARGET):
    os.makedirs(AUDIO_TARGET)

if not os.path.exists(LABELS):
    os.makedirs(LABELS)


def convert_16k(source: Path, target: str, hz: int = 16000):
    ravdess = []
    source_list = np.array([f for f in source.rglob("*.wav")])
    print("Aqui vamos con RAVDESS.")
    for f in tqdm(source_list):
        data, sr = sf.read(f.__str__())
        data_16k = librosa.resample(data, orig_sr=sr, target_sr=hz)
        newaudio = os.path.join(target, f.name)
        sf.write(newaudio, data_16k, hz)
        # valence, arousal, dominance
        v, a, d = int(f.name[12:14]), int(f.name[15:17]), int(f.name[18:20])
        cat = vad.vad2categorical(v, a, d, k=1)
        cat = cat[0][0]['term']
        ravdess.append([os.path.join(target, f.name), cat, data.shape[0] / sr])
    return np.array(ravdess)


def build_dataset(ravdess):
    np.random.shuffle(ravdess)
    X, y = ravdess[:, 0], ravdess[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    dataset = {
        "Train": {x: y_train[i] for i, x in enumerate(X_train)},
        "Val": {x: y_val[i] for i, x in enumerate(X_val)},
        "Test": {x: y_test[i] for i, x in enumerate(X_test)}
    }

    pd.DataFrame(ravdess).to_csv(
        os.path.join(LABELS, CSV_FILE),
        header=None,
        index=None)

    return dataset


if __name__ == "__main__":
    ravdess = convert_16k(RAVDESS_PATH, AUDIO_TARGET)
    dataset = build_dataset(ravdess)
    Path(f"{LABELS}/ravdess.json")
    with open(f"{LABELS}/ravdess.json", 'w') as f:
        json.dump(dataset, f)