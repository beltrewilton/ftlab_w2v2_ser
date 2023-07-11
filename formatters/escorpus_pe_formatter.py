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

ESCORPUS_PATH = Path("/Users/beltre.wilton/Downloads/SER-Datasets/Corpus_Globalv1")
AUDIO_TARGET = "/Users/beltre.wilton/apps/ftlab_w2v2_ser/rawdata/escorpus_pe_16k"
LABELS = "/Users/beltre.wilton/apps/ftlab_w2v2_ser/rawdata/labels_escorpus_pe"
CSV_FILE = "escorpus_pe.csv"


if not os.path.exists(AUDIO_TARGET):
    os.makedirs(AUDIO_TARGET)

if not os.path.exists(LABELS):
    os.makedirs(LABELS)


def convert_16k(source: Path, target: str, hz: int = 16000):
    escorpus_pe = []
    source_list = np.array([f for f in source.rglob("*.wav")])
    print("Aqui vamos con ES Corpus Peru.")
    for f in tqdm(source_list):
        data, sr = librosa.load(f.__str__(), mono=True, sr=hz)
        newaudio = os.path.join(target, f.name)
        sf.write(newaudio, data, hz)
        # valence, arousal, dominance
        split = f.name.split("-")
        try:
            v, a, d = int(split[1]), int(split[2]), int(split[3][:2])
        except Exception as ex:
            v, a, d = int(split[1]), int(split[2]), 1

        cat = vad.vad2categorical(v, a, d, k=1)
        cat = cat[0][0]['term']
        escorpus_pe.append([os.path.join(target, f.name), cat, data.shape[0] / sr])
    return np.array(escorpus_pe)


def build_dataset(escorpus_pe):
    np.random.shuffle(escorpus_pe)  # <---- random it train_dataloader shuffle=True also randomized it?
    X, y = escorpus_pe[:, 0], escorpus_pe[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    dataset = {
        "Train": {x: y_train[i] for i, x in enumerate(X_train)},
        "Val": {x: y_val[i] for i, x in enumerate(X_val)},
        "Test": {x: y_test[i] for i, x in enumerate(X_test)}
    }

    pd.DataFrame(escorpus_pe).to_csv(
        os.path.join(LABELS, CSV_FILE),
        header=None,
        index=None)

    return dataset


if __name__ == "__main__":
    escorpus_pe = convert_16k(ESCORPUS_PATH, AUDIO_TARGET)
    dataset = build_dataset(escorpus_pe)
    Path(f"{LABELS}/escorpus_pe.json")
    with open(f"{LABELS}/escorpus_pe.json", 'w') as f:
        json.dump(dataset, f)
