import os
from pathlib import Path
import json

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import soundfile as sf
from tqdm import tqdm


MESS_PATH = Path("/Users/beltre.wilton/Downloads/SER-Datasets/MESS-compressed")
TARGET = "/Users/beltre.wilton/apps/ftlab_w2v2_ser/rawdata/mess_16k"
LABELS = "/Users/beltre.wilton/apps/ftlab_w2v2_ser/rawdata/labels_mess"
CSV_FILE = "mess.csv"


if not os.path.exists(TARGET):
    os.makedirs(TARGET)

if not os.path.exists(LABELS):
    os.makedirs(LABELS)


def convert_16k(source: Path = MESS_PATH, target: str = TARGET, hz: int = 16000):
    mess_list = []
    source_list = np.array([f for f in source.rglob("*.wav")])
    print("Aqui vamos con MESS.")
    for f in tqdm(source_list):
        data, sr = sf.read(f.__str__())
        data_16k = librosa.resample(data, orig_sr=sr, target_sr=hz)
        newaudio = os.path.join(target, f.name)
        sf.write(newaudio, data_16k, hz)
        mess_list.append([os.path.join(target, f.name), f.name[:1], data.shape[0] / sr])

    return np.array(mess_list)


def build_dataset(mess_list):
    np.random.shuffle(mess_list)
    X, y = mess_list[:, 0], mess_list[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    dataset = {
        "Train": {x: y_train[i] for i, x in enumerate(X_train)},
        "Val": {x: y_val[i] for i, x in enumerate(X_val)},
        "Test": {x: y_test[i] for i, x in enumerate(X_test)}
    }

    pd.DataFrame(mess_list).to_csv(
        os.path.join(LABELS, CSV_FILE),
        header=None,
        index=None)

    return dataset


if __name__ == "__main__":
    mess = convert_16k(MESS_PATH, TARGET, hz=16000)
    dataset = build_dataset(mess)
    Path(f"{LABELS}/mess.json")
    with open(f"{LABELS}/mess.json", 'w') as f:
        json.dump(dataset, f)
