import os
from pathlib import Path
import json

import librosa
import numpy as np
import pandas as pd
import scipy.signal as signal
from sklearn.model_selection import train_test_split
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm
from vad.vad_lab import VAD

vad = VAD(mapping="Ekman")

module_directory = Path(__file__).parent.parent
os.chdir(module_directory)
root = os.getcwd()

GRIDSPACE_PATH = Path("/Users/beltre.wilton/Downloads/SER-Datasets/gridspace-stanford-harper-valley/data/audio")
META_DATA = Path("/Users/beltre.wilton/Downloads/SER-Datasets/gridspace-stanford-harper-valley/data/metadata")
GRIDSPACE_PATH_JSON = Path("/Users/beltre.wilton/Downloads/SER-Datasets/gridspace-stanford-harper-valley/data/transcript")
AUDIO_TARGET = f"{root}/rawdata/gridspace_16k"
AUDIO_PREPROCESS = f"{root}/rawdata/gridspace_16k/preprocess"
LABELS = f"{root}/rawdata/labels_gridspace"
CSV_FILE = "gridspace.csv"
CSV_FILE_TASK = "gridspace_task.csv"

if not os.path.exists(AUDIO_PREPROCESS):
    os.makedirs(AUDIO_PREPROCESS)

if not os.path.exists(AUDIO_TARGET):
    os.makedirs(AUDIO_TARGET)

if not os.path.exists(LABELS):
    os.makedirs(LABELS)


def convert_16k(source: Path, target: str, hz: int = 16000):
    gridspace = []
    gridspace_task = []
    source_list = np.array([f for f in source.rglob("*.json")])
    print("Aqui vamos con Gridspace")
    for f in tqdm(source_list):
        meta = None
        with open(f.__str__(), 'r') as jf:
            meta = json.load(jf)

        wavfile = f.name[:-4] + 'wav'
        agent = AudioSegment.from_file(GRIDSPACE_PATH.joinpath('agent').joinpath(wavfile).__str__(), format="wav")
        caller = AudioSegment.from_file(GRIDSPACE_PATH.joinpath('caller').joinpath(wavfile).__str__(), format="wav")
        merged = agent.overlay(caller)
        merged_file = os.path.join(AUDIO_PREPROCESS, f.name[:-5] + '-merged.wav')
        merged.export(merged_file, format="wav")

        merged, sr = sf.read(merged_file)
        resample_ratio = hz / sr
        resampl = int(len(merged) * resample_ratio)
        up_data = signal.resample(merged, resampl)
        sf.write(merged_file, up_data, hz)

        meta = [meta[i] for i in range(len(meta)) if meta[i]['human_transcript'] != "[noise]"]
        for i in range(len(meta)):
            # 2: agent, 1: caller
            # if meta[i]['channel_index'] == 2:
            #     wav_path_channel = GRIDSPACE_PATH.joinpath('agent').joinpath(wavfile).__str__()
            # else:
            #     wav_path_channel = GRIDSPACE_PATH.joinpath('caller').joinpath(wavfile).__str__()

            chunk = None
            mult = 16    # 16kz because start_ms is in milliseconds
            if len(meta) - 1 == i:
                chunk = up_data[int(meta[i]['start_ms'] * mult):]
            else:
                chunk = up_data[int(meta[i]['start_ms'] * mult): int(meta[i+1]['start_ms'] * mult)]

            newwavfile = f.name[:-5] + f'-{i}-{meta[i]["channel_index"]}.wav'
            newaudio = os.path.join(target, newwavfile)
            sf.write(newaudio, chunk, hz)

            metadata = META_DATA.joinpath(f.name[:-4] + 'json')
            with open(metadata.__str__(), 'r') as mf:
                metafile = json.load(mf)

            task = metafile['tasks'][0]['task_type']

            neutral, negative, positive = meta[i]['emotion']['neutral'], meta[i]['emotion']['negative'], meta[i]['emotion']['positive']
            emot = np.argmax([neutral, negative, positive])
            emot = np.array(['neutral', 'negative', 'positive'])[emot]

            gridspace.append([f.name[:-5], newaudio, neutral, negative, positive, emot,
                              chunk.shape[0] / hz, task, meta[i]["channel_index"], i])
        gridspace_task.append([f.name[:-5], merged_file, merged.shape[0]/ sr, task])
    return np.array(gridspace), np.array(gridspace_task)


def build_dataset(gridspace, gridspace_task):
    # np.random.shuffle(escorpus_pe)  # <---- random it train_dataloader shuffle=True also randomized it?
    ## some balancing btw classes.
    df = pd.DataFrame(data=gridspace, columns=['Audio_Name', 'Audio_Path', 'Neutral', 'Negative', 'Positive', 'Emotion', 'Duration', 'Task', 'Channel', 'Utterance#'])
    df.Duration = df.Duration.astype(float)
    # df = df[(df.Duration > 1.99) & (df.Duration < 10.1)]
    # dfn = df[df.Emotion == 'neutral']
    # dfp = df[df.Emotion == 'positive']
    # dfx = df[df.Emotion == 'negative']
    # max = int(dfx.shape[0] * 1.8)
    # dfn = dfn.sample(max)
    # dfp = dfp.sample(max)
    # df = pd.concat([dfn, dfp, dfx])
    gridspace = df.to_numpy()

    X, y = gridspace[:, 1], gridspace[:, 5]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    Xt, yt = gridspace_task[:, 1], gridspace_task[:, 3]
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(Xt, yt, test_size=0.2, random_state=42)
    Xt_train, Xt_val, yt_train, yt_val = train_test_split(Xt_train, yt_train, test_size=0.2, random_state=42)

    dataset = {
        "Train": {x: y_train[i] for i, x in enumerate(X_train)},
        "Val": {x: y_val[i] for i, x in enumerate(X_val)},
        "Test": {x: y_test[i] for i, x in enumerate(X_test)}
    }

    dataset_task = {
        "Train": {x: yt_train[i] for i, x in enumerate(Xt_train)},
        "Val": {x: yt_val[i] for i, x in enumerate(Xt_val)},
        "Test": {x: yt_test[i] for i, x in enumerate(Xt_test)}
    }

    pd.DataFrame(gridspace).to_csv(
        os.path.join(LABELS, CSV_FILE),
        header=None,
        index=None)

    pd.DataFrame(gridspace_task).to_csv(
        os.path.join(LABELS, CSV_FILE_TASK),
        header=None,
        index=None)

    return dataset, dataset_task


if __name__ == "__main__":
    gridspace, gridspace_task = convert_16k(GRIDSPACE_PATH_JSON, AUDIO_TARGET)
    dataset, dataset_task = build_dataset(gridspace, gridspace_task)

    Path(f"{LABELS}/gridspace.json")
    with open(f"{LABELS}/gridspace.json", 'w') as f:
        json.dump(dataset, f)

    Path(f"{LABELS}/gridspace_task.json")
    with open(f"{LABELS}/gridspace_task.json", 'w') as f:
        json.dump(dataset_task, f)
