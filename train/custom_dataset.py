import os
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
import json
import soundfile as sf
from torch.utils.data.dataset import T_co
from torch.utils.data.dataloader import default_collate


class CustomDataset:
    def __init__(self, audiopath, labelpath, maxseqlen):
        super().__init__()
        self.maxseqlen = maxseqlen * 16000  # Assume sample rate of 16000 , w2v2 rules!
        with open(labelpath, 'r') as f:
            self.label = json.load(f)
        self.emoset = list(set([emo for split in self.label.values() for emo in split.values()]))  # labels ['H', 'C', 'A', 'S']
        self.emoset = list(sorted(self.emoset))
        self.nemos = len(self.emoset)
        self.train_dataset = DatasetImpl(audiopath, self.label['Train'], self.emoset, 'training')
        if self.label['Val']:
            self.val_dataset = DatasetImpl(audiopath, self.label['Val'], self.emoset, 'validation')
        if self.label['Test']:
            self.test_dataset = DatasetImpl(audiopath, self.label['Test'], self.emoset, 'testing')

    ##[Wilton] overriding collate_fn -> calling default_collate at the end.
    def seqCollate(self, batch):
        getlen = lambda x: x[0].shape[0]
        max_seqlen = max(map(getlen, batch))
        target_seqlen = min(self.maxseqlen, max_seqlen)

        def trunc(x):
            x = list(x)
            if x[0].shape[0] >= target_seqlen:
                x[0] = x[0][:target_seqlen]
                output_length = target_seqlen
            else:
                output_length = x[0].shape[0]
                over = target_seqlen - x[0].shape[0]
                x[0] = np.pad(x[0], [0, over]) ## [Wilton] add 0's
            ret = (x[0], output_length, x[1])
            return ret

        batch = list(map(trunc, batch))
        return default_collate(batch)


class DatasetImpl(Dataset):
    def __init__(self, audiopath, label, emoset, split, maxseqlen=10): #[Wilton ] fix to 10 instead 12
        super().__init__()
        self.maxseqlen = maxseqlen * 16000  # [Wilton] <----  TWO TIMES??? Assume sample rate of 16000 #
        self.split = split
        self.label = label  # {wavname: emotion_label}
        self.emos = Counter([self.label[n] for n in self.label.keys()])  # Counter({'C': 302, 'A': 292, 'H': 290, 'S': 268})
        self.emoset = emoset
        self.labeldict = {k: i for i, k in enumerate(self.emoset)}
        self.datasetbase = list(self.label.keys())
        self.dataset = [os.path.join(audiopath, x) for x in self.datasetbase]  # Clean .json file with only file name

        # Print statistics:
        print(f'Statistics of {self.split} splits:')
        print('----Involved Emotions----')
        for k, v in self.emos.items():
            print(f'{k}: {v} examples')
        l = len(self.dataset)
        print(f'Total {l} examples')
        print('----Examples Involved----\n')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i) -> T_co:
        dataname = self.dataset[i]
        wav, _sr = sf.read(dataname)
        _label = self.label[self.datasetbase[i]]
        label = self.labeldict[_label]
        return wav.astype(np.float32), label

    def get_sample(self, wav_name=None):
        if wav_name is None:
            wav_name = np.random.choice(self.dataset)
        else:
            wav_name = self.dataset[wav_name in self.label]
        wav, _sr = sf.read(wav_name, dtype='double')
        duration = wav.shape[0] / _sr
        _label = self.label[wav_name]
        label = self.labeldict[_label]
        wav = torch.Tensor(wav).reshape(1, -1)
        return wav, label, _label, wav_name, duration

    def to_label(self, i):
        ld = self.labeldict
        return list(ld.keys())[list(ld.values()).index(i)]
