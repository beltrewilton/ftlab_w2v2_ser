import torch
import torch.nn as nn
import lightning.pytorch as pl
from train.wav2vec2_wrapper import Wav2vec2Wrapper


class RNNHead(pl.LightningModule):
    def __init__(self, n_classes, wav2vecpath=None):
        super().__init__()
        self.backend = "wav2vec2"
        self.wav2vec2 = Wav2vec2Wrapper(pretrain=False, wav2vecpath=wav2vecpath)
        feature_dim = 768 # base 768,  large 1024

        """
        In this paper, we explore methods for fine-tuning wav2vec
        2.0 on SER. We show that by adding a simple neural network 
        self.rnn_head on top of wav2vec 2.0, vanilla fine-tuning (V-FT) 
        outperforms state-of-the-art (SOTA)
        """
        self.rnn_head = nn.LSTM(feature_dim, 256, 1, bidirectional=True)
        # self.rnn_head = nn.LSTM(feature_dim, 512, 1, bidirectional=True)
        self.linear_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(768, n_classes)
        )

    def trainable_params(self):
        return list(self.rnn_head.parameters()) + list(self.linear_head.parameters()) + list(self.wav2vec2.trainable_params())

    def forward(self, x, length):
        reps = self.wav2vec2(x, length)
        last_feat_pos = self.wav2vec2.get_feat_extract_output_lengths(length) - 1
        logits = reps.permute(1, 0, 2) #L, B, C
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0), -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        logits = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        hidden_reps = logits ## [Wilton]
        logits = self.linear_head(logits)
        return logits
