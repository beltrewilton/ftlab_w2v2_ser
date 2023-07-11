import torch
import numpy as np
import time
from dotmap import DotMap
from torch.utils.data import DataLoader
from torchviz import make_dot
from train.rnn_head import RNNHead
from train.custom_dataset import CustomDataset
from train.main_impl import MainImplementation

### Hyperparameters
hparams = DotMap()
hparams.batch_size = 64
hparams.lr = 1e-4
hparams.max_epochs = 2 # 15
hparams.maxseqlen = 10 # check the avg of the all audios.
hparams.nworkers = 1 # it was 4
hparams.precision = 32
hparams.saving_path = 'downstream/checkpoints/custom'
hparams.audiopath = "rawdata/escorpus_pe_16k"
hparams.labelpath = "rawdata/labels_escorpus_pe/escorpus_pe.json"
hparams.pretrained_path = None
hparams.model_type = 'wav2vec2'
hparams.save_top_k = 1
hparams.num_exps = 1
hparams.outputfile = f"log_file_{time.time_ns()}.log"


dataset = CustomDataset(hparams.audiopath, hparams.labelpath, maxseqlen=hparams.maxseqlen)
# loader = DataLoader(dataset=dataset.test_dataset, batch_size=1, drop_last=False)
# audio, label = next(iter(loader))
# (3.068125 * 60) / 153  = 1.2031862745098039
# (2.9464375 * 60) / 147 = 1.202627551020408


# chkpt = "./downstream/checkpoints/custom/epoch=01-valid_loss=0.506-valid_UAR=0.86464.ckpt"
# chkpt = "./downstream/checkpoints/custom/ravdess-epoch=06-valid_loss=0.284-valid_UAR=0.91667.ckpt"
chkpt = "./downstream/checkpoints/custom/escorpus_pe-epoch=07-valid_loss=0.918-valid_UAR=0.66108.ckpt"
model = MainImplementation.load_from_checkpoint(chkpt, hparams=hparams, inference=True)

samples=10

print(f"\n ========= Runnning {samples} random inference samples ========= \n")
for i in range(samples):
    audio, label, label_str, wav_name, duration = dataset.test_dataset.get_sample()
    # obj = dataset.seqCollate([audio])
    lenght = torch.LongTensor([audio.size(1)])  # .to(label.device)

    print(f"random sample #{i + 1}\n\tfile: {wav_name} \n\taudio duration {duration}")
    print(f"\ty_true: {label} {label_str}")

    pred = model(audio, lenght)

    y_hat = np.argmax(pred.detach().numpy())
    y_hat_str = dataset.test_dataset.to_label(y_hat)
    print(f"\ty_hat: {y_hat} {y_hat_str}")
    print()

    # torch.onnx.export(model, (audio, lenght), "model.onnx")
    # make_dot(pred, params=dict(model.named_parameters()),).render("diagram/ftlab_model", format="pdf")
