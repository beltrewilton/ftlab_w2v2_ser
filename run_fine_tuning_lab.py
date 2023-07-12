import os
import numpy as np
from dotmap import DotMap
import time
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from train.main_impl import MainImplementation

# change also line: 50
ds_name = "escorpus_pe"  #   <-- [Wilton] ravdess | escorpus_pe | mess

### Hyperparameters
hparams = DotMap()
hparams.batch_size = 64 # 64
hparams.lr = 1e-4
hparams.max_epochs = 10 # 15
hparams.maxseqlen = 10
hparams.nworkers = 1 # it was 4
hparams.precision = 32
# hparams.precision = 16
hparams.saving_path = 'downstream/checkpoints/custom'
hparams.audiopath = f"rawdata/{ds_name}_16k"
hparams.labeldir = f"rawdata/labels_{ds_name}"
# /Users/beltre.wilton/apps/ftlab_w2v2_ser/pretrained_path/wav2vec2-base
hparams.pretrained_path = "/Users/beltre.wilton/apps/ftlab_w2v2_ser/pretrained_path/wav2vec2-base-es-voxpopuli-v2"
# hparams.pretrained_path = "/Users/beltre.wilton/apps/ftlab_w2v2_ser/pretrained_path/wav2vec2-base"
# hparams.pretrained_path = None  --> to download auto the default model from huggingface/facebook
hparams.model_type = 'wav2vec2'
hparams.save_top_k = 1
hparams.num_exps = 1
hparams.outputfile = f"log_{ds_name}_file_{time.time_ns()}.log"

if not os.path.exists(hparams.saving_path):
    os.makedirs(hparams.saving_path)

# We can split the file folds
jsonfile = [f for f in os.listdir(hparams.labeldir) if "json" in f]
nfolds = len(jsonfile)

metrics, confusion = np.zeros((4, hparams.num_exps, nfolds)), 0.

for ifold, foldlabel in enumerate(jsonfile):
    print(f"Running experiment [{ds_name}] epochs: {hparams.max_epochs}, fold {ifold+1} / {nfolds}...")
    hparams.labelpath = os.path.join(hparams.labeldir, foldlabel)

    model = MainImplementation(hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.saving_path,
        filename='escorpus_pe-{epoch:02d}-{valid_loss:.3f}-{valid_UAR:.5f}' if hasattr(model, 'valid_met') else None,
        save_top_k=hparams.save_top_k if hasattr(model, 'valid_met') else 0,
        verbose=True,
        save_weights_only=True,
        monitor='valid_UAR' if hasattr(model, 'valid_met') else None,
        mode='max'
    )

    trainer = Trainer(
        precision=hparams.precision, # default 32
        # amp_backend='native',
        callbacks=[checkpoint_callback] if hasattr(model, 'valid_met') else None,
        # checkpoint_callback=hasattr(model, 'valid_met'),
        # resume_from_checkpoint=None,
        check_val_every_n_epoch=1,
        max_epochs=hparams.max_epochs,
        num_sanity_val_steps=2 if hasattr(model, 'valid_met') else 0,
        # gpus=1, # problems on macOS
        logger=False,
        accelerator="auto", # auto to mps on macOS or gpu on linux
        # log_every_n_steps=5,
        # gradient_checkpointing=False,
    )
    # trainer.checkpoint_callback = hasattr(model, 'valid_met')
    trainer.fit(model)

    try:
        if hasattr(model, 'valid_met'):
            trainer.test()
        else:
            trainer.test(model)
        met = model.test_met
        metrics[:, 0, ifold] = np.array([met.uar * 100, met.war * 100, met.macroF1 * 100, met.microF1 * 100])
        confusion += met.m
    except Exception as ex:
        print(ex)


outputstr = f"+++ SUMMARY [{ds_name}] +++\n"
for nm, metric in zip(('UAR [%]', 'WAR [%]', 'macroF1 [%]', 'microF1 [%]'), metrics):
    outputstr += f"Mean {nm}: {np.mean(metric):.2f}\t"
    outputstr += f"Fold Std. {nm}: {np.mean(np.std(metric, 1)):.2f}\t"
    outputstr += f"Fold Median {nm}: {np.mean(np.median(metric, 1)):.2f}\t"
    outputstr += f"Run Std. {nm}: {np.std(np.mean(metric, 1)):.2f}\t"
    outputstr += f"Run Median {nm}: {np.median(np.mean(metric, 1)):.2f}\n\n"


print(outputstr)

print('\n\n************************* END *************************\n\n')
