import yaml
import json
import random
import wandb

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

from pathlib import Path
from sklearn.model_selection import train_test_split
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.metrics import MeanAbsoluteError
from utils import *

train_datapath = "./data/dichalcogenides_public"
additional_datapath = "./data/dichalcogenides_additional"

parser = argparse.ArgumentParser()
parser.add_argument("--seed",type=int,default=17)
parser.add_argument("--epochs",type=int,default=800)
parser.add_argument("--batch_size",type=int,default=128)
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument("--cutoff",type=float,default=4.0)
parser.add_argument("--nblocks",type=int,default=3)
parser.add_argument("--npass",type=int,default=2)
parser.add_argument("--width",type=float,default=0.8)
parser.add_argument("--nfeat_bond",type=int,default=10)
parser.add_argument("--embedding_dim",type=int,default=16)
args = parser.parse_args()

wandb.init(project="IDAO-2022", entity="asu-clowns")
wandb.config.update(args)


checkpoint_filepath = "/tmp/checkpoint"
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_ewt",
    mode="max",
    save_best_only=True)

reduce_lr_callback  = ReduceLROnPlateau(monitor="val_loss",mode="min", factor=0.2,patience=30, min_lr=0.00001,verbose=1)
scheduler_callback = LearningRateScheduler(scheduler,verbose=1)


def ewt(prediction, target):
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    e_thresh = 0.02
    error_energy = tf.math.abs(target - prediction)

    success = tf.math.count_nonzero(error_energy < e_thresh)
    total = tf.size(target)
    return success / tf.cast(total, tf.int64)

def prepare_dataset():
    train_path = Path(train_datapath)
    additional_path = Path(additional_datapath)
    train_targets = pd.read_csv(train_path / "targets.csv", index_col=0)
    additional_targets = pd.read_csv(additional_path / "targets.csv", index_col=0)
    targets = pd.concat([train_targets,additional_targets],axis=0)

    train_structs = {
        item.name.strip(".json"): read_pymatgen_dict(item)
        for item in (train_path / "structures").iterdir()
    }
    additional_structs ={
        item.name.strip(".json"): read_pymatgen_dict(item)
        for item in (additional_path / "structures").iterdir()
    }
    struct =  {**train_structs, **additional_structs}
    data = pd.DataFrame(columns=["structures"], index=struct.keys())
    data = data.assign(structures=struct.values(), targets=targets)

    return train_test_split(data, test_size=0.25)

 
def prepare_model(config):
    nfeat_bond = config["nfeat_bond"]
    r_cutoff = config["cutoff"]
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = config["width"]
    
    return MEGNetModel(
        graph_converter=CrystalGraph(cutoff=r_cutoff),
        centers=gaussian_centers,
        width=gaussian_width,
        nblocks=config["nblocks"],
        loss=["MAE"],
        npass=config["npass"],
        lr=config["lr"],
        embedding_dim=config["embedding_dim"],
        metrics=[ewt,tf.keras.metrics.MeanAbsoluteError()]
    )

def main(config):
    seed_everything(config["seed"])
    train, test = prepare_dataset()
    model = prepare_model(config)
    model.train(
        train.structures,
        train.targets,
        validation_structures=test.structures,
        validation_targets=test.targets,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[model_checkpoint_callback,WandbCallback(),reduce_lr_callback],
        save_checkpoint=False,
    )


if __name__ == "__main__":
    with open("config.json", 'w') as f:
        json.dump(vars(args), f, indent=4)
    main(wandb.config)
