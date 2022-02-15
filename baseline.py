import yaml
import json
import random

import pandas as pd
import numpy as np
import tensorflow as tf
import argparse

from pathlib import Path
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanAbsoluteError

train_datapath = "./data/dichalcogenides_public"

parser = argparse.ArgumentParser()
parser.add_argument("--seed",type=int,default=17)
parser.add_argument("--epochs",type=int,default=800)
parser.add_argument("--batch_size",type=int,default=128)
parser.add_argument("--lr",type=float,default=2e-4)
parser.add_argument("--cutoff",type=float,default=4.0)
parser.add_argument("--nblocks",type=int,default=3)
parser.add_argument("--npass",type=int,default=2)
parser.add_argument("--width",type=int,default=0.8)
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


def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ewt(prediction, target):
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    e_thresh = 0.02
    error_energy = tf.math.abs(target - prediction)

    success = tf.math.count_nonzero(error_energy < e_thresh)
    total = tf.size(target)
    return success / tf.cast(total, tf.int64)

def prepare_dataset():
    dataset_path = Path(train_datapath)
    targets = pd.read_csv(dataset_path / "targets.csv", index_col=0)
    struct = {
        item.name.strip(".json"): read_pymatgen_dict(item)
        for item in (dataset_path / "structures").iterdir()
    }

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
    train, test = prepare_dataset()
    model = prepare_model(config)
    model.train(
        train.structures,
        train.targets,
        validation_structures=test.structures,
        validation_targets=test.targets,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[model_checkpoint_callback,WandbCallback()],
        save_checkpoint=False,
    )


if __name__ == "__main__":
    with open("config.json", 'w') as f:
        json.dump(vars(args), f, indent=4)
    main(wandb.config)
