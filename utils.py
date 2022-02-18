import numpy as np
import tensorflow as tf
import random
import json
from pymatgen.core import Structure


def scheduler(epoch,lr):
    if epoch < 120:
        return 0.001
    
    return 0.0001

def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)