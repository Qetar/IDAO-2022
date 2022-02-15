import yaml
import json
import pandas as pd

from pathlib import Path
from baseline import prepare_model
from baseline import read_pymatgen_dict


checkpoint_path = "/tmp/checkpoint"
test_datapath = "./data/dichalcogenides_private"

def main(config):
    model = prepare_model(config)
    model.load_weights(checkpoint_path)

    dataset_path = Path(test_datapath)
    struct = {item.name.strip('.json'): read_pymatgen_dict(item) for item in (dataset_path/'structures').iterdir()}
    private_test = pd.DataFrame(columns=['id', 'structures'], index=struct.keys())
    private_test = private_test.assign(structures=struct.values())
    private_test = private_test.assign(predictions=model.predict_structures(private_test.structures))
    private_test[['predictions']].to_csv('./submission.csv', index_label='id')

if __name__ == '__main__':
    with open("config.json") as f:
        config = json.loads(f.read())
    main(config)