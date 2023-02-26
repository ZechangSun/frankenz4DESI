import os
import numpy as np
import argparse
from frankenz4DESI.config import get_config
from frankenz4DESI.model import frankenz4DESIKNN
from frankenz4DESI.utils import load_hdf5


parser = argparse.ArgumentParser(description='photometric redshift prediction')
parser.add_argument('--cfg', type=str, required=False, help='path of the config file')
parser.add_argument('--model', type=str, required=False, help='path of the model photometry')
parser.add_argument('--data', type=str, required=False, help='path of the data photometry')
parser.add_argument('--name', type=str, required=False, help='which kind of photometry to use')
parser.add_argument('--output', type=str, required=False, help='path of the output dir')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs")
args = parser.parse_args()



config = get_config(args)


if __name__ == "__main__":
    if not os.path.exists(config.DATA.OUTPUT):
        os.mkdir(config.DATA.OUTPUT)
    with open(os.path.join(config.DATA.OUTPUT, 'config.yaml'), 'w') as f:
        f.write(config.dump()) 
    model = frankenz4DESIKNN(config)
    data = load_hdf5(config.DATA.DATA, name=config.DATA.NAME)
    pdf = model.predict(data['flux'], data['flux_err'], np.ones_like(data['flux'], dtype=bool), output_path=os.path.join(config.DATA.OUTPUT, 'result.hdf5'), object_id=data['object_id'], z=data['z'])




    