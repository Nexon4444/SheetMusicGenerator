import json

import h5py

filepath = "models/weights.h5"
f = h5py.File(filepath, mode='r')
model_config = f.attrs.get('model_config')
model_config = json.loads(model_config.decode('utf-8'))