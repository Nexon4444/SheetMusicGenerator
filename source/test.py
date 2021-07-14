import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from pathlib import Path
from absl import app
from absl import flags
from absl import logging
import keras
from keras import backend as K
from keras.models import model_from_json, model_from_yaml

K.set_learning_phase(0)
FLAGS = flags.FLAGS
input_model_path="models"
keras.models.load_model(Path(input_model_path), compile=False)