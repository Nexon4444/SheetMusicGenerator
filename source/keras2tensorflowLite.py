import tensorflow as tf

from absl import flags
from absl import app
from tensorflow.keras import backend as K
from pathlib import Path
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
import os
flags.DEFINE_string('input_model', None, 'Path to the input model.')
flags.DEFINE_string('output_model', None, 'Path where the converted model will '
                                          'be stored.')

flags.mark_flag_as_required('input_model')
flags.mark_flag_as_required('output_model')
def main(args):
    FLAGS = flags.FLAGS
    K.set_learning_phase(0)
    FLAGS = flags.FLAGS
    # toco  --keras_model_file ="cpc.h5" --output_file = "cpc.tflite"

    # Convert the model.
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(str(Path(FLAGS.input_model)))
    tflite_model = converter.convert()
    # model_meta = _metadata_fb.ModelMetadataT()


    # Save the model.
    with open(FLAGS.output_model, 'wb') as f:
        f.write(tflite_model)
    # Convert the model.
    # print(os.path.exists(FLAGS.input_model))
    # converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(Pa th(FLAGS.input_model))
    # converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(os.path.join(FLAGS.input_model))
    # tflite_model = converter.convert()

    # Save the model.

if __name__ == "__main__":
    app.run(main)