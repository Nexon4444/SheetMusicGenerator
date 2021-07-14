import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt

IMG_SIZE = 28
batch_holder = np.zeros((2, IMG_SIZE, IMG_SIZE, 1))
img_dir='test_set/'
loaded_model = tf.keras.models.load_model('models/model3.h5')

for i,img_path in enumerate(Path(img_dir).iterdir()):
  img = cv2.imread(str(img_path))
  plt.imshow(img)
  plt.figure()

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # numpy.array

  resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
  reshaped = resized.reshape(28, 28, 1)
  plt.imshow(reshaped, cmap='gray', vmin=0, vmax=255)
  plt.figure()
  # plt.show()
  normalized = reshaped.astype('float32') / 255
  # img = image.load_img(os.path.join(img_dir,img), target_size=(IMG_SIZE,IMG_SIZE))
  plt.imshow(normalized, cmap='gray', vmin=0, vmax=1)
  plt.show()
  batch_holder[i, :] = normalized

result=loaded_model.predict_classes(batch_holder)
print(result)