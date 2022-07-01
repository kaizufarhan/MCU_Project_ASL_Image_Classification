# import modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import shutil
import time
import cv2 as cv2
import os
import seaborn as sns
import pathlib
from Configuration import *
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

#resnet is a variations of CNN layers
#classify images, for letters



# Making computer use GPU instead of CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# importing training and testing datasets
training_ds = tf.keras.utils.image_dataset_from_directory("C:/Users/fahmed1/Documents/Code/MSU_Project/ASL-database/asl_alphabet_train/asl_alphabet_train/")
# testing_ds = tf.keras.utils.image_dataset_from_directory("C:/Users/fahmed1/Documents/Code/MSU_Project/ASL-database/asl_alphabet_test/asl_alphabet_test/")

# checking class names
class_names = training_ds.class_names
print("Class Names:",class_names, "Total Class:", len(class_names))



# samples of images from the data
plt.figure(figsize=(10, 10))
for images, labels in training_ds.take(1):
  for i in range(20):
    ax = plt.subplot(6,5 , i + 1)
    plt.subplot()
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
# plt.show()

# Normalizing the pixel value between 0 and 1
# Each pixel has a value of 255 because of RGB colors. Since we want it from 0 to 1, if we divide by 255, we get a percentage between 0-1.
# train_ds = training_ds / 255.0
# test_ds = testing_ds / 255.00


# Making the model
model = ResNet50()
print(model.summary())

model2 = ResNet50(include_top=False)
print(model2.summary())

