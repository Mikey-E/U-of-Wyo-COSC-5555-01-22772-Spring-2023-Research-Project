#Author: Michael Elgin

#File to load the preprocessed data from the data folder.
#Two functions to use: load_data, load_labels
#Example:
#	load_data("train", 32) #Would return our (10000, 32, 32, 3) tensor
# 	load_labels("train") #Would return the labels in sync with that [0, 1, 0, ...]

import os
import numpy as np
import pickle
import tensorflow as tf

def load_image(file_path):
	"""Helper for a mapping. No need for a direct call."""
	image = tf.io.read_file(file_path)
	image = tf.image.decode_png(image, channels=3)
	image = tf.cast(image, tf.float32) / 255.0
	return image

def load_data(set_name:str, size:int|str) -> np.ndarray:
	"""
	Loads a single dataset. No labels.
	Returns the numpy array of shape (N, length, width, channels).
	"""

	#Pre-input sanitization. Must know possible sizes.
	possible_sizes = []
	for folder in os.listdir("../data/"):
		try:
			possible_sizes.append(int(folder))
		except ValueError:
			continue

	#Some input sanitization
	assert(set_name == "train" or set_name == "test")
	assert(int(size) in possible_sizes)

	#Some locals determined by function args
	image_dir = "../data/" + str(size) + "/" + set_name
	image_size = (size, size)
	N = len(os.listdir(image_dir))

	#Create a list of file paths for all png images in the directory
	file_paths = tf.io.gfile.glob(image_dir + "/*.png")

	#Sort - otherwise they don't stay in sync with labels
	file_paths = sorted(file_paths, key=lambda x: int(x.split("\\")[-1].split(".")[0]))

	#Create a TensorFlow dataset from the file paths
	dataset = tf.data.Dataset.from_tensor_slices(file_paths)

	#Map the dataset from each file to actual png format
	dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

	dataset = dataset.batch(N)#(full epoch)

	return next(iter(dataset)).numpy()

def load_labels(set_name:str) -> list:
	"""Loads the list of labels that is kept in sync with every dataset"""

	#Some input sanitization
	assert(set_name == "train" or set_name == "test")

	with open("../data/labels/" + set_name + ".pickle", "rb") as f:
		return pickle.load(f)
