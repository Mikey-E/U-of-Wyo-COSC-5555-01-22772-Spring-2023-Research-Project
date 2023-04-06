#Author: Michael Elgin (melgin@uwyo.edu)

#File to preprocess data from ../data/cifar-10-batches-py
#Assumes cifar-10-batches-py contains the extract from cifar-10-python.tar.gz
#Extracts the wanted 32x32 images to new folder(s) with their labels.
#This file should only need to be executed once.

import os
import pickle
import matplotlib.pyplot as plt

def main():

	DATA_PATH = "../data/cifar-10-batches-py/"
	IMG_PATH = "../data/32/"
	LABELS_PATH = "../data/labels/" #To match all datasets

	#asserts will stop the file from running if it need not be.
	assert(os.path.exists(DATA_PATH))
	assert(len(os.listdir(DATA_PATH)) != 0) #stuff inside
	assert(not os.path.exists(IMG_PATH))

	os.mkdir(IMG_PATH)
	os.mkdir(IMG_PATH + "train")
	os.mkdir(IMG_PATH + "test")
	os.mkdir(LABELS_PATH)

	#This list becomes the labels for horse and ship subsets of cifar-10
	train_label_subset = []
	test_label_subset = []
	
	#horses and ships now get gathered with their labels in sync
	batch_files = [DATA_PATH + 'data_batch_' + str(i) for i in range(1,6)] + [DATA_PATH + "test_batch"]
	item = 0
	for batch_file in batch_files:
		with open(batch_file, 'rb') as f:
			batch_dict = pickle.load(f, encoding='bytes')
		batch_data = batch_dict[b'data']
		batch_labels = batch_dict[b'labels']
		for i in range(len(batch_labels)):
			if (batch_labels[i] == 7) or (batch_labels[i] == 8): #horse or ship
				img = batch_data[i].reshape(3,32,32)
				img = img.transpose(1,2,0) #Ref binarystudy.com
				if (batch_file == DATA_PATH + "test_batch"):
					test_label_subset.append(batch_labels[i])
					plt.imsave(IMG_PATH + "test/" + str(item) + ".png", img)
				else:
					train_label_subset.append(batch_labels[i])
					plt.imsave(IMG_PATH + "train/" + str(item) + ".png", img)
				item += 1	

	assert(len(train_label_subset) == 10000) #5000 horse + 5000 ship
	assert(len(test_label_subset) == 2000) #1000 horse + 1000 ship

	with open(LABELS_PATH + "train.pickle", "wb") as f:
		pickle.dump(train_label_subset, f)
	with open(LABELS_PATH + "test.pickle", "wb") as f:
		pickle.dump(test_label_subset, f)

if __name__ == "__main__":
	main()
