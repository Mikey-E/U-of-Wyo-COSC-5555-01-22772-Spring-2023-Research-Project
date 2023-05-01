import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from load_data import load_data, load_labels

import numpy as np

train_data32 = load_data("train", 32)
test_data32 = load_data("test", 32)

train_data64 = load_data("train", 64)
test_data64 = load_data("test", 64)

train_labels = np.array(load_labels("train"))
test_labels = np.array(load_labels("test"))

def get_model():
    """example from which to base changes on"""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model

def get_model2(inputShape=train_data32.shape[1:]):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
  
    return model

def train_model(imgSize:str, batch_size=32, epochs=25):

    train_data = train_data32 if imgSize == "32" else train_data64
    test_data = test_data32 if imgSize == "32" else test_data64

    model = get_model2(inputShape=train_data.shape[1:])

    model.fit(
        train_data, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_data, test_labels),
        shuffle=True
    )

    model.save('../models/model2_' + imgSize + '.h5')

def score_model(path):

    print("Scoring model " + path)

    model = keras.models.load_model(path)

    test_data = test_data32 if "_32.h5" in path else test_data64

    scores = model.evaluate(test_data, test_labels, verbose=1)
    print('Test accuracy:', scores[1])

	#Confusion matrix values
    predictions = model.predict(test_data)
    thresholded_predictions = (predictions > 0.5).astype(int)
    thresholded_predictions = thresholded_predictions[:, 0]
    TP = sum([True for i in zip(test_labels, thresholded_predictions) if (i[0] == 1 and i[1] == 1)])
    TN = sum([True for i in zip(test_labels, thresholded_predictions) if (i[0] == 0 and i[1] == 0)])
    FP = sum([True for i in zip(test_labels, thresholded_predictions) if (i[0] == 0 and i[1] == 1)])
    FN = sum([True for i in zip(test_labels, thresholded_predictions) if (i[0] == 1 and i[1] == 0)])

    print("TP: " + str(TP))
    print("TN: " + str(TN))
    print("FP: " + str(FP))
    print("FN: " + str(FN))

	#Other metrics
    print("Precision: " + str(TP / (TP + FP)))
    print("Recall: " + str(TP / (TP + FN)))

def main():
#	train_model("32", batch_size=32, epochs=25)
#	train_model("64", batch_size=32, epochs=25)
    score_model("../models/model2_32.h5")
    score_model("../models/model2_64.h5")

if __name__ == "__main__":
	main()
