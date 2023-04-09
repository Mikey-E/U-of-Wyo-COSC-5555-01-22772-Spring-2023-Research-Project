import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from load_data import load_data, load_labels

import numpy as np

train_data = load_data("train", 32)
test_data = load_data("test", 32)

train_label = np.array(load_labels("train"))
test_label = np.array(load_labels("test"))

def get_model():
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

def get_model2():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=train_data.shape[1:]))
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

def train_model():
    batch_size = 32
    epochs = 25
    model = get_model2()

    model.fit(
        train_data, train_label,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_data, test_label),
        shuffle=True
    )

    model.save('../models/model2.h5')

def load_model():
    model = keras.models.load_model('../models/model2.h5')

    scores = model.evaluate(test_data, test_label, verbose=1)
    print('Test accuracy:', scores[1])

def main():
	#train_model()
    #load_model()

if __name__ == "__main__":
	main()
