import csv

import cv2
from sklearn.utils import shuffle

import matplotlib.image as mpimg
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split

dataset_path = "dataset_triple.csv"
samples = []


def add_to_samples(csv_filepath, samples):
    with open(csv_filepath, encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


samples = add_to_samples(dataset_path, samples)

train_samples, test_samples = train_test_split(samples, test_size=0.1)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            labels = []
            for batch_sample in batch_samples:
                try:
                    label = batch_sample[1]
                    name = format(batch_sample[0])
                except IndexError:
                    continue
                center_image = cv2.imread(name)
                images.append(center_image)
                labels.append(label)

            x_train = np.array(images)
            y_train = np.array(labels)
            yield shuffle(x_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(test_samples, batch_size=32)

input_shape = (28, 28, 3)
chanDim = -1
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# use a *softmax* activation for single-label classification
# and *sigmoid* activation for multi-label classification
model.add(Dense())
model.add(Activation("softmax"))

model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=32),
    validation_data=(testX, testY),
    steps_per_epoch=len(train_samples),
    epochs=nb_epoch, verbose=1)

print("Model summary:\n", model.summary())

## 4. Train model
batch_size = 32
nb_epoch = 10
# Train model using generator

print(train_generator)
print(validation_generator)

model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(test_samples), nb_epoch=nb_epoch)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model_with_weights.h5")
model.save_weights("model_weights_only.h5")
print("Saved model to disk")
