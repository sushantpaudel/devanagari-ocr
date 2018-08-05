import csv
from random import shuffle

import matplotlib.image as mpimg
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

dataset_path = "dataset.csv"
samples = []


def add_to_samples(csv_filepath, samples):
    with open(csv_filepath) as csvfile:
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
                label = batch_sample[1]
                name = format(batch_sample[0])
                center_image = mpimg.imread(name)
                images.append(center_image)
                labels.append(label)
            x_train = np.array(images)
            y_train = np.array(labels)
            yield shuffle(x_train, y_train)


train_generator = generator(train_samples, batch_size=32)

input_shape = (28, 28, 1)
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print("Model summary:\n", model.summary())

## 4. Train model
batch_size = 32
nb_epoch = 50
# Train model using generator

print(train_generator)

model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    nb_epoch=nb_epoch)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model_with_weights.h5")
model.save_weights("model_weights_only.h5")
print("Saved model to disk")
