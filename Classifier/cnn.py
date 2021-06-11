from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
from skimage.io import imread_collection
import numpy as np
from keras.preprocessing.image import ImageDataGenerator



drPath = "../ProcessedImages/DR/*.jpg"
healthyPath = "../ProcessedImages/Healthy/*.jpg"

datagen = ImageDataGenerator(rotation_range=90, horizontal_flip=True)

drCollection = imread_collection(drPath)
healthyCollection = imread_collection(healthyPath)


dataset = np.concatenate((drCollection, healthyCollection))
labels = np.array([0 for i in range(len(drCollection))] +
                  [1 for i in range(len(healthyCollection))])

x_train, x_test, y_train, y_test = train_test_split(
    dataset, labels, test_size=0.2, random_state=42)

# Build Input Vector
x_train = x_train.reshape(x_train.shape[0], 1728, 1700, 1)
x_test = x_test.reshape(x_test.shape[0], 1728, 1700, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))
it = datagen.flow(x_train, y_train)

# Normalizing the data to help with the training
x_train /= 255
x_test /= 255

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(50, kernel_size=(3, 3), strides=(4, 4),
                 padding='same', activation='relu', input_shape=(1728, 1700, 1)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3, 3), strides=(
    4, 4), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3, 3), strides=(
    4, 4), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(500, kernel_size=(3, 3), strides=(
    4, 4), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# flatten output of conv
model.add(Flatten())
# hidden layer
model.add(Dense(10000, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
# output layer
model.add(Dense(1, activation='sigmoid'))

# compiling the sequential model
model.compile(loss='binary_crossentropy', metrics=[
              'accuracy'], optimizer='adam')

# model.fit(x_train, y_train, batch_size=1, epochs=10, validation_data=(x_test, y_test))
model.fit(it, validation_data=(x_test, y_test), epochs=20)
