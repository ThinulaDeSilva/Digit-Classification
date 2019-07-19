import tensorflow as tf
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import time


def build_dataset():
    (X_train, Y_train),(X_test, Y_test) = keras.datasets.mnist.load_data()

    # the arrays need to be 4 dimensions to work in keras
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # convert data to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # the data is normalized by dividing the RGB values to the max RGB value
    X_train /= 255
    X_test /= 255

    return X_train,Y_train,X_test,Y_test


def init_model(input_shape):
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(56, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))
    return model


start = time.time()
X_train, Y_train, X_test, Y_test = build_dataset()
input_shape = (28, 28, 1)
model = init_model(input_shape)
end = time.time()
initTime = end-start

# Compile and fit the model on our training sets
start = time.time()
optimizer = SGD(lr=0.01, momentum=0.9, decay=0.01)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=X_train, y=Y_train, epochs=10, verbose=2)
end = time.time()
compTime = end-start

# Evaluate the model accuracy
start = time.time()
results = model.evaluate(X_test, Y_test, verbose=0)
end = time.time()
evalTime = end-start

print("Initialization took %.3f seconds" %initTime)
print("Compilation took %.3f seconds" %compTime)
print("Evaluation took %.3f seconds" %evalTime)
print(results)

model.save("Model.h5")