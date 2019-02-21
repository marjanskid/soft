import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# --------- part 1 - Downloading the Mnist Data -------------------------------
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print('uspeo da ocita dataset')

# --------- part 2 - Provera --------------------------------------------------
image_index = 7654              # za proveru je uzeta slika s indeksom 7654 iz trening seta
print(y_train[image_index])     # rezultat je 2
cv2.imshow('2', x_train[image_index])

# --------- part 3 - 
x_train.shape

# --------- part 4 - Reshaping and Normalizing the Images ----------------
# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# -------- part 5 - Building the Convolutional Neural Network ------------------
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

# -------- part 6 - Compiling and Fitting the Model ---------------------------
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=10)

# -------- part 7 - Evaluating the Model --------------------------------------
model.evaluate(x_test, y_test)

# -------- part 8 - Testiranje posle evaluacije -------------------------------
image_index = 4444
img_rows = 28
img_cols = 28
plt.imshow(x_test[image_index].reshape(img_rows, img_cols),cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, img_rows, img_cols, 1))
print(pred.argmax())

print(' ')
print('tu sam i nema gresaka')

