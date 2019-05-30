import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from pathlib import Path

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


class NeuralNetwork:

    network = None
    network_weights = None

    def __init__(self):
        self.network = Path('Network/network.json')
        self.network_weights = Path("Network/network.h5")
        
        if self.network.is_file():
            print("Ucitan fajl sa sacuvanim modelom")
            self.load()#ovde sam stao, predstoji hard_work
        else:
            print("Nema sacuvanih modela")
            self.create()
            self.train()
            self.save()

    def load(self):
        # load json and create model
        json_file = open('Network/network.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.network = model_from_json(loaded_model_json)
        # load weights into new model
        self.network.load_weights('Network/network.h5')
        print("Loaded model from disk...")

    def create(self):
        # Creating a Sequential Model and adding the layers
        self.network = Sequential()
        input_shape = (28, 28, 1)
        self.network.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
        self.network.add(MaxPooling2D(pool_size=(2, 2)))
        # Flattening the 2D arrays for fully connected layers
        self.network.add(Flatten())
        self.network.add(Dense(128, activation=tf.nn.relu))
        self.network.add(Dropout(0.2))
        self.network.add(Dense(10, activation=tf.nn.softmax))

        # Compiling and Fitting the Model
        self.network.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    def train(self):
        # --------- part 1 - Downloading the Mnist Data -------------------------------
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        print('uspeo da ocita dataset')

        x_train.shape

        # --------- part 4 - Reshaping and Normalizing the Images ----------------
        # Reshaping the array to 4-dims so that it can work with the Keras API
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        # Making sure that the values are float so that we can get decimal points after division
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print('Number of images in x_train', x_train.shape[0])
        print('Number of images in x_test', x_test.shape[0])

        self.network.fit(x=x_train, y=y_train, epochs=10)

        self.evaluate_network(x_test, y_test)

    def evaluate_network(self, x_test, y_test):
        # -------- part 7 - Evaluating the Model --------------------------------------
        self.network.evaluate(x_test, y_test)

        # -------- part 8 - Testiranje posle evaluacije -------------------------------
        image_index = 4444
        img_rows = 28
        img_cols = 28
        plt.imshow(x_test[image_index].reshape(
            img_rows, img_cols))
        pred = self.network.predict(
            x_test[image_index].reshape(1, img_rows, img_cols, 1))
        print(pred.argmax())

        print('tu sam i nema gresaka')

    def save(self):
        print("Cuvanje modela mreze u toku...")
        # serialize model to JSON
        network_json = self.network.to_json()
        with open("Network/network.json", "w") as json_file:
            json_file.write(network_json)
        # serialize weights to HDF5
        self.network.save_weights("Network/network.h5", overwrite=True)
        print("Saved model to disk")

    # 3 - funkcija koja vrsi predikciju broja
    def predict_number(self, image_part, img_rows, img_cols):
        number_region_gray = cv2.cvtColor(image_part, cv2.COLOR_BGR2GRAY)
        number_region_reshaped = number_region_gray.reshape(1, img_rows, img_cols, 1)
        #cv2.show("pre predikcije", number_region_reshaped)
        prediction = self.network.predict(number_region_reshaped)
        return prediction.argmax()
