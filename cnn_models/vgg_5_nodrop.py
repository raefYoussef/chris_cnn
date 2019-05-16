# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense


class VGG_5_NoDrop:
    @staticmethod
    def build(width, height, depth, classes):
        # init network
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # CONV => RELU => POOL
        # output is 32x32x32
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(3, 3)))

        # 2*(CONV => RELU) => POOL
        # Double number of filters to 64
        # output is 16x16x64
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 2*(CONV => RELU) => POOL
        # Double number of filters to 128
        # output is 8x8x128
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
