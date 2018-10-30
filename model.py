from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout,Concatenate
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet201
from keras.applications.xception import Xception
# from keras.layers import BatchNormalization
def CNN2D(n_classes, image_shape):
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)

    model = Sequential()

    for layer in vgg16.layers:
        layer.trainable = False
        model.add(layer)

    model.add(Flatten(name='block6_flatten'))
    model.add(Dense(512, activation='relu', name='block7_dense1'))
    model.add(Dropout(0.5, name='block7_dropout1'))
    model.add(Dense(256, activation='relu', name='block7_dense2'))
    model.add(Dropout(0.5, name='block7_dropout2'))
    model.add(Dense(n_classes, activation='softmax', name='block7_dense3'))

    return model
