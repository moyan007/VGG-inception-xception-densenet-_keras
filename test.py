import random
import os
from dataset import Dataset
from model import CNN2D
from keras.models import load_model
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

batch_size = 64
image_shape = (200, 200, 3)
n_epochs = 100

# create dataset
data = Dataset(batch_size, image_shape)

# create model
model = load_model('./log/train/model_epochs_' + str(n_epochs) + '.h5')
model.summary()

# load weights
model.load_weights('./log/train/weights.hdf5')

X_temp = None
X_all = []
Y_pred = None
Y_all = []
for sample in (data.test):
    X_temp = sample
    Y_pred = model.predict(np.array(data.load_sample(X_temp)))[0].argmax()
    X_all.append(X_temp[0])
    Y_all.append(Y_pred+1)

dataframe = pd.DataFrame({'filename':X_all, 'type':Y_all})
dataframe.to_csv('test.csv',index = False)





