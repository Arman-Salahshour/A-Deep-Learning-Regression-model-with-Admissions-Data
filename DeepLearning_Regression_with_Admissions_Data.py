"""Add Important Libraries"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import product

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer


"""Load the dataset"""
data = pd.read_csv('admissions_data.csv')

"""Investigate the dataset"""
print(data.info())
print(data.describe())

"""divide features from labels"""
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]
columns = features.columns

"""Convert all of the columns' datatype to float64"""
features = features.apply(lambda col: col.astype('float64'))


"""Standardize all of the features"""
scaler = StandardScaler()
normalized_data = scaler.fit_transform(features)
features = pd.DataFrame(data = normalized_data, columns = columns)
print(features.describe())


"""Anomaly detection: removing outliers by z-score test"""
for col in features.columns:
  indexes = features[abs(features[col]) > 3].index
  features.drop(indexes, inplace = True)


"""Divide features and labels for train and test parts"""
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20)




"""Create ANN model"""
def design_model(feature_numbers, learning_rate, layers_modification = []):
    #Create Sequential model object
    model = Sequential()

    #Initialize input layer
    input = layers.InputLayer(input_shape = (feature_numbers,))
    model.add(input)


    if len(layers_modification) == 0:
        #Initialize hidden layer
        hidden_layer = layers.Dense(units = 16, activation = 'relu')

        #Add hidden layers
        model.add(hidden_layer)
        model.add(layers.Dropout(0.2))


    else:
        if len(layers_modification) !=3 :
            raise ValueError('layers_modification list must have three items')

        #Determine the number of layers and the number of units in each of them
        n_layers, first_layer_nodes, last_layer_nodes = layers_modification
        neurons_counter = math.ceil((last_layer_nodes - first_layer_nodes) / n_layers)

        #Add layers
        for i in range(first_layer_nodes, last_layer_nodes, neurons_counter):
            model.add(layers.Dense(units = i))
            if np.random.rand() > 0.5:
                model.add(layers.Dropout(np.random.rand() * 0.3 + 0.2))

        model.add(layers.Dense(units = abs(neurons_counter), activation = 'relu'))
        

    #Initialize optimizer
    optimizer = Adam(learning_rate = learning_rate)

    #Initialize output layer
    output = layers.Dense(units = 1, activation = 'sigmoid')
    model.add(output)

    #Compile model and set loss, metrics and optimizer argumans
    model.compile(loss = 'mse', metrics = ['mae'], optimizer = optimizer)

    return model


"""Design a model through design_model without layers_modification argument"""
model = design_model(len(columns), 0.01)
print(model.summary())



"""Initialize callback"""
early = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 40, verbose = 0 )

"""Fit ANN model"""
model.fit(features_train, labels_train, epochs = 500, callbacks = [early], batch_size = 5, validation_split = 0.2, verbose = 1)



"""Create a Dummy Regressor model to find the least accuracy that model should have"""
dummy = DummyRegressor()

"""Fit dummy model"""
dummy.fit(features_train, labels_train)


"""Achieve dummy score"""
labels_predicted = dummy.predict(features_test)
dummy_score = mean_absolute_error(labels_test, labels_predicted)
print(f'dummy score (MAE): {dummy_score}')

"""Achieve achieve ANN model score before tuning"""
res_mse, res_mae = model.evaluate(features_test, labels_test, verbose = 0)
print(f'ANN model score (MAE) before tuning: {res_mae}')


"""Start tuning the ANN model"""
n_layer = [2,3]
fist_layer_nodes = [64, 32, 16]
last_layer_nodes = [4]

hidden_layers = list(product(n_layer, fist_layer_nodes, last_layer_nodes))

regressor_model = KerasRegressor(build_fn = design_model)
distributions = dict(epochs = [i for i in range(100,150)],
                    batch_size = [i for i in range(1,10)],
                    feature_numbers = [len(columns),] , learning_rate = [0.01,], layers_modification = hidden_layers)

clf = RandomizedSearchCV(estimator = regressor_model, param_distributions = distributions, n_jobs = -1)
search = clf.fit(features_train, labels_train)

print(search.best_params_)