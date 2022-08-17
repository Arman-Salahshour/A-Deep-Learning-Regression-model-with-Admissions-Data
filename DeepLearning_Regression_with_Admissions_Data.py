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
  '''Create Sequential model object'''
  model = Sequential()

  return model