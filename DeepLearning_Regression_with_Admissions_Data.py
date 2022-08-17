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