import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('nba_2017_popularity.csv')
X = dataset.iloc[:, :-1].values
X_ref = X[:,0].astype(float)

X = np.delete(X, 1, 1)
X = np.delete(X, 25, 1)
X = np.delete(X, 0, 1)
X = X.astype(dtype="float32")

y = dataset.iloc[:, 39].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 0, strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

model = Sequential()
model.add(Dense(80, input_dim=32, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, batch_size = 10, nb_epoch = 1000)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(y_test, y_pred)