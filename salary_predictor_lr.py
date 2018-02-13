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
                    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.linear_model import LinearRegression
classifier = LinearRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(y_test, y_pred)