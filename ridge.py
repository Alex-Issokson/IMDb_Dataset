import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

xtrain = pd.read_csv("xtrain.csv")
ytrain = pd.read_csv("ytrain.csv")

xTrain, xTest = train_test_split(xtrain, test_size = 0.2, random_state = 1)
yTrain, yTest = train_test_split(ytrain, test_size = 0.2, random_state = 1)

predict = ["duration", "budget"]
target = "imdb_score"

# ridge = Ridge(alpha = alpha)
# ridge.fit(x[predict], y)
# ridge.coef_
# ridge.intercept_
# sklearn_predictions = ridge.predict(testX[predict])
#predict - sklearn_predictions

def ridge_fit(train, predict, target, alpha):
    x = train[predict].copy()
    y = train[[target]].copy()

    x_mean = x.mean()
    x_std = x.std()

    x = (x - x_mean) / x_std
    x["intercept"] = 1
    x = x[["intercept"] + predict]

    penalty = alpha * np.identity(x.shape[1])
    penalty[0][0] = 0

    B = np.linalg.inv(x.T @ x + penalty) @ x.T @ y #get ridge regression coefficients
    B.index = ["intercept", "duration", "budget"]
    return B, x_mean, x_std

def ridge_predict(test, predict, x_mean, x_std, B):
    xTest = test[predict]
    xTest = (xTest - x_mean) / x_std
    xTest["intercept"] = 1
    xTest = xTest[["intercept"] + predict]

    predict = xTest @ B
    return predict

errors = []
alphas = [10**i for i in range (-2, 4)]

for alpha in alphas:
    B, x_mean, x_std = ridge_fit(xTrain, predict, target, alpha)
    predict = ridge_predict(xTest, predict, x_mean, x_std, B)
    errors.append(mean_absolute_error(xTest[target], predict))



