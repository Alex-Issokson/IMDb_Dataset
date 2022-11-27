import argparse
import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor



def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainXFile",
                        default="prepro_xtrain.csv",
                        help="filename of the x training data")
    parser.add_argument("--trainYFile",
                        default="ytrain.csv",
                        help="filename of the x training data")
    parser.add_argument("--testXFile",
                        default="prepro_xtest.csv",
                        help="filename of the y test data")
    parser.add_argument("--testYFile",
                    default="ytest.csv",
                    help="filename of the y test data")
    parser.add_argument("--pcaTest",
                        default="pca_xtest.csv",
                        help="filename of the y test data")
    parser.add_argument("--pcaTrain",
                    default="pca_xtrain.csv",
                    help="filename of the y test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainXFile)
    xTest = pd.read_csv(args.testXFile)
    pcaTrain = pd.read_csv(args.pcaTrain)
    pcaTest = pd.read_csv(args.pcaTest)
    yTrain = pd.read_csv(args.trainYFile)
    yTest = pd.read_csv(args.testYFile)

    param={'max_depth':  range(3, 18, 3),
        'gamma': range(1,9,3),
        'reg_alpha' : range(40,180,20),
        'reg_lambda' :  np.linspace(0,1,11),
        'colsample_bytree' : np.linspace(.5,1,6),
        'min_child_weight' : range(0, 10, 1),
        'n_estimators': [180],
    }
    gsearch1 = GridSearchCV(estimator = XGBRegressor(), 
        param_grid = param, scoring='neg_mean_absolute_error',n_jobs=4, cv=3, verbose=2)
    gsearch1.fit(xTrain, yTrain)
    print(gsearch1.best_params_, gsearch1.best_score_)
    model = XGBRegressor()

    model.fit(xTrain, yTrain)
    xgb_yHat = model.predict(xTest)
    r2 = metrics.r2_score(yTest, xgb_yHat.ravel())
    mse = metrics.mean_squared_error(yTest, xgb_yHat.ravel())
    print("XGB R2 value:", r2)
    print("XGB MSE value:", mse)
    
    
    estimators = [('XGB', model),('svr', SVR(C=1, coef0=0, degree=3, gamma='auto', kernel='rbf')) ]
    
    reg = StackingRegressor(
     estimators=estimators,
     final_estimator=RandomForestRegressor(n_estimators=100))
    reg.fit(xTrain, yTrain)
    yHat = reg.predict(xTest)
    r2 = metrics.r2_score(yTest, yHat.ravel())
    mse = metrics.mean_squared_error(yTest, yHat.ravel())
    print("Stack R2 value:", r2)
    print("Stack MSE value:", mse)

if __name__ == "__main__":
    main()
