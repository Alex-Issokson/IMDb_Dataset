import argparse
import numpy as np
from sklearn import svm
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDRegressor
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


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



    # lasso_params = {'alpha':[0.005, .0075, .0089, .0095, 0.011, 0.012, 0.01, 0.02, 0.030, 0.04, 0.05, 0.06, 0.07, 0.08, .09, .1, .15, .25, .5, .75, 1, 5, 10]}
    # lasso = GridSearchCV(Lasso(), param_grid=lasso_params, scoring='neg_mean_absolute_error', cv=5)
    # results = lasso.fit(xTrain, yTrain)
    # print('MAE: %.5f' % results.best_score_)
    # print('Config: %s' % results.best_params_)
    
    finalModel = Lasso(alpha=0.0089)
    
    
    finalModel.fit(xTrain, yTrain)
    
    yHat = finalModel.predict(xTest)
    r2 = metrics.r2_score(yTest, yHat.ravel())
    mse = metrics.mean_squared_error(yTest, yHat.ravel())
    print("Lasso R2 value:", r2)
    print("Lasso MSE value:", mse)

    Coefficient = pd.DataFrame(data={'Attribute': xTrain.columns,'Coefficient': finalModel.coef_})
    Coefficient = Coefficient.sort_values(by='Coefficient', ascending=False)
    plt.bar(x=Coefficient['Attribute'], height=Coefficient['Coefficient'])
    plt.title('Lasso Coefficients', size=20)
    plt.xticks(rotation='vertical')
    plt.show()
    
    #1b. Calculate the PCA until 95 variance of original and fit training data and transform test data
  
    # lasso_params = {'alpha':[0.005, .0075, .0089, .0095, 0.011, 0.012, 0.01, 0.02, 0.030, 0.04, 0.05, 0.06, 0.07, 0.08, .09, .1, .15, .25, .5, .75, 1, 5, 10]}
    # lasso = GridSearchCV(Lasso(), param_grid=lasso_params, scoring='neg_mean_absolute_error', cv=5)
    # pca_results = lasso.fit(pcaTrain, yTrain)
    # print('PCA MAE: %.5f' % pca_results.best_score_)
    # print('PCA Config: %s' % pca_results.best_params_)
    
    finalModel = Lasso(alpha=0.0089)
    finalModel.fit(pcaTrain, yTrain)
    pca_yHat = finalModel.predict(pcaTest)
    r2 = metrics.r2_score(yTest, pca_yHat.ravel())
    print("PCA R2 value:", r2)
    mse = metrics.mean_squared_error(yTest, pca_yHat.ravel())
    print("PCA MSE value:", mse)
    
 
    
    # param = {'coef0' : [0, 0.01,0.5, .1],'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [3,8],'gamma' : ('auto','scale')}
    # grid_search = GridSearchCV(estimator = svm.SVR(), param_grid = param, scoring='neg_mean_absolute_error',
    #                   cv = 3, n_jobs=5, verbose=2)
    # results = grid_search.fit(xTrain, yTrain)
    # print('SVR MAE: %.5f' % results.best_score_)
    # print('SVR Config: %s' % results.best_params_)
    
    regr = svm.SVR(C=1, coef0=0, degree=3, gamma='auto', kernel='rbf')
    regr.fit(xTrain, yTrain)
    svr_yHat = regr.predict(xTest)
    r2 = metrics.r2_score(yTest, svr_yHat.ravel())
    mse = metrics.mean_squared_error(yTest, svr_yHat.ravel())
    print("Svr R2 value:", r2)
    print("Svr MSE value:", mse)
    
    regr = svm.SVR(C=1, coef0=0, degree=3, gamma='auto', kernel='rbf')
    regr.fit(pcaTrain, yTrain)
    svr_yHat = regr.predict(pcaTest)
    r2 = metrics.r2_score(yTest, svr_yHat.ravel())
    mse = metrics.mean_squared_error(yTest, svr_yHat.ravel())
    print("PCA Svr R2 value:", r2)
    print("PCA Svr MSE value:", mse)
    
    # elas_params = {'alpha' : [.000001, .00001, .001, .01, .1, 0.0, 1.0, 10.0, 100.0], 'l1_ratio': [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    # elas_gs = GridSearchCV(ElasticNet(), param_grid=elas_params, scoring='neg_mean_absolute_error', cv=5)
    # elas_res = elas_gs.fit(xTrain, yTrain)
    # print('ElasMAE: %.5f' % elas_res.best_score_)
    # print('PCA Config: %s' % elas_res.best_params_)
    elast = ElasticNet(alpha=0.01, l1_ratio= 0.8)
    elast.fit(xTrain, yTrain)
    elast_yHat = elast.predict(xTest)
    r2 = metrics.r2_score(yTest, elast_yHat.ravel())
    print("Elastic Net R2 value:", r2)
    mse = metrics.mean_squared_error(yTest, elast_yHat.ravel())
    print("Elastic Net MSE value:", mse)
    
    


if __name__ == "__main__":
    main()
