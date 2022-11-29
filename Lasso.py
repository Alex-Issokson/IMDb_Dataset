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
from xgboost import XGBRegressor


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
    # pca_results = lasso.fit(pcaTrain, yTrain)
    # print('PCA MAE: %.5f' % pca_results.best_score_)
    # print('PCA Config: %s' % pca_results.best_params_)
    r2array = []
    msearray = []
    models = ["PCA Lasso", "PCA ElasticNet", "Elasticnet", "Lasso", "PCA XGB", "SVR", "PCA SVR", "XGB"]
    
    
    finalModel = Lasso(alpha=0.0089)
    finalModel.fit(pcaTrain, yTrain)
    pca_yHat = finalModel.predict(pcaTest)
    r2 = metrics.r2_score(yTest, pca_yHat.ravel())
    print("PCA R2 value:", r2)
    mse = metrics.mean_squared_error(yTest, pca_yHat.ravel())
    print("PCA MSE value:", mse)
    
    r2array.append(r2)
    msearray.append(mse)
    
    # elas_params = {'alpha' : [.000001, .00001, .001, .01, .1, 0.0, 1.0, 10.0, 100.0], 'l1_ratio': [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    # elas_gs = GridSearchCV(ElasticNet(), param_grid=elas_params, scoring='neg_mean_absolute_error', cv=5)
    # elas_res = elas_gs.fit(xTrain, yTrain)
    # print('ElasMAE: %.5f' % elas_res.best_score_)
    # print('PCA Config: %s' % elas_res.best_params_)
    elast = ElasticNet(alpha=0.01, l1_ratio= 0.8)
    elast.fit(pcaTrain, yTrain)
    elast_yHat = elast.predict(pcaTest)
    r2 = metrics.r2_score(yTest, elast_yHat.ravel())
    print("PCA Elastic Net R2 value:", r2)
    mse = metrics.mean_squared_error(yTest, elast_yHat.ravel())
    print("PCA Elastic Net MSE value:", mse)
    r2array.append(r2)
    msearray.append(mse)

    
    elast.fit(xTrain, yTrain)
    elast_yHat = elast.predict(xTest)
    r2 = metrics.r2_score(yTest, elast_yHat.ravel())
    print("Elastic Net R2 value:", r2)
    mse = metrics.mean_squared_error(yTest, elast_yHat.ravel())
    print("Elastic Net MSE value:", mse)
    r2array.append(r2)
    msearray.append(mse)
    
 
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
    
    r2array.append(r2)
    msearray.append(mse)

    model = XGBRegressor(colsample_bytree = 1, gamma = 7, max_depth=15, min_child_weight=9, n_estimators=180, reg_alpha=40, reg_lambda=1.0)
    model.fit(pcaTrain, yTrain)
    xgb_yHat = model.predict(pcaTest)
    r2 = metrics.r2_score(yTest, xgb_yHat.ravel())
    mse = metrics.mean_squared_error(yTest, xgb_yHat.ravel())
    print("PCA XGB R2 value:", r2)
    print("PCA XGB MSE value:", mse)
    r2array.append(r2)
    msearray.append(mse)

 
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
    
    r2array.append(r2)
    msearray.append(mse)
    
    regr = svm.SVR(C=1, coef0=0, degree=3, gamma='auto', kernel='rbf')
    regr.fit(pcaTrain, yTrain)
    svr_yHat = regr.predict(pcaTest)
    r2 = metrics.r2_score(yTest, svr_yHat.ravel())
    mse = metrics.mean_squared_error(yTest, svr_yHat.ravel())
    print("PCA Svr R2 value:", r2)
    print("PCA Svr MSE value:", mse)
    
    
    r2array.append(r2)
    msearray.append(mse)
    

    
    # param={'max_depth':  range(3, 1, 3),
    #     'gamma': range(1,9,3),
    #     'reg_alpha' : range(40,180,20),
    #     'reg_lambda' :  np.linspace(0,1,11),
    #     'colsample_bytree' : np.lin(space.5,1,6),
    #     'min_child_weight' : range(0, 10, 1),
    #     'n_estimators': [180],
    # }
    # gsearch1 = GridSearchCV(estimator = XGBRegressor(), 
    #     param_grid = param, scoring='neg_mean_absolute_error',n_jobs=4, cv=3, verbose=2)
    # gsearch1.fit(xTrain, yTrain)
    # print(gsearch1.best_params_, gsearch1.best_score_)
    
    model = XGBRegressor(colsample_bytree = 1, gamma = 7, max_depth=15, min_child_weight=9, n_estimators=180, reg_alpha=40, reg_lambda=1.0)
    model.fit(xTrain, yTrain)
    xgb_yHat = model.predict(xTest)
    r2 = metrics.r2_score(yTest, xgb_yHat.ravel())
    mse = metrics.mean_squared_error(yTest, xgb_yHat.ravel())
    print("XGB R2 value:", r2)
    print("XGB MSE value:", mse)
    r2array.append(r2)
    msearray.append(mse)





    trainr2 = []
    trainmse = []
    finalModel = Lasso(alpha=0.0089)
    finalModel.fit(pcaTrain, yTrain)
    pca_yHat = finalModel.predict(pcaTrain)
    r2 = metrics.r2_score(yTrain, pca_yHat.ravel())
    print("Train PCA R2 value:", r2)
    mse = metrics.mean_squared_error(yTrain, pca_yHat.ravel())
    print("Train PCA MSE value:", mse)
    
    trainr2.append(r2)
    trainmse.append(mse)
    
    # elas_params = {'alpha' : [.000001, .00001, .001, .01, .1, 0.0, 1.0, 10.0, 100.0], 'l1_ratio': [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
    # elas_gs = GridSearchCV(ElasticNet(), param_grid=elas_params, scoring='neg_mean_absolute_error', cv=5)
    # elas_res = elas_gs.fit(xTrain, yTrain)
    # print('ElasMAE: %.5f' % elas_res.best_score_)
    # print('PCA Config: %s' % elas_res.best_params_)
    elast = ElasticNet(alpha=0.01, l1_ratio= 0.8)
    elast.fit(pcaTrain, yTrain)
    elast_yHat = elast.predict(pcaTrain)
    r2 = metrics.r2_score(yTrain, elast_yHat.ravel())
    print("Train PCA Elastic Net R2 value:", r2)
    mse = metrics.mean_squared_error(yTrain, elast_yHat.ravel())
    print("Train PCA Elastic Net MSE value:", mse)
    trainr2.append(r2)
    trainmse.append(mse)

    
    elast.fit(xTrain, yTrain)
    elast_yHat = elast.predict(xTrain)
    r2 = metrics.r2_score(yTrain, elast_yHat.ravel())
    print("Train Elastic Net R2 value:", r2)
    mse = metrics.mean_squared_error(yTrain, elast_yHat.ravel())
    print("Train Elastic Net MSE value:", mse)
    trainr2.append(r2)
    trainmse.append(mse)
    
 
    # lasso_params = {'alpha':[0.005, .0075, .0089, .0095, 0.011, 0.012, 0.01, 0.02, 0.030, 0.04, 0.05, 0.06, 0.07, 0.08, .09, .1, .15, .25, .5, .75, 1, 5, 10]}
    # lasso = GridSearchCV(Lasso(), param_grid=lasso_params, scoring='neg_mean_absolute_error', cv=5)
    # results = lasso.fit(xTrain, yTrain)
    # print('MAE: %.5f' % results.best_score_)
    # print('Config: %s' % results.best_params_)
    
    
    finalModel = Lasso(alpha=0.0089)
    finalModel.fit(xTrain, yTrain)
    
    yHat = finalModel.predict(xTrain)
    r2 = metrics.r2_score(yTrain, yHat.ravel())
    mse = metrics.mean_squared_error(yTrain, yHat.ravel())
    print("Train Lasso R2 value:", r2)
    print("Train Lasso MSE value:", mse)
    
    trainr2.append(r2)
    trainmse.append(mse)

    model = XGBRegressor(colsample_bytree = 1, gamma = 7, max_depth=15, min_child_weight=9, n_estimators=180, reg_alpha=40, reg_lambda=1.0)
    model.fit(pcaTrain, yTrain)
    xgb_yHat = model.predict(pcaTrain)
    r2 = metrics.r2_score(yTrain, xgb_yHat.ravel())
    mse = metrics.mean_squared_error(yTrain, xgb_yHat.ravel())
    print("Train PCA XGB R2 value:", r2)
    print("Train PCA XGB MSE value:", mse)
    trainr2.append(r2)
    trainmse.append(mse)

 
    # param = {'coef0' : [0, 0.01,0.5, .1],'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [3,8],'gamma' : ('auto','scale')}
    # grid_search = GridSearchCV(estimator = svm.SVR(), param_grid = param, scoring='neg_mean_absolute_error',
    #                   cv = 3, n_jobs=5, verbose=2)
    # results = grid_search.fit(xTrain, yTrain)
    # print('SVR MAE: %.5f' % results.best_score_)
    # print('SVR Config: %s' % results.best_params_)
    
    regr = svm.SVR(C=1, coef0=0, degree=3, gamma='auto', kernel='rbf')
    regr.fit(xTrain, yTrain)
    svr_yHat = regr.predict(xTrain)
    r2 = metrics.r2_score(yTrain, svr_yHat.ravel())
    mse = metrics.mean_squared_error(yTrain, svr_yHat.ravel())
    print("Train Svr R2 value:", r2)
    print("Train Svr MSE value:", mse)
    
    trainr2.append(r2)
    trainmse.append(mse)
    
    regr = svm.SVR(C=1, coef0=0, degree=3, gamma='auto', kernel='rbf')
    regr.fit(pcaTrain, yTrain)
    svr_yHat = regr.predict(pcaTrain)
    r2 = metrics.r2_score(yTrain, svr_yHat.ravel())
    mse = metrics.mean_squared_error(yTrain, svr_yHat.ravel())
    print("Train PCA Svr R2 value:", r2)
    print("Train PCA Svr MSE value:", mse)
    
    
    trainr2.append(r2)
    trainmse.append(mse)
    
    model.fit(xTrain, yTrain)
    xgbTrain_yHat = model.predict(xTrain)
    r2 = metrics.r2_score(yTrain, xgbTrain_yHat.ravel())
    mse = metrics.mean_squared_error(yTrain, xgbTrain_yHat.ravel())
    print("Train XGB R2 value:", r2)
    print("Train XGB MSE value:", mse)
    trainr2.append(r2)
    trainmse.append(mse)
    
    

    
    
    Coefficient = pd.DataFrame(data={'Attribute': xTrain.columns,'Coefficient': finalModel.coef_})
    Coefficient = Coefficient.sort_values(by='Coefficient', ascending=False)
    plt.bar(x=Coefficient['Attribute'], height=Coefficient['Coefficient'])
    plt.title('Lasso Coefficients', size=20)
    plt.xticks(rotation='vertical')
    plt.show()
    

    x = np.array([0,1,2,3,4,5,6,7])
    plt.xticks(x, models)
    plt.plot(x, msearray)
    plt.plot(x, trainmse)
    plt.legend(['Testing MSE', 'Training MSE'])
    plt.title("MSE")
    plt.show()
    
    x = np.array([0,1,2,3,4,5,6,7])
    plt.xticks(x, models)
    plt.plot(x, r2array)
    plt.plot(x, trainr2)
    plt.legend(['Testing R2', 'Training R2'])
    plt.title("R2")
    plt.show()



if __name__ == "__main__":
    main()
