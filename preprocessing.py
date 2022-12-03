import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d

def calculate_graph_Pearson(xFeat, y):
    #calculate pearson
    graph = 0
    xFeat["Rating"] = y
    Pearson = xFeat.corr()
    
    #graph heatmpa
    ax = sns.heatmap(Pearson, xticklabels=1, yticklabels=1)
    plt.show()
    #Selecting highly correlated features
    cols = list(Pearson.columns)
    for col in cols:
        cor_target = abs(Pearson[col])
        relevant_features = cor_target[cor_target>0.5]
        print(relevant_features)
        print("")
    
    cor_target = abs(Pearson["Rating"])
    relevant_features = cor_target
    print(relevant_features)

def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    #Select only these features
    cols = ['Rating', 'PG', 'Animation', 'Family', 'cast_total_facebook_likes', 'actor_3_facebook_likes', 'Black and White', 'PG-13',]
    df = df.drop(columns=cols)
    return df
def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # Normalize the data using standad
    
    stdScale = StandardScaler().fit(trainDF)
    cols = list(trainDF.columns)
    trainDF[cols]=stdScale.transform(trainDF)
    testDF[cols]=stdScale.transform(testDF)
    return trainDF, testDF

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainFile",
                        default="xtrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="xtest.csv",
                        help="filename of the test data")
    parser.add_argument("--ytrainFile",
                        default="ytrain.csv",
                        help="filename of the training data")
    parser.add_argument("--ytestFile",
                        default="ytest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    yTrain = pd.read_csv(args.ytrainFile)
    yTest = pd.read_csv(args.ytestFile)
    
    # Graph features pearson
    calculate_graph_Pearson(xTrain, yTrain)
    calculate_graph_Pearson(xTest, yTest)
    xTrain = select_features(xTrain)
    xTest = select_features(xTest)
    xTrainTr, xTestTr = preprocess_data(xTrain, xTest)
    
    
    sklearn_pca = PCA(n_components=0.95)
    X_pca=sklearn_pca.fit_transform(xTrainTr)
    testPca = sklearn_pca.transform(xTestTr)
    
    print("PCA dataset:")
    print(X_pca)
    print("PCA components:")
    print(sklearn_pca.components_)
    print("PCA variance:")
    print(sklearn_pca.explained_variance_)
    print("PCA variance ratio:")
    print(sklearn_pca.explained_variance_ratio_)
    
    # save it to csv
    xTrainTr.to_csv("prepro_xtrain.csv", index=False)
    xTestTr.to_csv("prepro_xtest.csv", index=False)
    
    trainPca = pd.DataFrame(X_pca)
    testPca = pd.DataFrame(testPca)
    trainPca.to_csv("pca_xtrain.csv", index=False)
    testPca.to_csv("pca_xtest.csv", index=False)
    
    
    
   


if __name__ == "__main__":
    main()
