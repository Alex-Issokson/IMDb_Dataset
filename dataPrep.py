import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import json




def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--moviesFile",
                        default="movies.csv",
                        help="filename of the x training data")
    parser.add_argument("--creditFile",
                        default="credits.csv",
                        help="filename of the x training data")

    args = parser.parse_args()
    # load the train and test data
    movie_data = pd.read_csv('movie_metadata.csv')
    cols = list(movie_data.columns)

    movie_data = movie_data[movie_data['director_name'].notna()]
    movie_data = movie_data[movie_data['actor_1_name'].notna()]
    movie_data = movie_data[movie_data['actor_3_name'].notna()]
    movie_data = movie_data[movie_data['budget'].notna()]
    movie_data = movie_data[movie_data['color'].notna()]
    movie_data = movie_data[movie_data['language'].notna()]
    movie_data = movie_data[movie_data['country'].notna()]
    movie_data['content_rating'].fillna("Not Rated", inplace=True)
    movie_data['aspect_ratio'].fillna((movie_data['aspect_ratio'].mode()), inplace=True)
    movie_data['duration'].fillna((movie_data['duration'].mean()), inplace=True)
    
    
    movie_data["genres"] = movie_data["genres"].str.split('|')
    
    mlb = MultiLabelBinarizer(sparse_output=True)

    movie_data = movie_data.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(movie_data.pop('genres')),
                index=movie_data.index,
                columns=mlb.classes_))
    
    
    content_dummies = pd.get_dummies(movie_data.content_rating)
    movie_data = pd.concat([movie_data, content_dummies], axis=1)
    color_dummies = pd.get_dummies(movie_data.color)
    movie_data = pd.concat([movie_data, color_dummies], axis=1)
    print(movie_data.columns)
    
    features = movie_data[['duration',
       'director_facebook_likes', 'actor_3_facebook_likes', 
       'actor_1_facebook_likes','cast_total_facebook_likes',
       'budget', 'title_year', 'actor_2_facebook_likes',
       'aspect_ratio', 'Action', 'Adventure','Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama',
        'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music',
       'Musical', 'Mystery', 'News', 'Romance', 'Sci-Fi', 'Short', 'Sport',
       'Thriller', 'War', 'Western', 'Approved', 'G', 'GP', 'M', 'NC-17',
       'Not Rated', 'PG', 'PG-13', 'Passed', 'R', 'TV-14', 'TV-G', 'TV-PG',
       'Unrated', 'X', ' Black and White', 'Color']]
    
    labels = movie_data["imdb_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=.20, random_state=0)
    
    X_train.to_csv('xtrain.csv',index=False)
    y_train.to_csv('ytrain.csv',index=False)
    X_test.to_csv('xtest.csv',index=False)
    y_test.to_csv('ytest.csv',index=False)
   


    
    


if __name__ == "__main__":
    main()