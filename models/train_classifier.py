import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# NLP imports
import nltk
import re
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

## model import

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin

# Save import
import pickle

def load_data(database_filepath):
    '''
        function to load data from a sqlite database

        Parameters:
        database_filepath (str): location of the sqlite database

        Returns:
            X (DataFrame): Object containing all the messages from the loaded data
            Y (DataFrame): Object containing all the categories in which messages can be classified
            category_names (array): names of the categories in the Y variable
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("select * from messages",engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns.values

    return X, Y, category_names


def tokenize(text):
    '''
        Function to perform tokenization, normalization and lemmatization of a text

        Parameters:
            text (str): elemento to be processed

        Returns:
            tokens (array): elements tokenized
    '''
    text = re.sub('[^A-Za-z0-9]',' ',text.lower())
    
    tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model(X_train, Y_train):
    '''
        Function to perfom a Build a model which implements a RandomForestClassifier and 
        MultiOutputClassifier on the input data

        Parameters:
            X_train (array): input elements for the model training
            Y_train (array): output elements for the model training

        Returns
            cv (Pipeline): fitted model
    '''
    clf_multi = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
    
    ('text_pipeline',Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer())            
    ])),
    
    ('clf',clf_multi)   
    ])

    cv = pipeline.fit(X_train,Y_train)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
        function to print the scores of the model stated as parameter,
        shows precission, recall, f1-score and support for each category

        Parameters:
            model (Estimator): a fitted model 
            X_test (array): input elements for the model evalueation
            Y_test (array): output elements for the model evaluation
            category_names: names of each category in the Y_test array
    '''
    y_pred = model.predict(X_test)

    y_pred2 = pd.DataFrame(y_pred,columns=Y_test.columns)

    for col in Y_test.columns.values:
        print('classification report: {} \n'.format(col))
        print(classification_report(Y_test[col],y_pred2[col]))


def save_model(model, model_filepath):
    '''
        function to save a model in pickle object

        Parameters:
            model (Estimator): a trained model
            model_filepath (str): path and name of the pickle model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()