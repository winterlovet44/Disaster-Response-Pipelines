import sys
import pickle

import re
import pandas as pd
# import numpy as np
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


# Global pattern of http for regex
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


# Check if nltk package was dowloaded
def package_checker(package_name):
    """
    Helper function for check NLTK has been dowloaded

    Parameters:
    package_name: str
        Name of package will be checked

    return: bool
        Return True if package has been dowloaded,
        otherwise return False
    """
    try:
        s = nltk.data.find(package_name)
        return True
    except:
        return False


# Global check NLTK package
package_list = ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
for package in package_list:
    if not package_checker(package):
        nltk.download(package)


def load_data(database_filepath):
    """
    Load data contains in database.

    Parameter:
    database_filepath: str
        Filepath of database

    Return:
    X: pd.DataFrame
        DataFrame contain training data input
    Y: pd.DataFrame
        DataFrame contain label to predict
    category_cols: list
        List of columns name of training data
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table = "disaster_data"
    # We don't need values 2 from column related.
    df = pd.read_sql(f"select * from {table} where related != 2", engine)
    category_cols = df.columns[4:].tolist()
    X = df.message.values
    Y = df[category_cols].values
    return X, Y, category_cols


def tokenize(text):
    """
    Helper function for preprocess text data.
    This function perform NLP preprocessing to clean text data.

    Parameter:
    text: str
        Input text will be cleaned

    Return:
    text: str
        String has been cleaned
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(use_gridsearch=False):
    """
    Build model pipeline with sklearn.pipeline.Pipeline

    parameter:
    use_gridsearch: bool, default: False
        if True, pipeline will perform GridSearchCV to find best parameters

    Return:
    pipeline: sklearn.pipeline.Pipeline
        Pipeline of model
    """
    pipeline = Pipeline([
        ('features', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        "clf__estimator__n_estimators": [3, 4, 5],
        "clf__estimator__max_features": ["auto", "sqrt"],
        "clf__estimator__max_depth": [5, 6],
        "clf__estimator__criterion": ["gini", "entropy"],
    }
    if use_gridsearch:
        cv = GridSearchCV(pipeline, parameters)
        return cv
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Compute and return model performance report.
    Function will print report of model by using
    sklearn.metrics.classification_report

    Parameters:
    model: sklearn.BaseEstimator
        Model of Scikit-learn. it must has predict function
    X_test: array-like
        Input test data
    Y_test: array-like
        True label of input test
    category_names: list
        List of columns name

    Return:
    None
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save and write model to a .pkl file

    Parameters:
    model: sklearn.BaseEstimator
        Model will be saved
    model_filepath: str
        string of filepath to save the model

    Return:
        None
    """
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)
        f.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10)
        print(f"Training shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")

        print('Building model...')
        model = build_model()

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
