# Uda-Prj-2-disaster

## Overview

This is Project 2 - Disaster Response in Udacity Data Scientist Nano-degree, in which Twitter responses from CSV files are extracted, transformed and loaded into SQL Lite DB, before NLP/Machine Learning pipelines are implemented to tokenize, extract features, model and evaluate to give insight about nature of responses in disaster and support decision making actions.

## File Repository

    +---app 
    |  \---templates 
    |          <> go.html 
    |          <> master.html
    +---data
    |          disaster_categories.csv
    |          DisasterResponse.db
    |          disaster_messages.csv
    |          process_data.py
    \---models
    |          train_classifier.py
    |---README.md

## How to run
Repository comprises of different folders for different purposes: <br> <br>
**1. data:** this folder contains raw data files (categories.csv, messages.csv) and process_data.py that run ETL code to transform the data of Twitter messages to load into DisasterResponse.db file (SQL Lite).
To run the the ETL process, execute the command <br>
  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

Expected Output: <br>
    
    Loading data...
        MESSAGES: disaster_messages.csv
        CATEGORIES: disaster_categories.csv
    Cleaning data...
    Saving data...
        DATABASE: DisasterResponse.db
    Cleaned data saved to database!

**2. models:** this folder contains python file (train_classifier.py) to connect into DisasterResponse.db file, get transformed data to train and evaluate model.
To run the classifier training process, execute the command <br>
  `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

Expected Output: <br>

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    Loading data...
        DATABASE: data/DisasterResponse.db
    Building model...
    Pipeline parameters:
    {'memory': None, 'steps': [('vect', CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=<function tokenize at 0x7e0993cca950>, vocabulary=None)), ('tfidf', TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True)), ('clf', RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False))], 'vect': CountVectorizer(analyzer='word', binary=False, decode_error='strict',
            dtype=<class 'numpy.int64'>, encoding='utf-8', input='content',
            lowercase=True, max_df=1.0, max_features=None, min_df=1,
            ngram_range=(1, 1), preprocessor=None, stop_words=None,
            strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
            tokenizer=<function tokenize at 0x7e0993cca950>, vocabulary=None), 'tfidf': TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True), 'clf': RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False), 'vect__analyzer': 'word', 'vect__binary': False, 'vect__decode_error': 'strict', 'vect__dtype': <class 'numpy.int64'>, 'vect__encoding': 'utf-8', 'vect__input': 'content', 'vect__lowercase': True, 'vect__max_df': 1.0, 'vect__max_features': None, 'vect__min_df': 1, 'vect__ngram_range': (1, 1), 'vect__preprocessor': None, 'vect__stop_words': None, 'vect__strip_accents': None, 'vect__token_pattern': '(?u)\\b\\w\\w+\\b', 'vect__tokenizer': <function tokenize at 0x7e0993cca950>, 'vect__vocabulary': None, 'tfidf__norm': 'l2', 'tfidf__smooth_idf': True, 'tfidf__sublinear_tf': False, 'tfidf__use_idf': True, 'clf__bootstrap': True, 'clf__class_weight': None, 'clf__criterion': 'gini', 'clf__max_depth': None, 'clf__max_features': 'auto', 'clf__max_leaf_nodes': None, 'clf__min_impurity_decrease': 0.0, 'clf__min_impurity_split': None, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 2, 'clf__min_weight_fraction_leaf': 0.0, 'clf__n_estimators': 10, 'clf__n_jobs': 1, 'clf__oob_score': False, 'clf__random_state': None, 'clf__verbose': 0, 'clf__warm_start': False}
    Training model...
    Evaluating model...
    .........................
    
    Category: fire
                 precision    recall  f1-score   support
    
              0       0.99      1.00      0.99      5187
              1       0.00      0.00      0.00        57
    
    avg / total       0.98      0.99      0.98      5244
    
    F1 score: 0.9837253504395342
    Precision: 0.9783790170132325
    Recall: 0.9891304347826086
    .........................

    Saving model...
        MODEL: models/classifier.pkl
    Model saved to models/classifier.pkl
    Trained model saved!

**3. app:** this folder contains html files and run.py file to fetch data on web.
To run the classifier training process, execute the command <br>
  `python app/run.py`

Finally, open local web deployment `http://127.0.0.1:3001/` in your web browser.
