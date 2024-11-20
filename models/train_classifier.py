# import libraries
import sys
import pandas as pd
import pickle
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    Load data from SQLite. Parameter: database_filepath: Path to file sqlite.db
    Return:
    X: Contain message
    Y: Contain categories
    category_names: List of category names
    
    '''
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    '''
    This function tokenizes a text string into a list of words.
    Parameter: text: A string containing the text to be tokenized.
    Returns: A list of tokens (single words) separated from input text.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # Lemmatize and normalize case
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Build pipeline to run 3 libraries: CountVectorizer, TfidfTransformer, RandomForestClassifier
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    print("Pipeline parameters:")
    print(pipeline.get_params())
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model
    model: Pipeline containing classifier.
    Metrics: F1 score, Recall, Precision
    '''
    # Make predictions on the test set
    Y_pred = model.predict(X_test)
    
    # Iterate through each category in the target variable
    for i, category in enumerate(category_names):
        print('Category: {}'.format(category))
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        # Calculate metrics for the current category
        metrics = {
            'F1 score': f1_score(Y_test.iloc[:, i], Y_pred[:, i], average='weighted'),
            'Precision': precision_score(Y_test.iloc[:, i], Y_pred[:, i], average='weighted'),
            'Recall': recall_score(Y_test.iloc[:, i], Y_pred[:, i], average='weighted')
        }

        # Print the metrics
        for metric_name, metric_value in metrics.items():
            print('{}: {}'.format(metric_name, metric_value))


def save_model(model, model_filepath):
    '''
    Save model into pickle file
    '''
    try:
        with open(model_filepath, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {model_filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        #model.fit(X_train, Y_train)
        parameters = {
            'clf__estimator__n_estimators': [2, 4],  # Number of trees in the forest
            'clf__estimator__max_depth': [None, 2],  # Maximum depth of the tree
            'clf__estimator__min_samples_split': [2, 4],  # Minimum number of samples required to split an internal node
        }
        # Set up GridSearchCV
        grid_search = GridSearchCV(model, parameters, cv=5, scoring='f1_weighted', verbose=2)
        
        # Fit GridSearchCV
        grid_search.fit(X_train, Y_train)

        # Get the best parameters and best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {best_score}")
        
        print('Evaluating model...')
        evaluate_model(grid_search.best_estimator_, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(grid_search.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()