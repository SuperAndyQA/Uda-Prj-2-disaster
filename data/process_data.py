import sys
import pandas as pd 
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Read data from csv files and merge into data frame
    Using code in ETL Pipeline Preparation notebook
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(left=messages, right=categories, on='id', how='inner')
    return df


def clean_data(df):
    '''
    Cleans the input DataFrame by splitting the 'categories' column into individual category columns, converting the values to binary (0 or 1), and dropping duplicates.

    Parameters:
    df (DataFrame): The input DataFrame containing a 'categories' column.

    Returns:
    DataFrame: The cleaned DataFrame with individual category columns.
    '''
    # Create a dataframe of the 36 individual category columns
    categories_split = df['categories'].str.split(';', expand=True)
    
    # Extract new column names for categories
    category_colnames = categories_split.iloc[0].tolist()
    for col in range(len(category_colnames)):
        category_colnames[col] = category_colnames[col].split('-')[0]
        
    # Rename the columns of the categories_split DataFrame
    categories_split.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for col in category_colnames:
        # Set each value to be the last character of the string, then convert to numeric
        categories_split[col] = categories_split[col].astype(str).str[-1].astype(int)
        
    
    # Drop the original categories column from df
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new categories dataframe
    df = pd.concat([df, categories_split], axis=1)
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Remove related == 2 as no meaning
    df = df[df['related'] != 2]
    
    return df

    


def save_data(df, database_filename):
    '''
    Save df to sqlite db as database_filename.db, table = 'DisasterMessages'
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    with engine.connect() as connection:
        # Drop the table if it exists
        connection.execute("DROP TABLE IF EXISTS DisasterMessages")
    # Save the DataFrame to the database
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()