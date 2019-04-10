import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load, merge messages and create dataframe
    '''    
    # load dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge dataframes
    df = messages.merge(categories, on='id', how='left')
    return df


def clean_data(df):
    '''
    Split categories and converts them into binary (0 or 1)
    '''  
    # Split categories into separate columns
    categories = df.categories.str.split(';', expand=True)

    # Rename columns
    row = categories.loc[0,:].values
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames

    # Convert category values to binary ( 0 or 1)
    for column in categories:
      # Set value to the last character of the string
      categories[column] = categories[column].astype(str).str[-1]
      
      # Convert column from string to numeric
      categories[column] = categories[column].astype(int)

    # Replace categories column in df with new columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    Load dataframe to a SQLite database
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('categories_messages', engine, index=False)
 


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