import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
        Return a single dataframe with the information contained
        in the filepaths passed as parameter

        Parameters:
        messages_filepath (str): filepath of messages file
        categories_filepath (str): filepath of categories file

        Returns:
        DataFrame: contain the joined information from messages and categories
    '''
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    
    
    return pd.merge(df_messages,df_categories,on='id')


def clean_data(df):
    '''
        Cleans and prepares the data contained in the dataframe parameter:
            1. divide into columns information in the categories columns
            2. extract numeric part from the divided columns
            3. drop duplicated data

        Parameters:
        df (DataFrame): object to be cleaned 
        
        Returns:
        DataFrame: cleaned dataframe
    '''
    # separate categories columns to be treated a part
    categories = df['categories'].str.split(';',expand=True)

    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x : x.split('-')[0])
    
    categories.columns = category_colnames

    # Get the numeric part (1 - 0) from each of the columns, since they are stored as "request-0"
    for column in categories:
        
        categories[column] = categories[column].str.split('-').str[1]
        categories[column] = pd.to_numeric(categories[column])

    
    # apend new columns and discard previous one
    df.drop(columns='categories',inplace=True)
    df = pd.concat([df,categories],axis=1)

    # discar duplicates
    df = df[~df['id'].duplicated(keep='first')]
        
    return df


def save_data(df, database_filename):
    '''
        save the dataframe into a indicated sqlite database
    '''

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False)
    pass  


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