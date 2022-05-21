import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function using for load data from csv file to pandas DataFrame.
    
    Parameters:
    messages_filepath: str
        Filepath of messages data
    categories_filepath: str
        Filepath of categories data
        
    Return: DataFrame
        Dataframe contains data of messages and categories has been merged
    """
    mess_df = pd.read_csv(messages_filepath)
    cate_df = pd.read_csv(categories_filepath)
    return pd.merge(mess_df, cate_df, on='id')

def transform_categories(categories):
    """
    Transform data in categories Series to Dataframe with boolean value.
    Only 0 and 1 values was kept, another values will be dropped.
    
    Parameters:
    categories: pandas Series
        Pandas Series contains data of categories columns
        
    Return: DataFrame
        Dataframe converted
    """
    assert isinstance(categories, pd.Series)
    row = categories[0].split(";")
    category_colnames = [x[:-2] for x in row]
    categories_df = categories.str.split(";", expand=True)
    categories_df.columns = category_colnames
    for col in category_colnames:
        categories_df[col] = categories_df[col].str[-1]
        # make sure that only 0 and 1 in the column
#         if categories_df[col].nunique() > 2:
#             idx_drop = categories_df[categories_df[col].isin(["0","1"]) == False].index
#             categories_df = categories_df.drop(index=idx_drop)
#     categories_df = categories_df.reset_index(drop=True)
    categories_df = categories_df.astype(int)
    return categories_df

def clean_data(df):
    """
    Clean data has been loaded.
    This function use transform_categories function to transform categories column and
    drop duplicated column in dataframe.
    
    Parameters:
    df: pandas DataFrame
        DataFrame will be cleaned
        
    Return: DataFrame
    """
    print("    Transform categories")
    categories_df = transform_categories(df.categories)
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories_df], axis=1)
#     df = df.dropna(subset=categories_df.columns)
    print("    Drop duplicated records")
    df = df.drop_duplicates()
    return df.reset_index(drop=True)


def save_data(df, database_filename, if_exists='replace'):
    """
    Write data stored in Dataframe to database.
    
    Parameters:
    df: pandas DataFrame
        DataFrame has been cleaned.
    database_filename: str
        Name of database.
    if_exists: str, default: replace
        How to behave if the table already exists.
        It will be replace by new table by default
        
    Return: None
    """
    engine = create_engine("sqlite:///" + database_filename)
    df.to_sql("disaster_data", engine, index=False, if_exists=if_exists)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f"Data size after cleaned: {df.shape}")
        
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