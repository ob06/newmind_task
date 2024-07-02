import pandas as pd
import logging

def load_data(topics_path, opinions_path):
    topics_df = pd.read_csv(topics_path)
    opinions_df = pd.read_csv(opinions_path)

    # Check for null values in the data
    if topics_df.isnull().values.any():
        logging.warning("Null values found in topics data.")
        topics_df = topics_df.dropna()

    if opinions_df.isnull().values.any():
        logging.warning("Null values found in opinions data.")
        opinions_df = opinions_df.dropna()

    # Setting the type of the text column to string
    topics_df['text'] = topics_df['text'].astype(str)
    opinions_df['text'] = opinions_df['text'].astype(str)

    return topics_df, opinions_df
