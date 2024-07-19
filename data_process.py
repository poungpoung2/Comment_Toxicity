import pandas as pd
from pathlib import Path
import re
import unidecode
import math



pd.options.mode.chained_assignment = None  

def remove_emoji(string):
    # Remove emoji components
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def preprocess(df, text_column="comment_text", score='toxicity'):
    print('Processing...')
    url_pattern = r"https?://\S+|www\.\S+"
    
    # Remove url
    df[text_column] = df[text_column].str.replace(url_pattern, " ")
    # Apply unidecode
    df[text_column] = df[text_column].map(unidecode.unidecode)
    # Remove emoji
    df[text_column] = df[text_column].map(remove_emoji)
    # Apply lower
    df[text_column] = df[text_column].str.lower()
    
    # Round Score    
    df[score] = df[score].apply(lambda x: round(x, 1))
    
    return df

def get_dataframe(path):
    # Read the csv and preprocess the data 
    df_full = pd.read_csv(path)
    df = df_full[['comment_text', 'toxicity']]
    df = preprocess(df)
    return df
