from data_processing.text_processing import text_clean
import pandas as pd
import numpy as np


def reviews_clean(src_path, dest_path):
    """
    Cleans amazon's user reviews dataset.
    params:
    src_path : path for dataset
    dest_path : path where cleaned data will be stored
    """
    df = pd.read_json(src_path)#, lines=True)
    features_not_req = ['reviewTime', 'style', 'image']
    df.drop(features_not_req, axis=1, inplace=True)
    to_impute = ['reviewerName', 'reviewText', 'summary'] #text features 
    df[to_impute] = df[to_impute].fillna('')
    df['vote'].fillna(0, inplace=True)    #most reviews we not voted by anyone.
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    df.to_json(dest_path, compression='gzip')
    return df


def meta_clean(src_path, dest_path):
    """
    Cleans amazon's user reviews meta dataset.
    params:
    src_path : path for dataset
    dest_path : path where cleaned data will be stored.
    """
    df = pd.read_json(src_path)#, lines=True)
    features_not_req = ['category', 'tech1', 'fit', 'tech2', 'feature', 'date',
                        'image', 'main_cat', 'also_buy', 'rank', 'also_view',
                        'similar_item', 'details']
    # category,tech2,fit were 100% Nan.
    # tech1,feature,similar_item were also 99% Nan.
    #date,image,rank -unrelated
    #main_cat - same value for all rows
    #also_buy,also_view also max nan values + influenced by amazons own rec. engine
    #details - info related to shipping.

    df.drop(features_not_req, axis=1, inplace=True)

    apply_func = lambda x: np.nan if (isinstance(x, list) and len(x) == 0) else (np.nan if x == '' else x)

    for col in df.columns:
        df[col] = df[col].apply(apply_func) #replacing all blank spaces with nan

    df['price'] = df.loc[:, 'price'].apply(lambda x: str(x).strip(' $') if x is not None else np.nan) #price imputation
    price_filter = lambda x: (float(x) if len(x) <= 6 else np.nan) if x is not None and not isinstance(x, float) else x
    df['price'] = df.loc[:, 'price'].apply(price_filter).astype(float)

    desc_filter = lambda x: (" ".join(x) if isinstance(x, list) else x) if x is not None else np.nan #description imputation
    df['description'] = df.loc[:, 'description'].apply(desc_filter)

    df.dropna(subset=['title'], inplace=True) #dropping records where tiltle is missing. 

    df['description'].fillna(df['title'], inplace=True) #imputing missing discription with titles.

    df['brand'].fillna("", inplace=True) #missing brands with space
    
#text preprocessing for text features.

    df['title'] = df['title'].apply(text_clean)
    df['description'] = df['description'].apply(text_clean)

    df['price'].fillna(df['price'].mean(), inplace=True) #imputing price with mean.

    df['price'] = df['price'].apply(lambda x: round(x, 2)) #rounding price

    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    df.to_json(dest_path, compression='gzip')
    return df
