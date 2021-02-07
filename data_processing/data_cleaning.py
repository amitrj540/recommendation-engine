from data_processing.text_processing import stem_text, rem_stopwords, text_clean
import pandas as pd
import numpy as np


def reviews_clean(src_path, dest_path):
    """
    Cleans amazon's user reviews dataset.
    params:
    src_path : path for dataset
    dest_path : path where cleaned data will be stored
    """
    df = pd.read_json(src_path, lines=True)
    features_not_req = ['reviewTime', 'style', 'image']
    df.drop(features_not_req, axis=1, inplace=True)
    to_impute = ['reviewerName', 'reviewText', 'summary']
    df[to_impute] = df[to_impute].fillna('')
    df['vote'].fillna(0, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    df.to_json(dest_path, compression='gzip')
    return True


def meta_clean(src_path, dest_path):
    """
    Cleans amazon's user reviews meta dataset.
    params:
    src_path : path for dataset
    dest_path : path where cleaned data will be stored.
    """
    df = pd.read_json(src_path, lines=True)
    features_not_req = ['category', 'tech1', 'fit', 'tech2', 'feature', 'date',
                        'image', 'main_cat', 'also_buy', 'rank', 'also_view',
                        'similar_item', 'details']

    df.drop(features_not_req, axis=1, inplace=True)

    apply_func = lambda x: np.nan if (isinstance(x, list) and len(x) == 0) else (np.nan if x == '' else x)

    for col in df.columns:
        df[col] = df[col].apply(apply_func)

    df['price'] = df.loc[:, 'price'].apply(lambda x: str(x).strip(' $') if x is not None else np.nan)
    price_filter = lambda x: (float(x) if len(x) <= 6 else np.nan) if x is not None and not isinstance(x, float) else x
    df['price'] = df.loc[:, 'price'].apply(price_filter).astype(float)

    desc_filter = lambda x: (" ".join(x) if isinstance(x, list) else x) if x is not None else np.nan
    df['description'] = df.loc[:, 'description'].apply(desc_filter)

    df.dropna(subset=['title'], inplace=True)

    df['description'].fillna(df['title'], inplace=True)
    df['brand'].fillna("", inplace=True)

    df['title'] = df['title'].apply(text_clean)
    df['description'] = df['description'].apply(text_clean)

    df['price'].fillna(df['price'].mean(), inplace=True)

    df['price'] = df['price'].apply(lambda x: round(x, 2))

    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)
    df.to_json(dest_path, compression='gzip')
    return df
