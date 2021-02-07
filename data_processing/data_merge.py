from data_processing.data_cleaning import reviews_clean, meta_clean

import pandas as pd


def final_data(dest_path, review_path=None, meta_path=None,
               temp_rev='./data/processed/clean_reviews.json.gz',
               temp_meta='./data/processed/clean_meta.json.gz'):

    # reviews_clean(review_path, temp_rev)
    # meta_clean(meta_path, temp_meta)
    if review_path is None and meta_path is None:
        df_review = pd.read_json(temp_rev)
        df_meta = pd.read_json(temp_meta)
    else:
        df_review = pd.read_json(review_path)
        df_meta = pd.read_json(meta_path)

    one_df = df_review.merge(df_meta, on='asin')
    one_df = one_df[one_df['verified']]
    features = ['asin', 'reviewerID', 'verified', 'overall', 'review_count', 'summary', 'reviewText']
    one_df = one_df[features]
    one_df.drop_duplicates(subset=['asin', 'reviewerID'], inplace=True)
    one_df.to_json(dest_path, compression='gzip')
    return one_df
