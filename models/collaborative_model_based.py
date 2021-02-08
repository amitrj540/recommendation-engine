import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
# Import the dataset and give the column names


def train(df_path, sample_frac, idx='asin', col='reviewerID', val='positive_prob'):
    df = pd.read_json(df_path)
    df = df.sample(frac=sample_frac)
    # Matrix with row per 'item' and column per 'user'
    pivot_df = df.pivot_table(index=idx,
                              columns=col,
                              values=val).fillna(0)
    svd = TruncatedSVD(n_components=10)
    print('Training model...')
    decomposed_matrix = svd.fit_transform(pivot_df)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    print(pivot_df.index)
    return (correlation_matrix, pivot_df.index)


def recommend(product, model=None, df_path=None, sample_frac=0.05, corr_thresh=0.5):
    if model is None:
        correlation_matrix, pivot_idx = train(df_path, sample_frac)
    else:
        correlation_matrix, pivot_idx = model

    product_names = list(pivot_idx)
    if product not in product_names:
        return []
    product_id = product_names.index(product)
    correlation_product_id = correlation_matrix[product_id]
    recom = list(pivot_idx[correlation_product_id > corr_thresh])
    recom.remove(product)
    return recom
