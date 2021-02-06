from sklearn.decomposition import TruncatedSVD
# Import the dataset and give the column names
def tsvd_model(df_path,sample_frac, idx=None, col = 'reviewerID', val='positive_prob'):
    df = pd.read_json(df_path)
    df = df.sample(frac=sample_frac)
    # Matrix with row per 'item' and column per 'user'
    pivot_df = df.pivot_table(index = idx,
                               columns =col,
                               values = val).fillna(0)
    SVD = TruncatedSVD(n_components=10)
    print('Training model...')
    decomposed_matrix = SVD.fit_transform(pivot_df)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    return correlation_matrix, pivot_df.index

def collaborative_modb(product, model=None, df_path=None, sample_frac=0.05,corr_thresh=0.5):
    if model is None:
        correlation_matrix, product_names = tsvd_model(df_path, sample_frac)
    else:
        correlation_matrix, product_names= model

    product_ID = product_names.index(product)
    correlation_product_ID = correlation_matrix[product_ID]
    recommend = list(product_names.index[correlation_product_ID>corr_thresh])
    recommend.remove(i)
    return recommend
