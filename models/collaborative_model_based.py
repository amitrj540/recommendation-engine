from sklearn.decomposition import TruncatedSVD
# Import the dataset and give the column names
def tsvd_model(df_path,sample_frac):
    df = pd.read_json(df_path)
    df = df.sample(frac=sample_frac)
    # Matrix with row per 'item' and column per 'user'
    pivot_df = df.pivot_table(index = 'asin',
                               columns ='reviewerID',
                               values = 'positive_prob').fillna(0)
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(pivot_df)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    return correlation_matrix, pivot_df.index

def collaborative_mb(df_path,sample_frac,product,corr_thresh):

    correlation_matrix, product_names = tsvd_model(df_path,sample_frac)
    product_ID = product_names.index(product)
    correlation_product_ID = correlation_matrix[product_ID]
    recommend = list(pivot_df.index[correlation_product_ID>corr_thresh])
    recommend.remove(i)
    return recommend
