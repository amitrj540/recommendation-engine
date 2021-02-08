import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def cbf_data(df_path='final.json.gz'):
    """
    Generate data for content based filtering
    input :
    df_path='All_Beauty_One_final.json.gz'
    output :
    DataFrame
    """
    main_df = pd.read_json(df_path)
    feat1 = ['asin', 'description', 'title', 'price']
    feat2 = ['asin', 'overall']
    df1 = main_df.loc[:, feat1].drop_duplicates().reset_index().drop('index', axis=1)
    df2 = main_df.loc[:, feat2].groupby('asin').mean()
    return df1.merge(df2, on='asin')


def indices(df):
    return pd.Series(df.index, index=df['asin']).drop_duplicates()


def cosine_sim(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_mat = tfidf.fit_transform(df)
    return linear_kernel(tfidf_mat, tfidf_mat)


def recommend(prod_asin, cosine_sim, indices, cbf_df, lim=5, min_rate=2):
    """
    tuning param for price is lim=5
    tuning param for rating is min_rate=2
    """
    df = cbf_df
    if prod_asin not in indices:
        return []
    idx = indices[prod_asin]
    price = df.iloc[idx]['price']
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # sim_scores = sim_scores[1:11]
    prod_indices = [i[0] for i in sim_scores]
    temp = df.iloc[prod_indices]

    return temp[(temp['price'] >= price-lim) & (temp['price'] <= price+lim) &
                (temp['overall'] >= min_rate)]['asin'].tolist()
