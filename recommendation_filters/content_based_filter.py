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
    df1 = main_df.loc[:, feat1].drop_duplicates().reset_index().drop('index', axis=1) #faetures to check duplicate records
    df2 = main_df.loc[:, feat2].groupby('asin').mean() # mean of overall corresponding to 1 asin(many ratings for 1 asin)
    return df1.merge(df2, on='asin')


def indices(df):
    """
    Generate Series with itemID as index and index of itemID in the dataframe as value.
    """
    return pd.Series(df.index, index=df['asin']).drop_duplicates() # index is asin and value is index(0,1,2,3....). done to keep record of index corresponding to asin.


def cosine_sim(df):
    """
    Generate cosine similarity with TfidfVectorizer and linear_kernel
    """
    tfidf = TfidfVectorizer(stop_words='english') 
    tfidf_mat = tfidf.fit_transform(df) #generrating vectors from text(description)
    return linear_kernel(tfidf_mat, tfidf_mat) #making a matrix which gives similarity between different procucts.one vector has similarity of all products related to that product.


def recommend(prod_asin, cosine_sim, indices, cbf_df, lim=5, min_rate=2):
    """
    Recommend products for prod_asin
    cosine_sim = <cosine similarity matrix>
    indices = <indices>
    cbf_df = <data>
    lim=5(default) 
    deviation in price for similar priced item filtering
    min_rate=2
    minimum rating for item to be in list
    """
    df = cbf_df
    if prod_asin not in indices:
        return []
    idx = indices[prod_asin] #index value corresponding to asin
    price = df.iloc[idx]['price'] #price of asin given by user
    sim_scores = list(enumerate(cosine_sim[idx])) #from similarity matrix only row corresponding to given asin is selected.

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #sorting according to heighest similarity value

    # sim_scores = sim_scores[1:11]
    prod_indices = [i[0] for i in sim_scores] #storing index value corresponding to similarity generated
    temp = df.iloc[prod_indices] #subsetting dataframe according to indices generating

    return temp[(temp['price'] >= price-lim) & (temp['price'] <= price+lim) &
                (temp['overall'] >= min_rate)]['asin'].tolist() #to give products only in price range and with good rating.
