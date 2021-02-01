# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_json('All_Beauty_One_final.json.gz')

df1 = df.loc[:,['asin', 'description',	'title', 'price']].drop_duplicates().reset_index().drop('index', axis=1)

df2 = df.loc[:,['asin', 'overall']].groupby('asin').mean()

df = df1.merge(df2, on='asin')

#from sklearn.feature_extraction.text import CountVectorizer

tfidf = TfidfVectorizer(stop_words='english')

#count_vec = CountVectorizer(min_df=1, ngram_range=(1,4))

#count_vec_fit_trans = count_vec.fit_transform(df1['title'])

tfidf_mat = tfidf.fit_transform(df1['description'])

"""#Finding Cosine Similarity (title)"""

cosine_sim = linear_kernel(tfidf_mat, tfidf_mat)

"""## Generating recommendation function"""

indices = pd.Series(df1.index, index=df['asin']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, lim=5, min_rate=2):
    idx = indices[title]
    price = df.iloc[idx]['price']
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #sim_scores = sim_scores[1:11]

    prod_indices = [i[0] for i in sim_scores]
    temp = df.iloc[prod_indices]
    temp = temp[(temp['price']>= price-lim) & (temp['price']<= price+lim) & (temp['overall']>= min_rate)][['asin', 'title', 'price', 'overall']]     
    return temp

#tuning param for price is lim=5
#tuning param for rating is min_rate=2

def content_based_recommendation():
    print("Enter Product Details: (ProductID)")
    userResponse = input()
    checkData = df['asin'].tolist()
    if(userResponse in checkData):
        return get_recommendations(userResponse,min_rate=5)
    else:
        return "No Such Products! "
    return 0
