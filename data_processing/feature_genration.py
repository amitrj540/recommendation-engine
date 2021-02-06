# -*- coding: utf-8 -*-
from text_processing import stem_text, rem_stopwords, text_clean
import pandas as pd
import pickle

def review_count(df):
    return df['asin'].map(dict(df.groupby('asin').count()['reviewerID']))



def svc_features(df_ser):
    print('Loading models...')
    ngram_vect = pickle.load(open('/models/ngram_vect.pkl', 'rb'))
    svc_model = pickle.load(open('/models/final_svc.pkl', 'rb'))

    print('Cleaning text...')
    df_ser = df_ser.apply(text_clean)

    print('Removing Stopwords...')
    df_ser = df_ser.apply(rem_stopwords)

    print('Stemming text ...')
    df_ser = df_ser.apply(stem_text)

    ngram = ngram_vect.transform(df_ser)
    svc_pred = svc_model.predict(ngram)

    return svc_pred.to_frame(name='reviewText_senti')


def nb_features(df_ser):
    print('Loading models...')
    count_vect = pickle.load(open('count_vect_file.pkl', 'rb'))
    tfidf_vect = pickle.load(open('tfidf_vect_file.pkl', 'rb'))
    nb_model = pickle.load(open('final_mnb_file.pkl', 'rb'))

    print('Cleaning text...')
    df_ser = df_ser.apply(text_clean)
    nb_model_prediction = nb_model.predict_proba(tfidf_vect.transform(count_vect.transform(df_ser)))

    return pd.DataFrame(nb_model_prediction, columns=['negative_prob', 'neutral_prob', 'positive_prob'])


def all_feature(df_path = 'All_Beauty_clean.json.gz', dest_path='/data/raw/clean_reviews.json.gz'):
    df = pd.read_json(df_path)
    req_features = ['asin', 'reviewerID', 'reviewText', 'summary']
    feat_df = df[req_features]
    feat_df['review_count'] = review_count(df[['asin', 'reviewerID']])
    feat_df = pd.concat([feat_df, svc_features(df['reviewText'])], axis=1)
    feat_df = pd.concat([feat_df, nb_features(df['summary'])], axis=1)
    feat_df.drop(req_features, inplace=True)
    feat_df.to_json(dest_path, compression='gzip')
    return feat_df