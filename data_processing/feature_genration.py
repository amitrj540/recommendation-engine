from data_processing.text_processing import stem_text, rem_stopwords, text_clean
import pandas as pd
import pickle


def review_count(df):
    return df['asin'].map(dict(df.groupby('asin').count()['reviewerID']))


def svc_features(df_ser):
    print('Loading models...')
    ngram_vect = pickle.load(open('./models/pickle_files/svc/ngram_vec.pkl', 'rb'))
    svc_model = pickle.load(open('./models/pickle_files/svc/final_Lin_SVC.pkl', 'rb'))

    print('Cleaning text...')
    df_ser = df_ser.apply(text_clean)

    print('Removing Stopwords...')
    df_ser = df_ser.apply(rem_stopwords)

    print('Stemming text ...')
    df_ser = df_ser.apply(stem_text)

    ngram = ngram_vect.transform(df_ser)
    svc_pred = svc_model.predict(ngram)

    del ngram_vect
    del svc_model
    return pd.DataFrame(data=svc_pred, columns=['reviewText_senti'])


def nb_features(df_ser):
    print('Loading models...')
    count_vect = pickle.load(open('./models/pickle_files/nb/count_vect_file.pkl', 'rb'))
    tfidf_vect = pickle.load(open('./models/pickle_files/nb/tfidf_vect_file.pkl', 'rb'))
    nb_model = pickle.load(open('./models/pickle_files/nb/final_nb_file.pkl', 'rb'))

    print('Cleaning text...')
    df_ser = df_ser.apply(text_clean)
    nb_model_prediction = nb_model.predict_proba(tfidf_vect.transform(count_vect.transform(df_ser)))
    del count_vect
    del tfidf_vect
    del nb_model
    return pd.DataFrame(nb_model_prediction, columns=['negative_prob', 'neutral_prob', 'positive_prob'])


def all_feature(df_path='All_Beauty_clean.json.gz', dest_path='./data/processed/clean_reviews.json.gz'):
    df = pd.read_json(df_path)
    req_features = ['asin', 'reviewerID', 'reviewText', 'summary']
    feat_df = df[req_features]
    #feat_df['review_count'] = 0
    feat_df.loc[:, 'review_count'] = review_count(df[['asin', 'reviewerID']])
    feat_df = pd.concat([feat_df, svc_features(df['reviewText'])], axis=1)
    feat_df = pd.concat([feat_df, nb_features(df['summary'])], axis=1)
    feat_df.drop(['asin', 'reviewerID', 'reviewText', 'summary'], inplace=True, axis=1)
    final = pd.concat([df, feat_df], axis=1)
    final.to_json(dest_path, compression='gzip')
    del df
    return final
