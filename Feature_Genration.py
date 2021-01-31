# -*- coding: utf-8 -*-
import pandas as pd
import pickle
import re
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def genrate_Feature():
    stemmer = PorterStemmer()
    no_space = re.compile("[.;:!\'?,\"()\[\]#]")
    space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\n)|(\t)")
    eng_stop_words = stopwords.words('english')
    
    preprocess = lambda text : " ".join(space.sub(" ", no_space.sub("", text.lower())).split())
    rem_stopwords = lambda line : " ".join([word for word in line.split() if word not in eng_stop_words])
    stem_text = lambda line : " ".join([stemmer.stem(word) for word in line.split()])
    
    text_processing = lambda text : stem_text(rem_stopwords(preprocess(text)))
    
    df = pd.read_json('All_Beauty_clean.json.gz')
    
    df['review_count'] = df['asin'].map(dict(df.groupby('asin').count()['reviewerID']))
    
    df = df[df['verified']]
    
    features = ['asin', 'reviewerID', 'overall', 'review_count', 'summary', 'reviewText']
    df = df[features]
    
    df['reviewText'] = df['reviewText'].apply(text_processing)
    df['summary'] = df['summary'].apply(text_processing)
    
    """# Loading Trained Model
    
    ## SVM
    """
    ngram_vect = pickle.load(open('ngram_vect.pkl', 'rb'))
    svm_model = pickle.load(open('final_svm.pkl', 'rb'))
    
    ngram = ngram_vect.transform(df['reviewText'])
    
    df.reset_index(inplace=True)
    df.drop('index', axis=1,inplace=True)
    
    svm_pred = svm_model.predict(ngram)
    
    df = pd.concat([df,pd.DataFrame(svm_pred, columns=['reviewText_senti'])], axis=1)
    
    """## Naive Bayes"""
    
    count_vect = pickle.load(open('count_vect_file.pkl', 'rb'))
    tfidf_vect = pickle.load(open('tfidf_vect_file.pkl', 'rb'))
    mnb_model = pickle.load(open('final_mnb_file.pkl', 'rb'))
    
    mnb_prob = mnb_model.predict_proba(tfidf_vect.transform(count_vect.transform(df['summary'])))
    
    df = pd.concat([df,pd.DataFrame(mnb_prob, columns=['negative_prob','neutral_prob','positive_prob'])], axis=1)
    
    df.to_json('All_Beauty_semifinal.json.gz', compression='gzip')
    
    df.drop(['summary','reviewText'], axis=1, inplace=True)
    
    df.to_json('All_Beauty_final.json.gz', compression='gzip')
    return "Feature Genration Succesfull"