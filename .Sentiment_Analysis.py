# -*- coding: utf-8 -*-

import pandas as pd
import re
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

def SA():
    """
    overall = numerical data<br>
    verified = use to filter genuine review<br>
    reviewerID(feature-how many person reviewed that product), reviewText,	summary	= text based<br>
    asin = primary key
    """
    
    df = pd.read_json('All_Beauty_clean.json.gz')
    
    """
    **Accepted**
    * `overall` - contains raiting
    * `reviewerID` - for feature creation
    * `asin` - product id
    * `reviewText` - actual review
    * `summary` - truncated from of 'reviewText'
    
    **Rejected**
    * `reviewerName` - reviewer identity is not required as it does not show how good or bad the product is.
    * `unixReviewTime` - not required for recommendation engine
    * `vote` - shows usefulness of review rather than product
    """
    
    features = ['overall', 'reviewerID', 'asin', 'reviewText','summary']
    df = df[features]
    
    """
    # Data Preprocessing
    # Droping duplicate rows
    """
    
    df.drop_duplicates(inplace=True)
    senti = lambda x : -1 if x in [1,2] else (0 if x == 3 else 1)
    df['sentiment'] = df['overall'].apply(senti)
    low_row = df.loc[(df["sentiment"] == -1), 'overall'].value_counts().sum()
    neu_row = df.loc[(df["sentiment"] == 0), 'overall'].value_counts().sum()
    high_row = df.loc[(df["sentiment"] == 1), 'overall'].value_counts().sum()
    sample_nos = min(low_row,neu_row,high_row)
    neg = df.loc[df["sentiment"] == -1].sample(n=sample_nos, random_state=101)
    neu = df.loc[df["sentiment"] == 0].sample(n=sample_nos, random_state=101)
    pos = df.loc[df["sentiment"] == 1].sample(n=sample_nos, random_state=101)
    
    df=pd.concat([neg,neu,pos],axis=0)
    
    """## Removing Punctuations and HTML tags"""
    
    no_space = re.compile("[.;:!\'?,\"()\[\]#]")
    space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\n)|(\t)")
    preprocess = lambda text : " ".join(space.sub(" ", no_space.sub("", text.lower())).split())
    df['reviewText']=df['reviewText'].apply(preprocess)
    df['summary']=df['summary'].apply(preprocess)
    
    """# Text Processing"""
    
    eng_stop_words = stopwords.words('english')
    rem_stopwords = lambda line : " ".join([word for word in line.split() if word not in eng_stop_words])
    
    df['reviewText']=df['reviewText'].apply(rem_stopwords)
    df['summary']=df['summary'].apply(rem_stopwords)
    
    """## Normalizing text"""
    
    stemmer = PorterStemmer()
    stem_text = lambda line : " ".join([stemmer.stem(word) for word in line.split()])
    
    df['reviewText']=df['reviewText'].apply(stem_text)
    df['summary']=df['summary'].apply(stem_text)
    
    """## Exporting processed data"""
    
    df.to_json('All_Beauty_senti.json.gz', compression='gzip')
    df = pd.read_json('All_Beauty_senti.json.gz')
    
    """# Feature Extaction
    For LinearSVM model
    """
    
    df = df[['reviewText','sentiment']]
    
    """## Vectorization """
    
    ngram_vectorizer = CountVectorizer(binary=False, ngram_range=(1, 3))
    ngram_vectorizer.fit(df['reviewText'])
    X = ngram_vectorizer.transform(df['reviewText'])
    y = df['sentiment']
    
    """# Model selection
    
    ## Train Test Split
    """
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    """## Parameter tuning of model
    
    ```
    for c in [0.005, 0.01, 0.05, 0.25]:
        svm = LinearSVC(C=c)
        svm.fit(X_train, y_train)
        print (f"Accuracy for C={c}: {accuracy_score(y_test, svm.predict(X_test))}" )
    
    #Accuracy for C=0.005: 0.6976497695852535
    #Accuracy for C=0.01: 0.7009216589861751
    #Accuracy for C=0.05: 0.6999539170506912
    #Accuracy for C=0.25: 0.6898617511520737
    ```
    
    ## Final model
    """
    
    final_svm_ngram = LinearSVC(C=0.01)
    final_svm_ngram.fit(X_train, y_train)
    print (f"Final Accuracy: {accuracy_score(y_test, final_svm_ngram.predict(X_test))}")
    
    final_svm = LinearSVC(C=0.01)
    final_svm.fit(X,y)
    
    """## Sanity testing"""
    
    feat_coeff = {word : coeff for word, coeff in zip(ngram_vectorizer.get_feature_names(),final_svm.coef_[0])}
    
    pos_neg = sorted(feat_coeff.items(), key= lambda x : x[1], reverse=True)
    
    pos_neg[-5:]
    
    """# Model Testing"""
    
    reviews = ['This product was okish', 'I have mixed feelings about this product.', 'it is not upto mark', 'great', 'kinda okay']
    
    reviews= [preprocess(item) for item in reviews]
    reviews= [rem_stopwords(item) for item in reviews]
    reviews= [stem_text(item) for item in reviews]
    
    X = ngram_vectorizer.transform(reviews)
    
    final_svm.predict(X)
    
    """## Storing model in pickle file"""
    
    def text_processing(text):
      from nltk.stem import PorterStemmer
      import nltk
      nltk.download('stopwords')
      from nltk.corpus import stopwords
      stemmer = PorterStemmer()
      no_space = re.compile("[.;:!\'?,\"()\[\]#]")
      space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\n)|(\t)")
      eng_stop_words = stopwords.words('english')
      preprocess = lambda text : " ".join(space.sub(" ", no_space.sub("", text.lower())).split())
      rem_stopwords = lambda line : " ".join([word for word in line.split() if word not in eng_stop_words])
      stem_text = lambda line : " ".join([stemmer.stem(word) for word in line.split()])
      return stem_text(rem_stopwords(preprocess(text)))
    
    ngram_vect_file = 'ngram_vect.pkl'
    pickle.dump(ngram_vectorizer, open(ngram_vect_file, 'wb'))
    
    final_svm_file = 'final_svm.pkl'
    pickle.dump(final_svm, open(final_svm_file, 'wb'))
    
    #p2
    
    df = pd.read_json("All_Beauty_senti.json.gz")
    
    """# Feature Extraction"""
    
    df = df[['summary','sentiment']]
    X = df['summary']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
    
    count_vectorizer = CountVectorizer(min_df = 1, ngram_range = (1, 4))
    tfidf_transformer = TfidfTransformer()
    
    def vectorizer(dataFit, dataTrans):
      count_vectorizer = CountVectorizer(min_df = 1, ngram_range = (1, 4))
      tfidf_transformer = TfidfTransformer()
      tfidf_transformer.fit(count_vectorizer.fit_transform(dataFit))
      return tfidf_transformer.transform(count_vectorizer.fit_transform(dataTrans))
    
    count_vectorizer = CountVectorizer(min_df = 1, ngram_range = (1, 4))
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit_transform(count_vectorizer.fit_transform(X_train))
    
    X_train_vec = tfidf_transformer.transform(count_vectorizer.fit_transform(X_train))
    
    X_test_vec = tfidf_transformer.transform(count_vectorizer.transform(X_test))
    
    
    
    """# Multinomial Naïve Bayes learning method"""
    
    model_MultinomialNB = MultinomialNB().fit(X_train_vec, y_train)
    
    model_MultinomialNB.classes_
    
    prediction_MultinomialNB = model_MultinomialNB.predict(X_test_vec)
    
    accuracy_score(y_test,prediction_MultinomialNB)
    
    """# Bernoulli Naïve Bayes learning method"""
    
    model_BernoulliNB = BernoulliNB().fit(X_train_vec, y_train)
    prediction_Bernoulli = model_BernoulliNB.predict(X_test_vec)
    
    accuracy_score(y_test,prediction_Bernoulli)
    
    """# Final problistic model"""
    
    count_vectorizer = CountVectorizer(min_df = 1, ngram_range = (1, 4))
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit_transform(count_vectorizer.fit_transform(X))
    
    X_vec = tfidf_transformer.transform(count_vectorizer.fit_transform(X))
    
    final_mnb = MultinomialNB().fit(X_vec,y)
    
    
    """# Testing Model"""
    
    revs = ['good', 'okay', 'good enough', 'kind good']
    
    revs = pd.Series(revs)
    
    final_mnb.predict(tfidf_transformer.transform((count_vectorizer.transform(revs))))
    
    """# Storing trained model"""
    
    count_vect_file = 'count_vect_file.pkl'
    pickle.dump(count_vectorizer, open(count_vect_file, 'wb'))
    
    tfidf_vect_file = 'tfidf_vect_file.pkl'
    pickle.dump(tfidf_transformer, open(tfidf_vect_file, 'wb'))
    
    final_mnb_file = 'final_mnb_file.pkl'
    pickle.dump(final_mnb, open(final_mnb_file, 'wb'))
    return "Sentiment analysis Succesful"