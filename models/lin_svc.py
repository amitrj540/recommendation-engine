from data_processing.text_processing import stem_text, rem_stopwords, text_clean
from models.sampler_sentiment_generator import sentiment_generator, sampler
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle


def train(df_path='All_Beauty_clean.json.gz', test_reviews=None):
    """
    Trains LinearSVC model on balanced dataset of Amazon
    params:
    df_path = 'All_Beauty_clean.json.gz' #Path of the dataset
    test_reviews = None #List of reviews for model testing
    """
    print('Loading Data ...')
    main_df = pd.read_json(df_path)
    features = ['reviewText', 'overall']
    temp_df = main_df[features]

    print('Generating Sentiment ...')
    sentiment = sentiment_generator(temp_df['overall'])

    print('Sampling data ...')
    df = sampler(pd.concat([temp_df, sentiment], axis=1))

    print('Text Processing ...')
    df['reviewText'] = df['reviewText'].apply(text_clean)

    print('Removing Stopwords ...')
    df['reviewText'] = df['reviewText'].apply(rem_stopwords)

    print('Stemming text ...')
    df['reviewText'] = df['reviewText'].apply(stem_text)

    print('Vectorizing ...')
    ngram_vectorizer = CountVectorizer(binary=False, ngram_range=(1, 3))
    ngram_vectorizer.fit(df['reviewText'])

    X = ngram_vectorizer.transform(df['reviewText'])
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    print('Parameter tuning ...')
    acc_score = []
    for c in [0.005, 0.01, 0.05, 0.25]:
        svc = LinearSVC(C=c)
        svc.fit(X_train, y_train)
        acc_score.append((c, accuracy_score(y_test, svc.predict(X_test))))
        print(f"Accuracy for C={c}: {accuracy_score(y_test, svc.predict(X_test))}")
    acc_score.sort(reverse=True, key=lambda x: x[1])
    c = acc_score[0][0]
    print(f"Final C = {c}")

    print('Training Final model...')
    final_svc = LinearSVC(C=c)
    final_svc.fit(X, y)

    # """## Sanity testing"""

    # feat_coeff = {word: coeff for word, coeff in zip(ngram_vectorizer.get_feature_names(), final_svm.coef_[1])}

    # pos_neg = sorted(feat_coeff.items(), key=lambda x: x[1], reverse=True)
    print('Testing Final model...')
    if test_reviews is None:
        test_reviews = ['This product was okish', 'I have mixed feelings about this product.',
                        'it is not upto mark', 'great', 'kinda okay']

    test_reviews = [stem_text(text_clean(item)) for item in test_reviews]

    X = ngram_vectorizer.transform(test_reviews)

    for item in zip(test_reviews, final_svc.predict(X)):
        print(f"{item[0]} >> {item[1]}")

    print("Storing vectorizer and model in pickle file ...")
    ngram_vec_file = './models/pickle_files/svc/ngram_vec.pkl'
    pickle.dump(ngram_vectorizer, open(ngram_vec_file, 'wb'))

    print("Storing model in pickle file ...")
    final_svc_file = './models/pickle_files/svc/final_Lin_SVC.pkl'
    pickle.dump(final_svc, open(final_svc_file, 'wb'))
    # del ngram_vectorizer
    # del final_svc
    print('Done.')
    return (ngram_vectorizer, final_svc)
