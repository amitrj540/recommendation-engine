import pickle

import pandas as pd
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split


# Getting the new dataframe which contains users who has given 50 or more ratings
def train(df_path, limit):
    print('Loading data...')
    df = pd.read_json(df_path)
    new_df = df[df['review_count'] >= limit]
    # Reading the dataset
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(new_df[['reviewerID', 'asin', 'overall']], reader)
    # Splitting the dataset
    train_set, test_set = train_test_split(data, test_size=0.3, random_state=101)
    # Use user_based true/false to switch between user-based or item-based collaborative filtering
    print('Parameter Tuning...')
    acc_score = []
    for i in range(1, 10):
        algo = KNNWithMeans(k=i, sim_options={'name': 'pearson_baseline', 'user_based': False})
        algo.fit(train_set)
        acc_score.append((i, accuracy.rmse(algo.test(test_set))))
    # run the trained model against the testset
    acc_score.sort(reverse=True, key=lambda x: x[1])
    c = acc_score[0][0]
    print(f"Final C = {c}\nTest RMSE : {acc_score[0][1]}")
    final_knn = KNNWithMeans(k=c, sim_options={'name': 'pearson_baseline', 'user_based': False})
    trainset = data.build_full_trainset()
    final_knn.fit(trainset)

    print("Storing model in pickle file ...")
    final_knn_file = './models/pickle_files/knn/final_knn.pkl'
    pickle.dump(final_knn, open(final_knn_file, 'wb'))
    print('Done.')
    return final_knn
