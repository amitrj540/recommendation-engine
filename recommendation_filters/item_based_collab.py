import pickle


def ib_collab_recommend(df):
    final_knn = pickle.load(open('./models/pickle_files/knn/final_knn.pkl', 'rb'))
    return final_knn.test(df)
