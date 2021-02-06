from recommendation_filters import content_based_filter, popularity_filter
from data_processing import data_cleaning, data_merge, feature_generation, text_processing
from models import collaborative_item_item, model_train_nb, model_train_svm, sampler_sentiment_generator, collaborative_model_based
import pickle
def main():
    rev_path = '/data/raw/All_Beauty.json.gz'
    rev_clean_path = '/data/raw/clean_reviews.json.gz'
    meta_path = '/data/raw/meta_All_Beauty.json.gz'
    meta_clean_path = '/data/processed/clean_meta.json.gz'
    reviews_clean(rev_path, rev_clean_path)
    meta_clean(meta_path, meta_clean_path)
    lin_SVC(rev_clean_path)
    nb_model(rev_clean_path)
    all_feature(rev_clean_path)
    final_data(rev_path, meta_path, '/data/processed/final.json.gz')

    pop_res = popularity(df_path, rev_count, rating, sentiment)
    print(pop_res)

    df = cbf_data('/data/processed/final.json.gz')
    idx = indices(df)
    cosim = cosine_sim(df['description'])
    cbf_ip = input('Enter product asin : ')
    cbf_res=cbf_recommendation(prod_asin=cbf_ip, cosine_sim = cosim, indices= idx, cbf_df = df, lim=5, min_rate=2)
    print(cbf_res)

    svd_model = tsvd_model(df_path='/data/processed/final.json.gz',sample_frac=0.01, idx=None, col = 'reviewerID', val='positive_prob')
    modb_ip = input('Enter product asin : ')
    modb_res = collaborative_modb(product=modb_ip, model=svd_model, corr_thresh=0.5)
    print(modb_res)

    item_item_collaborative(rev_clean_path, limit=5)

if __name__ == "__main__":
    main()
