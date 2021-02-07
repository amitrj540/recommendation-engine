from recommendation_filters import content_based_filter, popularity_filter
from data_processing import data_cleaning, data_merge, feature_genration, text_processing
from models import collaborative_item_item, nb, lin_svc,\
    sampler_sentiment_generator, collaborative_model_based
import pickle


def main():
    rev_path = './data/raw/All_Beauty.json.gz'
    rev_clean_path = './data/processed/clean_reviews.json.gz'
    meta_path = './data/raw/meta_All_Beauty.json.gz'
    meta_clean_path = './data/processed/clean_meta.json.gz'
    data_cleaning.reviews_clean(rev_path, rev_clean_path)
    data_cleaning.meta_clean(meta_path, meta_clean_path)
    lin_svc.train(rev_clean_path)
    nb.train(rev_clean_path)
    feature_genration.all_feature(rev_clean_path)
    data_merge.final_data(dest_path='./data/processed/final.json.gz')
    final_df = './data/processed/final.json.gz'
    pop_res = popularity_filter.recommend(final_df, rev_count=500, rating=3, sentiment=0.6)
    print(pop_res)
    df = content_based_filter.cbf_data(final_df)
    idx = content_based_filter.indices(df)
    cosim = content_based_filter.cosine_sim(df['description'])
    cbf_ip = input('Enter product asin : ')
    cbf_res = content_based_filter.recommend(prod_asin=cbf_ip, cosine_sim=cosim, indices=idx, cbf_df=df, lim=5, min_rate=2)
    print(cbf_res)
    svd_model = collaborative_model_based.train(df_path=final_df, sample_frac=0.01, idx='asin', col='reviewerID', val='positive_prob')
    modb_ip = input('Enter product asin : ')
    modb_res = collaborative_model_based.recommend(product=modb_ip, model=svd_model, corr_thresh=0.5)
    print(modb_res)
    collaborative_item_item.train(final_df, limit=5)


if __name__ == "__main__":
    main()
