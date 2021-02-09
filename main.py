from recommendation_filters import content_based_filter, popularity_filter
from data_processing import data_cleaning, data_merge, feature_genration, text_processing
from models import collaborative_item_item, nb, lin_svc, collaborative_model_based, sampler_sentiment_generator
import pandas as pd

def main():
    rev_path = './data/raw/All_Beauty_25.json.gz'
    rev_clean_path = './data/processed/clean_reviews.json.gz'
    meta_path = './data/raw/meta_All_Beauty_25.json.gz'
    meta_clean_path = './data/processed/clean_meta.json.gz'
    data_cleaning.reviews_clean(rev_path, rev_clean_path)
    data_cleaning.meta_clean(meta_path, meta_clean_path)
    lin_svc.train(rev_clean_path)
    nb.train(rev_clean_path)
    feature_genration.all_feature(rev_clean_path)
    data_merge.final_data(dest_path='./data/processed/final.json.gz')
    name_df = pd.read_json('./data/asin_title.json.gz')
    final_df = './data/processed/final.json.gz'

    df = content_based_filter.cbf_data(final_df)
    idx = content_based_filter.indices(df)
    cosim = content_based_filter.cosine_sim(df['description'])

    svd_model = collaborative_model_based.train(df_path=final_df, sample_frac=0.5, idx='asin',
                                                col='reviewerID', val='positive_prob')
    exit_flag = False
    while not exit_flag:
        print('1. Select popularity filter')
        print('2. Select content based filter')
        print('3. Select collaborative model based filter')
        print('4. Select Hybrid filter')
        print('0. Exit')
        ch = input('Enter choice :: ')
        if ch == '1':
            pop_res = popularity_filter.recommend(final_df, rev_count=25, rating=3, sentiment=0.6)
            print(pop_res[:15])
            print(pd.DataFrame(pop_res[:15], columns=['asin']).merge(name_df, on='asin'))
        elif ch == '2':
            cbf_ip = input('Enter product asin : ')
            cbf_res = content_based_filter.recommend(prod_asin=cbf_ip, cosine_sim=cosim, indices=idx,
                                                     cbf_df=df, lim=5, min_rate=2)
            print(cbf_res[:15])
            print(pd.DataFrame(cbf_res[:15], columns=['asin']).merge(name_df, on='asin'))
        elif ch == '3':
            modb_ip = input('Enter product asin : ')
            modb_res = collaborative_model_based.recommend(product=modb_ip, model=svd_model, corr_thresh=0.5)
            print(modb_res[:15])
            print(pd.DataFrame(modb_res[:15], columns=['asin']).merge(name_df, on='asin'))
        elif ch == '4':
            hyb_ip = input('Enter product asin : ')
            rec = collaborative_model_based.recommend(product=hyb_ip, model=svd_model, corr_thresh=0.5)
            if len(rec) < 5:
                cbf_rec = content_based_filter.recommend(prod_asin=hyb_ip, cosine_sim=cosim, indices=idx,
                                                         cbf_df=df, lim=5, min_rate=2)
                rec.extend(cbf_rec)
            if len(rec) < 5:
                pop_rec = popularity_filter.recommend(final_df, rev_count=25, rating=3, sentiment=0.6)
                rec.extend(pop_rec)
            print(rec[:15])
            print(pd.DataFrame(rec[:15], columns=['asin']).merge(name_df, on='asin'))
        else:
            exit_flag = True

            
if __name__ == "__main__":
    main()
