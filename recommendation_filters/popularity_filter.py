import pandas as pd

#avg_rev_count = df['review_count'].mean()
#avg_rating = df['overall'].mean()
#avg_sentiment = df['reviewText_senti'].mean()

def popularity(df_path, avg_rev_count, avg_rating, avg_sentiment):
    df = pd.read_json(df_path)
    df = df.loc[:, ['asin', 'overall', 'review_count', 'title']]
    df1 = df.loc[:, ['asin', 'title']].drop_duplicates().reset_index().drop('index', axis=1)
    pop_prod = df.groupby('asin').mean()
    df1.merge(pop_prod, on='asin')
    pop_prod = pop_prod[(pop_prod['overall'] >= avg_rating) &
                        (pop_prod['review_count'] >= avg_rev_count) &
                        (pop_prod['reviewText_senti'] >= avg_sentiment)]
    ## Sorting Best products
    best_prod = pop_prod.sort_values('positive_prob', ascending=False)
    # Most Popular Products
    most_pop = best_prod.groupby('asin').mean().sort_values('positive_prob', ascending=False)
    most_pop = df1.merge(most_pop, on='asin', index=False)
    return most_pop[['asin', 'title']]

