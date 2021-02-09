import pandas as pd


# avg_rev_count = df['review_count'].mean()
#790
# avg_rating = df['overall'].mean()
#4.098
# avg_sentiment = df['reviewText_senti'].mean()
#0.44
def recommend(df_path, rev_count, rating, sentiment):
    """
    Returns the list of most popular item in the data based on its
    rev_count = <Minimum number of total reviews, required to be in list>
    rating =  <Minimum number of rating required to be in list>
    sentiment = <Minimum sentiment required to be in list>
    sentiment ranges from -1 to 1, representing most negative, neutral and positive sentiment as -1, 0, 1
    """
    df = pd.read_json(df_path)
    df = df.loc[:, ['asin', 'overall', 'review_count', 'reviewText_senti', 'positive_prob']]
    # df1 = df.loc[:, ['asin', 'title']].drop_duplicates().reset_index().drop('index', axis=1)
    pop_prod = df.groupby('asin').mean()
    # df1.merge(pop_prod, on='asin')
    pop_prod = pop_prod[(pop_prod['overall'] >= rating) &
                        (pop_prod['review_count'] >= rev_count) &
                        (pop_prod['reviewText_senti'] >= sentiment)]
    # Sorting Best products
    best_prod = pop_prod.sort_values('positive_prob', ascending=False)
    # Most Popular Products
    # most_pop = best_prod.groupby('asin').mean().sort_values('positive_prob', ascending=False)
    # most_pop = df1.merge(most_pop, on='asin', index=False)
    return best_prod.index.tolist()
