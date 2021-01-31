# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#required packages
from nltk.stem import PorterStemmer
def Visualize_Data():
 
    df_meta = pd.read_json('meta_All_Beauty.json', lines=True)
    print("Shape Of Data : \n")
    print(df_meta.shape)

    print("Summary of Null Values :\n")
    print(df_meta.isnull().sum())
    
    """#### Replacing spaces with Nan"""
    df_meta = df_meta.replace('', np.nan, regex=True)
    
    """#### Checking for length of [] to replace it by Nan"""
    #i=df_meta['category'][0]
    #len(i)
    #len(i) = 0
    #apply_func=lambda x:(np.nan if len(str(x).replace('.',''))==0 else x)
    apply_func = lambda x: (np.nan if len(x)==0 else " ".join(x)) if isinstance(x, list) else (np.nan if x=='' else x)
    
    for col in df_meta.columns:
        df_meta[col] = df_meta[col].apply(apply_func)
    
    """#### Data with all missing values labeled as Nan"""
    print("Data Info :\n")
    df_meta.info()
    
    """#### Checking for columns with all Nan values and dropping them"""
    df_meta.isnull().all()
    df_meta=df_meta.drop(columns=['category','fit','tech2','tech1', 'also_buy', 'image', 'feature', 'rank', 'also_view', 'main_cat', 'similar_item','date'], axis=1)
    per=(df_meta.isnull().sum()/df_meta.shape[0])*100
    
    """#### Dropping cols having more than 95% as Nan values"""
    
    df_meta=df_meta.drop(df_meta.columns[per>95],axis=1)
    
    """#### Dropping rows where all values are na(if exist)"""
    df_meta = df_meta.dropna(axis = 0, how ='all')
    
    """#### Checking for rows having more than 80% values as Nan """
    df_meta[df_meta.isnull().sum(axis=1) >=(df_meta.shape[1] * 0.8) ]
    
    """#### Dropping cols having more than 95% as Nan values"""
    df_meta=df_meta.drop(df_meta.columns[per>95],axis=1)
    
    """#### Dropping rows where all values are na(if exist)"""
    df_meta = df_meta.dropna(axis = 0, how ='all')
    
    """#### Checking for rows having more than 80% values as Nan """
    df_meta[df_meta.isnull().sum(axis=1) >=(df_meta.shape[1] * 0.8) ]
    df_meta.rename( columns = { 'asin': 'productID' }, inplace = True )
    
    #removing Nan from price
    df_meta = df_meta[df_meta['price'].notna()]
    per=(df_meta.isnull().sum()/df_meta.shape[0])*100
    
    #REMOVING $ SIGN FROM PRICE AND ',' IF EXISTS
    df_meta['price'] = df_meta['price'].str.replace('$', '')
    df_meta['price'] = df_meta['price'].str.replace(',', '')    
    df_meta = df_meta[df_meta['price'].map(len) < 10]
    df_meta['price'] = df_meta['price'].astype(float)
    #Rounding-Off The Prices
    df_meta.price = df_meta.price.round()
    #df_meta[['brand']]=df_meta[['brand']].fillna('unknown')
    per=(df_meta.isnull().sum()/df_meta.shape[0])*100
    df_meta.to_csv('meta_clean.csv',header = True, index = False)
    
    ###Visualization
    df_viz = pd.read_json('All_Beauty.json', lines=True)
    #Checking How Many Verified In Total
    df_viz['verified'].value_counts()
    
    fig1 = sns.countplot(x='verified', data=df_viz)
    fig1.figure.savefig("Plots/verified-unverified_count.png")
    
    fig2 = sns.countplot(x='overall', data=df_viz)
    fig2.figure.savefig("Plots/overall_count.png")
    
    most_rev_prod=df_viz['asin'].value_counts()[:50]
    plt.figure(figsize=(12,6))
    plt.xticks(rotation=45, ha='right')
    fig3 = sns.barplot(x=most_rev_prod.index,y=most_rev_prod.values, palette='Blues_r')
    plt.tight_layout()
    fig3.figure.savefig("Plots/bar-plot.png")
    
    """Top 50 most reviewed product"""
    fig4 = most_rev_prod.plot(kind='bar',figsize=(12,6), title="Top 50 Most Reviewed Products")
    fig4.figure.savefig('Plots/top50.png')
    
    most_rev_prod_rating=df_viz.groupby(['asin','overall']).count()[['reviewerID']]
    df_viz['unixReviewTime']= pd.to_datetime(df_viz['unixReviewTime'], unit='s')
    df_viz.sort_values('unixReviewTime', inplace=True)
    
    
    fig5 = df_viz.resample('M', on='unixReviewTime')['overall'].mean().plot(figsize=(10,6), title="Average reviews of all beauty product over time")
    fig5.figure.savefig('Plots/reviewOverTime.png')
    
    fig6 = df_viz.resample('M', on='unixReviewTime')['reviewerID'].count().plot(figsize=(10,6), title="Monthly Review counts over time")
    fig6.figure.savefig('Plots/MontlyReviewCount.png')
    
    df_viz['month']= df_viz['unixReviewTime'].apply(lambda x: x.month)
    df_viz['year']= df_viz['unixReviewTime'].apply(lambda x: x.year)

    fig7 = df_viz.groupby('month').count()['reviewerID'].plot(kind='bar', figsize=(10,6))
    fig7.figure.savefig('Plots/ReviewID_Month.png')
   
    #[print(i) for i in range(2015,2019)]
    
    fig8 = df_viz[df_viz['year']==2014].groupby('month').count()['reviewerID'].plot(kind='bar', figsize=(10,6))
    fig8.figure.savefig('Plots/ReviewID_2014.png')
  
    fig9 = df_viz[df_viz['year']==2015].groupby('month').count()['reviewerID'].plot(kind='bar', figsize=(10,6))
    fig9.figure.savefig('Plots/ReviewID_2015.png')
  
    fig10 = df_viz[df_viz['year']==2016].groupby('month').count()['reviewerID'].plot(kind='bar', figsize=(10,6))
    fig10.figure.savefig('Plots/ReviewID_2016.png')

    fig11 = df_viz[df_viz['year']==2017].groupby('month').count()['reviewerID'].plot(kind='bar', figsize=(10,6))
    fig11.figure.savefig('Plots/ReviewID_2017.png')

    fig12 = df_viz[df_viz['year']==2018].groupby('month').count()['reviewerID'].plot(kind='bar', figsize=(10,6))
    fig12.figure.savefig('Plots/ReviewID_2018.png')

    
    
    """### ALL_beauty cleaning"""
    df_beauty = pd.read_json('All_Beauty.json', lines=True)
    df_beauty.rename( columns = { 'overall': 'ratings','asin': 'productID' }, inplace = True )

    """### Handling missing value"""
    df_beauty.isnull().sum()
    (df_beauty.isnull().sum()/df_beauty.shape[0])*100
    
    """#### Removing unimportant features"""
    df_beauty=df_beauty[df_beauty['verified']]
    df_beauty.drop(columns=['reviewTime','style','image','reviewerName','vote','verified'], axis=1, inplace=True)
    df_beauty[['reviewText','summary']]=df_beauty[['reviewText','summary']].fillna('unknown')
    df_beauty.isnull().sum()
    
    """### Handling Duplicate records"""
    duplicates = df_beauty.duplicated()
    duplicates.sum()
    df_beauty[duplicates]
    #There are about 8713 duplicate records
 
    #Removing duplicate records
    df_beauty.drop_duplicates(inplace=True)
    duplicates = df_beauty.duplicated()
    duplicates.sum()
    
    """### Checking Outliers"""
    df_beauty[['ratings']]
    z = np.abs(stats.zscore(df_beauty[['ratings']]))
    #print(z)
    threshold = 3
    #print(np.where(z > 3))
    """There are no outliers."""
    
    #Saving The Cleaned Data
    df_beauty.to_csv('All_Beauty_Cleaned.csv',header=True,index= False)
    
    """### Merging the 2 datasets"""
    data_joined = df_beauty.merge (df_meta,on = 'productID',how ='inner')
    data_joined['review_count'] = data_joined['productID'].map(dict(data_joined.groupby('productID').count()['reviewerID']))
    data_joined.to_csv('Merged_Cleaned.csv',header=True,index= False)
    
    """### EDA"""
    new_data = pd.read_csv('Merged_Cleaned.csv')
    # Commented out IPython magic to ensure Python compatibility.
    # %matplotlib inline
    fig13 = data_joined.ratings.value_counts().plot(kind='barh')
    fig13.figure.savefig("Plots/bar.png")
    
    """### Text preprocessing"""
    
    def process_text():
        import re
        no_space = re.compile("[-.;:!\'?,\"()\[\]#&$*^@+=%|{}]")
        space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\n)|(\t)")
        preprocess = lambda text : " ".join(space.sub(" ", no_space.sub("", text.lower())).split())
        data_joined['title']=data_joined['title'].apply(preprocess)
        eng_stop_words = stopwords.words('english')
        rem_stopwords = lambda line : " ".join([word for word in line.split() if word not in eng_stop_words])
        data_joined['title']=data_joined['title'].apply(rem_stopwords)
        stemmer = PorterStemmer()
        stem_text = lambda line : " ".join([stemmer.stem(word) for word in line.split()])
        data_joined['title']=data_joined['title'].apply(stem_text)
        data_joined['title'].head(4)
        for index,text in enumerate(data_joined['reviewText'][35:40]):
          print('Review %d:\n'%(index+1),text)
        
        data_joined['reviewText']=data_joined['reviewText'].apply(preprocess)
        
        for index,text in enumerate(data_joined['reviewText'][35:40]):
          print('Review %d:\n'%(index+1),text)
        
        data_joined['reviewText']=data_joined['reviewText'].apply(rem_stopwords)
        data_joined['reviewText']=data_joined['reviewText'].apply(stem_text)
        data_joined[['title','reviewText']].groupby(by='title').agg(lambda x:' '.join(x))
        return 0
    return "EDA Successfull"