# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re

def final_data():
    df = pd.read_json('meta_All_Beauty_clean.json.gz')
    
    df.drop(['also_buy', 'rank', 'also_view', 'similar_item'], axis=1, inplace=True)
    
    prop_set = set()
    
    get_item = lambda x : [prop_set.add(item) for item in x.keys()]
    
    df['details'].apply(get_item)
    
    for item in prop_set:
      df[item.strip(': \n')] = df['details'].apply(lambda x : x.get(item))
    
    df.drop('details', axis=1, inplace=True)
    
    df.isna().sum()/df.shape[0]*100
    
    """Filtering Columns with less than 75% missing values"""
    
    df = df.loc[:,df.columns[~(df.isna().sum()/df.shape[0]*100 >75).values]]
    
    """Droping `UPC` and `Shipping Weight` because they are more related to delivery service rathar than actual product. And `ASIN` is redundant."""
    
    #df.drop(['UPC',	'Shipping Weight', 'ASIN'], axis=1, inplace=True)
    df.drop(['UPC',	'Shipping Weight'], axis=1, inplace=True)
    
    df['price'] = df.loc[:,'price'].apply(lambda x : x.strip(' $') if x is not None else np.nan)
    
    df['price'].isna().sum()/df.shape[0]*100
    
    price_filter = lambda x : (float(x) if len(x)<=6 else np.nan) if x is not None and not isinstance(x, float) else x
    df['price'] = df.loc[:,'price'].apply(price_filter).astype(float)
    
    desc_filter = lambda x : " ".join(x) if x is not None else np.nan
    df['description'] = df.loc[:,'description'].apply(desc_filter)
    
    df.isna().sum()/df.shape[0]*100
    
    df.dropna(subset=['title'], inplace=True)
    
    df['description'].fillna(df['title'], inplace=True)
    df['brand'].fillna("", inplace=True)
    
    space = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\n)|(\t)|(;)|(&amp)")
    preprocess = lambda text : " ".join(space.sub(" ", text).split())
    
    df['title']=df['title'].apply(preprocess)
    df['description']=df['description'].apply(preprocess)
    df['price'].fillna(df['price'].mean(), inplace=True)
    df['price'] = df['price'].apply(lambda x: round(x, 2))
    
    df.isna().sum()/df.shape[0]*100
    
    df.to_json('meta_All_Beauty_clean2.json.gz', compression='gzip')
    
    df2 = pd.read_json('All_Beauty_final.json.gz')
    
    one_df = df.merge(df2, on='asin')
    
    one_df.drop_duplicates(subset=['asin', 'reviewerID'], inplace=True)
    
    
    one_df.to_json('All_Beauty_One_final.json.gz', compression='gzip')
    
    one_df.to_csv('All_Beauty_One_final.csv')
    return "Merged Data File Created Succesfully"
