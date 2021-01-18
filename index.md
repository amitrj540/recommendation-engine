# Recommendation Engine
A product recommendation is just like a filtering system that seeks to predict and show the items that a user would like to purchase. It may not be entirely accurate, but if it shows what user would like then basically it is doing its job right.
## Dataset Information
*All_beauty* dataset is taken from Per-category data released as Amazon review data in 2018.<br>
The dataset is divided into two parts:
- All_Beauty.json.gz 
  This dataset contains reviews (371,345 reviews).<br>
  **features used**<br>
  * overall
  * verified
  * reviewerID
  * asin
  * reviewerName 	
  * reviewText 	
  * summary 	
  * unixReviewTime 	
  * vote
  
- meta_All_Beauty.json.gz
  This dataset contains  metadata (32,992 products)<br>
  **features used**<br>
  * title
  * description
  * also_buy
  * brand
  * rank
  * also_view
  * details
  * similar_item
  * price
  * asin

## Project Goals
1.  Perform EDA on the dataset
2.  Data Cleaning & Preprocessing
3.  Perform Sentiment Analysis
4.  Perform Popularity based recommendation
5.  Apply Content Based filtering
6.  Apply Model-based collaborative filtering
7.  Apply Collaberative filtering (Item-Item recommedation)

### Citation
**Justifying recommendations using distantly-labeled reviews and fined-grained aspects**

Jianmo Ni, Jiacheng Li, Julian McAuley
Empirical Methods in Natural Language Processing (EMNLP), 2019
