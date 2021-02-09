[![python-3.7.9](https://img.shields.io/badge/python-3.7.9-blue)](https://www.python.org/downloads/release/python-379/)
[![scikit-learn-0.23.2](https://img.shields.io/badge/scikit--learn-0.23.2-blue)](https://pypi.org/project/scikit-learn/0.23.2/)
[![pandas-1.1.3](https://img.shields.io/badge/pandas-1.1.3-blue)](https://pypi.org/project/pandas/1.1.3/)
[![nltk-3.5](https://img.shields.io/badge/nltk-3.5-blue)](https://pypi.org/project/nltk/3.5/)
[![scikit-surprise-1.1.1](https://img.shields.io/badge/scikit--surprise-1.1.1-blue)](https://pypi.org/project/scikit-surprise/1.1.1/)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-green)](https://www.gnu.org/licenses/gpl-3.0)

# Recommendation Engine
A product recommendation is just like a filtering system that seeks to predict and show the items that a user would like to purchase. It may not be entirely accurate, but if it shows what user would like then basically it is doing its job right.
## Dataset Information
*All_beauty* dataset is taken from Per-category data released as Amazon review data in 2018.<br>
The dataset is divided into two parts:

* All_Beauty.json.gz <br>
  This dataset contains reviews (371,345 reviews).<br>
  **features**
	*	overall - rating of the product
	*	verified - Verified review or not
	*	reviewerID - ID of the reviewer
	*	asin - ID of the product
	*	reviewerName - name of the reviewer
	*	reviewText - text of the review
	*	summary - summary of the review
	*	unixReviewTime - time of the review (unix time)
	*	vote - helpful votes of the review
  
* meta_All_Beauty.json.gz<br>
  This dataset contains  metadata (32,992 products)<br>
  **features**
 	*	title - name of the product
 	*	description - description of the product
 	*	also_buy -
 	*	brand - brand name
 	*	rank - sales rank information
 	*	also_view -
 	*	details - product and shipping details
 	*	similar_item - similar product table
 	*	price - price in US dollars (at time of crawl)
 	*	asin - ID of the product

## Project Details
1.  **EDA**
	*	Check for missing data and other mistakes.
	*	Gain maximum insight into the data set and its underlying structure.
	*	Uncover a parsimonious model, one which explains the data with a minimum number of predictor variables.
	*	Check assumptions associated with any model fitting or hypothesis test.
	*	Create a list of outliers or other anomalies.
	*	Find parameter estimates and their associated confidence intervals or margins of error.
	*	Identify the most influential variables.
	
2.  **Data Cleaning & Preprocessing**
	*	Impute missing data.
	*	Remove noise from Data.
	
3.  **Generating Sentiments**
	*	Data Preperation
		-	Remove unwanted characters (eg. ",.[]() etc.) and HTML tags (if present)
		-	Remove stopwords (eg. a an the in if etc.)
		-	Normalize the data by stemming (*PorterStemmer* is used here)
		-	Vectorize the data (*CountVectorizer* is used here)
		-	Scale the data if required.
		-	Sampling data for training and testing purposes.
	*	Building Model
		-	Use Classification model for classifying positive, negative and neutral words.
		-	Train the model on Train data.
	* 	Model Testing
		-	Test the model against the Test data
	*	Building Final model
	*	Generating sentiments on actual data.

4.  **Apply Popularity based filtering**
	*	Use`recommend` from  `recommendation_filters.popularity_filter` module, to get a list of most popular products.
5.  **Apply Content Based filtering**
	*	Import `recommendation_filters.content_based_filter` module.
	*	Use `cbf_data` to prepare data for content based filtering.
	*	Use `indices` to generate index Series.
	*	Use `cosine_sim` to generate similarity matrix.
	*	Use `recommend` to get recommendations.
6.  **Apply Model-based collaborative filtering**
	*	Import `models.collaborative_model_based` module.
	*	Use `train` to train the model.
	*	Use `recommend` to get recommendations.
	
7.  **Apply Hybrid filtering**
	*	Combination of popularity based, content based, model-based collaborative filters results in a very robust filter that can recommend products in most conditions even in worst condition that is when input data is not present in model.
	*	One of many approaches is used in main.py
	
---
### Citation
**Justifying recommendations using distantly-labeled reviews and fined-grained aspects**

Jianmo Ni, Jiacheng Li, Julian McAuley
Empirical Methods in Natural Language Processing (EMNLP), 2019
