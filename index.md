# Recommendation Engine
A product recommendation is just like a filtering system that seeks to predict and show the items that a user would like to purchase. It may not be entirely accurate, but if it shows what user would like then basically it is doing its job right.
## Dataset Information
*All_beauty* dataset is taken from Per-category data released as Amazon review data in 2018.<br>
The dataset is divided into two parts:

* All_Beauty.json.gz 
  This dataset contains reviews (371,345 reviews).<br>
  **features used**<br>
  -	overall
  - verified
  - reviewerID
  - asin
  - reviewerName 	
  - reviewText 	
  - summary 	
  - unixReviewTime 	
  - vote
  
* meta_All_Beauty.json.gz
  This dataset contains  metadata (32,992 products)<br>
  **features used**<br>
  - title
  - description
  - also_buy
  - brand
  - rank
  - also_view
  - details
  - similar_item
  - price
  - asin

## Project Goals
1.  **Perform EDA**
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
	
3.  **Perform Sentiment Analysis**
	*	Data Preperation
		-	Remove unwanted characters (eg. ",.[]() etc.) and HTML tags (if present)
		-	Remove stopwords (eg. a an the in if etc.)
		-	Normalize the data by stemming (*PorterStemmer* is used here)
		-	Vectorize the data (*CountVectorizer* is used here)
		-	Scale the data if required.
	*	Model Building
		-	Use Binary Classification model for classifying +ve and -ve words
		-	Train the model on Train data.
	* Model Testing
		-	Test the model against the Test data
		-	Perform sanity check.

4.  **Perform Popularity based recommendation**
5.  **Apply Content Based filtering**
6.  **Apply Model-based collaborative filtering**
7.  **Apply Collaberative filtering (Item-Item recommedation)**

---

### Citation
**Justifying recommendations using distantly-labeled reviews and fined-grained aspects**

Jianmo Ni, Jiacheng Li, Julian McAuley
Empirical Methods in Natural Language Processing (EMNLP), 2019
 
