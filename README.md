# A Sentiment Analysis on Brands and Product Emotions
![performing-twitter-sentiment-analysis1](https://user-images.githubusercontent.com/117165965/225626996-b827379b-04c7-4c15-a9ac-7e1cbdf7083d.jpg)
# Introduction
Company Goku is going to launch a new mobile phone soon. They are worried about how people will react to it, and they want to keep an eye on its popularity.

# Problem
The problem at hand is to build an NLP model that can analyze Twitter sentiment about Apple and Google products, which have the largest market dominance in the industry. The dataset comprises over 9,000 Tweets that have been rated by human raters as positive, negative, or neutral. This project aims to address the challenge faced by tech companies in understanding customer sentiment towards their products and gain valuable insights from customer feedback. This will improve customer satisfaction and enable Company Goku to stay ahead of the competition in the highly competitive tech industry.


# Data

The data is provided by [CrowdFlower](https://data.world/crowdflower) and is available for download from [data.world](https://data.world/crowdflower/brands-and-product-emotions)

# Methodology

* Exploratory Data Analysis to understand the data.This included plotting visualizations like word cloud to get a clearer picture of the nature of text and the most common words. Afterwards, histograms were also plotted to establish the distribution of the three sentiment classes. Here, a massive imbalance was noted between the classes, and was dealt with later in preprocessing.


* Preprocess the text data: First, some columns that would not be required for the modeling process were dropped. Then, tokenization was performed on the textual data to break it down into individual words or tokens. Stop words, such as "a", "an", "the", etc., were removed from the text since they do not add much meaning to the text. Lemmatization was also performed to reduce the inflected forms of words to their base form, so that similar words with the same meaning would be treated as the same. Stemming was also performed. The target variable y was encoded to convert categorical data to numerical data. Then, the feature matrix X was vectorized, so that the textual data could be represented in a numerical form, which could be used as input for the machine learning models. Finally SMOTE was done to deal with the class imbalance.

* Model Selection.The following models were developed: Multinomial Naive Bayes Model had a recall score of 0.23 and a f1 score of 0.37.
Decision Tree Classifier had a recall score of 0.26 and a f1 score of 0.54.
Support Vector Machine (SVM) implemented had a recall score of 0.42 and a f1 score of 0.62. The scores remained the same despite conducting hyper parameter tunning.
Random Forest Classifier had a recall score of 0.23 and a f1 score of 0.64.
From the scores. above, SVM was selected as it had a balanced and better score compared to the other models..

# Conclusion



# Recommendations



# Future Work
* Acquire more labeled Tweets to improve the model

The dataset used to train this model is relatively small, about 9000 tweets.  Retraining the model on a larger dataset should improve its performance.

* Expand the scope of the sentiment analysis monitoring

There is plenty of other publicly available text data that can be acquired and monitored for sentiment.  This data may be on other social media platforms or public forums, or could be product reviews. While product reviews often have an associated rating, that rating may differ from the overall sentiment of the review.  Classifying this other data will require a new model because its structure would differ from a tweet.

* Enhance the level of detail in the analysis of emotions.

Some text data is going to be more negative or more positive than others.  By creating a scale from very negative to somewhat negative to neutral to somewhat positive to very positive, more nuance will be able to be found in the sentiment analysis, and actions can be taken based on the severity of the situation.
