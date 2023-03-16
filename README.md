# A Sentiment Analysis on Brands and Product Emotions
![performing-twitter-sentiment-analysis1](https://user-images.githubusercontent.com/117165965/225626996-b827379b-04c7-4c15-a9ac-7e1cbdf7083d.jpg)
# Introduction
Company Goku is going to launch a new mobile phone soon. They are worried about how people will react to it, and they want to keep an eye on its popularity.

# Problem
The problem at hand is to build an NLP model that can analyze Twitter sentiment about Apple and Google products, which have the largest market dominance in the industry. The dataset comprises over 9,000 Tweets that have been rated by human raters as positive, negative, or neutral. This project aims to address the challenge faced by tech companies in understanding customer sentiment towards their products and gain valuable insights from customer feedback. This will improve customer satisfaction and enable Company Goku to stay ahead of the competition in the highly competitive tech industry.


# Data

The data is provided by [CrowdFlower](https://data.world/crowdflower) and is available for download from [data.world](https://data.world/crowdflower/brands-and-product-emotions)

# Methodology

* Exploratory Data Analysis to understand the data.


* Preprocess the text data: including removing unwanted punctuation, ..... and tokenization.


* Model Selection: A series of classification models were run on the data and evaluated on three metrics: Average Macro Recall, a balanced Recall score for each class, and overall accuracy.

# Conclusion



# Recommendations



# Future Work
* Acquire more labeled Tweets to improve the model

The dataset used to train this model is relatively small, about 9000 tweets.  Retraining the model on a larger dataset should improve its performance.

* Expand the scope of the sentiment analysis monitoring

There is plenty of other publicly available text data that can be acquired and monitored for sentiment.  This data may be on other social media platforms or public forums, or could be product reviews. While product reviews often have an associated rating, that rating may differ from the overall sentiment of the review.  Classifying this other data will require a new model because its structure would differ from a tweet.

* Enhance the level of detail in the analysis of emotions.

Some text data is going to be more negative or more positive than others.  By creating a scale from very negative to somewhat negative to neutral to somewhat positive to very positive, more nuance will be able to be found in the sentiment analysis, and actions can be taken based on the severity of the situation.
