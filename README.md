# A Sentiment Analysis on Brands and Product Emotions
![performing-twitter-sentiment-analysis1](https://user-images.githubusercontent.com/117165965/225626996-b827379b-04c7-4c15-a9ac-7e1cbdf7083d.jpg)
## Defining the Question
### a) Specifying the Data Analytic Question
Introduction: Company Goku is going to launch a new mobile phone soon. They are worried about how people will react to it, and they want to keep an eye on its popularity.

Problem Statement: The problem at hand is to build an NLP model that can analyze Twitter sentiment about Apple and Google products, which have the largest market dominance in the industry. The dataset comprises over 9,000 Tweets that have been rated by human raters as positive, negative, or neutral. This project aims to address the challenge faced by tech companies in understanding customer sentiment towards their products and gain valuable insights from customer feedback. This will improve customer satisfaction and enable Company Goku to stay ahead of the competition in the highly competitive tech industry.

Main Objective: Create a Sentiment Analysis model that can accurately predict whether tweets about the phone are positive, negative, or neutral.
### b) Defining the Metric for Success
The model will be considered a success if the model has an accuracy of 75%, a recall of 70% and macro-average recall 75%
### c) Recording the Experimental Design
1.Data Collection
2.Reading the Data
3.Checking the Data

4.External Data Source Validation

5.Data Cleaning

6.Exploratory Data Analysis

7.Data Modeling

8.Observations from the model

9.Conclusion

10.Recommendations

11.Future Improvement Ideas
### d) Data Understanding
1) Text_data : This column contains tweets in terms of text data about the different brands and products.
2) Emotion_in_tweet_is directed_at : This column contains data on the different services and products of different brands that the text is associated with. It contains a meaningful relationship between the brand and potential customers which describes basic emotions.
3) Is_therea_an_emotion directed at the brand :	Contains the classification based on the Emotion_in_tweet_is directed_at column, for example positive, negative, and neutral.

## Reading the Data
# Loading the required libraries 
import pandas as pd


import numpy as np


import seaborn as sns


import matplotlib.pyplot as plt



import re


from wordcloud import WordCloud


from textblob import TextBlob



import nltk


nltk.download('punkt')


from nltk.tokenize import RegexpTokenizer


from nltk.corpus import stopwords


from nltk.stem import PorterStemmer


from nltk.tokenize import TweetTokenizer


from nltk import FreqDist


import string


from nltk.tokenize import word_tokenize


from sklearn.preprocessing import LabelEncoder



nltk.download('wordnet')


from nltk.stem.wordnet import WordNetLemmatizer



from sklearn.model_selection import train_test_split



import warnings

### Loading the data 
tweet_df = pd.read_csv("judge-1377884607_tweet_product_company.csv",encoding='ISO-8859-1')
tweet_df.head()

	tweet_text	emotion_in_tweet_is_directed_at	is_there_an_emotion_directed_at_a_brand_or_product
0	.@wesley83 I have a 3G iPhone. After 3 hrs twe...	iPhone	Negative emotion
1	@jessedee Know about @fludapp ? Awesome iPad/i...	iPad or iPhone App	Positive emotion
2	@swonderlin Can not wait for #iPad 2 also. The...	iPad	Positive emotion
3	@sxsw I hope this year's festival isn't as cra...	iPad or iPhone App	Negative emotion
4	@sxtxstate great stuff on Fri #SXSW: Marissa M...	Google	Positive emotion

### Checking the shape of tweet_df

print(tweet_df.shape)

## 3. Checking the Data

An understanding of the data is critical for optimum results. The size of the dataset as well as the type of data is important when perfoming analysis and coming up with the best models for the data

### Previewing the top of our dataset
tweet_df.head()

tweet_text	emotion_in_tweet_is_directed_at	is_there_an_emotion_directed_at_a_brand_or_product
0	.@wesley83 I have a 3G iPhone. After 3 hrs twe...	iPhone	Negative emotion
1	@jessedee Know about @fludapp ? Awesome iPad/i...	iPad or iPhone App	Positive emotion
2	@swonderlin Can not wait for #iPad 2 also. The...	iPad	Positive emotion
3	@sxsw I hope this year's festival isn't as cra...	iPad or iPhone App	Negative emotion
4	@sxtxstate great stuff on Fri #SXSW: Marissa M...	Google	Positive emotion
