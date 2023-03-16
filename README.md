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

### Previewing the bottom of our dataset

tweet_df.tail()

### Checking whether each column has an appropriate datatype

tweet_df.info()

 0   tweet_text                                          9092 non-null   object
 

 1   emotion_in_tweet_is_directed_at                     3291 non-null   object
 
 
 2   is_there_an_emotion_directed_at_a_brand_or_product  9093 non-null   object
 
 
It can be noted that some columns in tweet_df contain null values. This will be tidied under data cleaning before any analysis is done.

### Check for the percentage of missing data

tweet_df.isna().sum()*100/len(tweet_df)

tweet_text                                             0.010997

emotion_in_tweet_is_directed_at                       63.807324

is_there_an_emotion_directed_at_a_brand_or_product     0.000000

dtype: float64

The column product has 5802 missing values which is 63.8% of the data. This is a significant amount of the data and even though the column may be usefull, the objective is to predict the sentiment (positive, negative, or neutral) expressed in the tweets related to a particular topic or brand and so the column should may be dropped later.

### Checking for emotion_in_tweet_is_directed_at value counts

value_counts1 = tweet_df['emotion_in_tweet_is_directed_at'].value_counts()
value_counts1

iPad                               946

Apple                              661

iPad or iPhone App                 470

Google                             430

iPhone                             297

Other Google product or service    293

Android App                         81

Android                             78

Other Apple product or service      35

Name: emotion_in_tweet_is_directed_at, dtype: int64

### Checking for is_there_an_emotion_directed_at_a_brand_or_prouct value counts

value_counts2 = tweet_df['is_there_an_emotion_directed_at_a_brand_or_product'].value_counts()
value_counts2

No emotion toward brand or product    5389

Positive emotion                      2978

Negative emotion                       570

I can't tell                           156

Name: is_there_an_emotion_directed_at_a_brand_or_product, dtype: int64

The dataset is now ready for cleaning and farther analysis

## 4. Cleaning the Dataset

### 4.1 Renaming columns 

#### Renaming the columns 

tweet_df.rename(columns = {'tweet_text': 'Tweet','emotion_in_tweet_is_directed_at':'Product',
                           'is_there_an_emotion_directed_at_a_brand_or_product':'Emotion'},
                inplace = True)
		
tweet_df.tail()

The dataframe contains very long column names that can be made more easier to deal with by renaming them.

## Deleting rows with no tweet data
### # Deleting rows with no tweet data

df_clean = tweet_df.dropna(subset=['Tweet'])

### Confirming that the row was deleted

df_clean.info()

0   Tweet    9092 non-null   object

1   Product  3291 non-null   object
 
2   Emotion  9092 non-null   object

It was noted that the column Tweet had missing information. Seeing as this was only one row in a dataset containing 9000+ rows and there is no way to find out the contents of the tweet, it was dropped.

### Deleting tweet not in English

df_clean = df_clean.drop(9092, axis = 0)
df_clean

### Deleting rows with "I can't tell"

df_clean = df_clean[df_clean.Emotion != "I can't tell"]

df_clean['Emotion'].unique()

array(['Negative emotion', 'Positive emotion',
       'No emotion toward brand or product'], dtype=object)
       

Since no information can be gained from the narration **"I can't tell"** ,the 156 rows were dropped.

## Filling missing values
### Checking for unique itemns under product

df_clean['Product'].unique()

array(['iPhone', 'iPad or iPhone App', 'iPad', 'Google', nan, 'Android',

       'Apple', 'Android App', 'Other Google product or service',
       
       'Other Apple product or service'], dtype=object)
       

# Creating a function for filling nan values in product


def categorize_devices(df, col_name, keywords=['iphone', 'apple', 'ipad', 'android', 'google']):

    # Convert column to lowercase string
    
    df[col_name] = df[col_name].astype(str).str.lower()
    
    # Initialize category column with 'Unknown'
    df['category'] = 'Unknown'
    
    # Categorize each row based on keywords
    for i, row in df.iterrows():
        text = row[col_name]
        if any(keyword in text for keyword in ['iphone', 'apple', 'ipad']):
            df.loc[i, 'category'] = 'Apple'
        elif 'android' in text:
            df.loc[i, 'category'] = 'Android'
        elif 'google' in text:
            df.loc[i, 'category'] = 'Google'
    
    return df

