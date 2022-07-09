#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:08:26 2022

@author: rosskearney
"""
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import matplotlib.pyplot as mlpt
import nltk
import unicodedata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import csv
import os
from datetime import date
from sklearn import linear_model

# =============================================================================
#  creating emplty lists and dataframe to begin
# =============================================================================

df = pd.DataFrame()

news_dates=[]
news_titles=[]
news_titlesList=[]
sentiments=[] 
pub_dates=[]
int_dates=[]

searchTerm = 'bitcoin'


# =============================================================================
# SCRAPING THE 'FT' FOR THE RESULTS ON THE FIRST 20 PAGES
# =============================================================================
for page in range(1,20):
    url = "https://www.ft.com/search?q="+str(searchTerm)+"&page="+str(page)+"&sort=relevance"
    result=requests.get(url)
    reshult=result.content
    soup=BeautifulSoup(reshult, "lxml")
    for title in soup.findAll("div",{"class":"o-teaser__heading"}):
        titles=title.findAll(text=True)
        fullTitle = ''.join(titles)
        news_titles.append(fullTitle)

# Finding dates of articles published
    for dates in soup.findAll("time",{"class":"o-teaser__timestamp-date"}):
        if dates.has_attr('datetime'):
            pubDate = (dates['datetime'][0:10])
            d = pubDate
            newd = re.sub('[^0-9]', '', str(d))
            int_dates.append(newd)
            pub_dates.append(pubDate)


# ensure lengths are all same, i.e. every article has a corresponding date
print(len(int_dates))
print(len(news_titles))
print(len(pub_dates))
print(len(sentiments))


# =============================================================================
# Sentiment analysis
# =============================================================================
for headline in news_titles:
    sid_obj= SentimentIntensityAnalyzer()
     # takes the 'compound' i.e aggregate sentiment from pos, neg, neutral.
    sent = sid_obj.polarity_scores(headline)['compound']
    news_titlesList.append(headline)
    sentiments.append(sent)
    

# =============================================================================
# Create dataframe
# =============================================================================


FTdf = pd.DataFrame(
    {'Date': pub_dates,
     'Title': news_titles,
     'IntDates' : int_dates})

# re sort, oldest to newest, reset index 
FTdf = FTdf.sort_values(by='IntDates', ascending=True)

FTdf = FTdf.reset_index()

FTdf.drop('index', inplace=True, axis=1)
FTdf.drop('IntDates', inplace=True, axis=1)


# =============================================================================
#                      CREATING 'Tweets.csv' FILE 
# =============================================================================
os.remove('FinTimes.csv')

FTdf.to_csv("FinTimes.csv")
FTdata=pd.DataFrame(columns=['Date','Title'])
total=100
index=0
for index,row in FTdf.iterrows():
    stre=row["Title"]
    my_new_string = re.sub('[^ a-zA-Z0-9]', '', stre)
    temp_df = pd.DataFrame([[FTdf["Date"].iloc[index], 
                            my_new_string]], columns = ['Date','Title'])
    FTdata = pd.concat([FTdata, temp_df], axis = 0).reset_index(drop = True)


# =============================================================================
#           CREATING 'DF' WITH NO DAYS REPEATING
# =============================================================================


FTdataF=pd.DataFrame(columns=['Date','Title'])
# i = 0
indx=0
get_title=""
for i in range(0,len(FTdata)-1):
    get_date=FTdata.Date.iloc[i]
    next_date=FTdata.Date.iloc[i+1]
    if(str(get_date)==str(next_date)):
        get_title=get_title+FTdata.Title.iloc[i]+" "
    if(str(get_date)!=str(next_date)):
        get_title = FTdata.Title.iloc[i] # **********
        temp_df = pd.DataFrame([[get_date, 
                                get_title]], columns = ['Date','Title'])
        FTdataF = pd.concat([FTdataF, temp_df], axis = 0).reset_index(drop = True)
        get_title=" "



# =============================================================================
#                       GETTING STOCK DATA
# =============================================================================

StartDate = FTdata['Date'].iloc[0]
EndDate = FTdata['Date'].iloc[-1]

ticker1 = 'BTC-USD'

read_stock_p = yf.download(ticker1, start=StartDate, end=EndDate) 

# RESETTING INDEX AWAY FROM DATES
read_stock_p = read_stock_p.reset_index()



# =============================================================================
# ADDING 'Prices' COLUMN TO TWEETS DATA AND PUTTING CORRESPONDING STOCK PRICE EACH DAY
# =============================================================================

FTdataF['Prices']=""


indx=0
for i in range (0,len(FTdataF)):
    for j in range (0,len(read_stock_p)):
        get_title_date=FTdataF.Date.iloc[i]

        get_stock_date=str(read_stock_p.Date.iloc[j])[0:10]
        if(str(get_stock_date)==str(get_title_date)):
            FTdataF['Prices'].iloc[i] = read_stock_p.Close[j]

            
# =============================================================================
#       PUTTING PREV. CLOSE INTO DATES WITH CLOSED MARKETS
# =============================================================================

FTdataF = FTdataF.mask(FTdataF == '')
    
FTdataF['Prices'].fillna(method='ffill', inplace=True)


FTdataF["Comp"] = ''
FTdataF["Negative"] = ''
FTdataF["Neutral"] = ''
FTdataF["Positive"] = ''
FTdataF


# =============================================================================
#               CREATING SENTIMENT COLUMNS
# =============================================================================
nltk.download('vader_lexicon')

sentiment_i_a = SentimentIntensityAnalyzer()
for indexx, row in FTdataF.T.iteritems():
    try:
        sentence_i = unicodedata.normalize('NFKD', FTdataF.loc[indexx, 'Title'])
        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
        FTdataF['Comp'].iloc[indexx] = sentence_sentiment['compound']
        FTdataF['Negative'].iloc[indexx] = sentence_sentiment['neg']
        FTdataF['Neutral'].iloc[indexx] = sentence_sentiment['neu']
        FTdataF['Positive'].iloc[indexx] = sentence_sentiment['compound']

    except TypeError:
        print (indexx)


# =============================================================================
#               PERCENTAGE POS/NEG TWEETS
# =============================================================================

posi=0
nega=0
for i in range (0,len(FTdataF)):
    get_val=FTdataF.Comp[i]
    if(float(get_val)<(0)):
        nega=nega+1
    if(float(get_val>(0))):
        posi=posi+1
    
posper=(posi/(len(FTdataF)))*100
negper=(nega/(len(FTdataF)))*100
print("% of positive titles= ",posper)
print("% of negative titles= ",negper)
arr=np.asarray([posper,negper], dtype=int)
mlpt.pie(arr,labels=['positive','negative'])
mlpt.plot()



# =============================================================================
#               MACHINE LEARING FOR STOCK PREDICTION STARTS HERE
# =============================================================================

#  create new datafram
FTdf_=FTdataF[['Date','Prices','Comp','Negative','Neutral','Positive']].copy()
FTdf_

FTdf_['Prices'].fillna(method='bfill', inplace=True)

# split dataframe into training and testing
percentage = round(len(FTdf_)/100*70) 

len(FTdf_)-percentage

test=FTdf_.iloc[percentage:len(FTdf_),:]
train = FTdf_.iloc[:percentage]



# Create training dataframe
sentiment_score_list = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray([FTdf_.loc[date, 'Negative'],FTdf_.loc[date, 'Positive']])
    sentiment_score_list.append(sentiment_score)
numpy_df_train = np.asarray(sentiment_score_list)

print(numpy_df_train)



# Create testing dataframe
sentiment_score_list = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray([FTdf_.loc[date, 'Negative'],FTdf_.loc[date, 'Positive']])
    sentiment_score_list.append(sentiment_score)
numpy_df_test = np.asarray(sentiment_score_list)

print(numpy_df_test)


# =============================================================================
# 
# =============================================================================

y_train = pd.DataFrame(train['Prices'])

y_test = pd.DataFrame(test['Prices'])
print(y_train)

# =============================================================================
#  FITTING SENTIMENTS (INDEPENDANT VARIABLE) AND PRICES (DEPENDANT VARIABLE)
#                           AND PRICE PREDICTIONS
# =============================================================================


rf = RandomForestRegressor()
rf.fit(numpy_df_train, y_train) 

prediction = rf.predict(numpy_df_test)

print(prediction)


test_start_index = test.index[0]
test_end_index = test.index[-1]


idx=np.arange(int(test_start_index),int(test_end_index)+1)
predictions_df_ = pd.DataFrame(data=prediction[0:], index = idx, columns=['Prices'])

predictions_df_

# =============================================================================
#           PLOTTING RANDOM FOREST PREDICTION VS ACTUAL PRICES
# =============================================================================
ax = predictions_df_.rename(columns={"Prices": "predicted_price"}).plot(title='Random Forest predicted prices')#predicted value
ax.set_xlabel("Indexes")
ax.set_ylabel("Stock Prices")
fig = y_test.rename(columns={"Prices": "actual_price"}).plot(ax = ax).get_figure()#actual value
fig.savefig("random forest.png")




reg = LinearRegression()
reg.fit(numpy_df_train, y_train)
print(reg.score(numpy_df_train, y_train))
reg.predict(numpy_df_test)

FTdf_

# =============================================================================
# 
# =============================================================================


percentage = round(len(FTdf_)/100*70) 

len(FTdf_)-percentage

test=FTdf_.iloc[percentage:len(FTdf_),:]
train = FTdf_.iloc[:percentage]


train_data_start = train['Date'].iloc[0]
train_data_end = train['Date'].iloc[-1]
test_data_start = test['Date'].iloc[0]
test_data_end = test['Date'].iloc[-1]

trainStart = ((FTdf_[FTdf_['Date']==train_data_start].index.values))[0]
trainEnd = ((FTdf_[FTdf_['Date']==train_data_end].index.values))[0]
testStart = ((FTdf_[FTdf_['Date']==test_data_start].index.values))[0]
testEnd = ((FTdf_[FTdf_['Date']==test_data_end].index.values))[0]

train = FTdf_.loc[trainStart : trainEnd]
test = FTdf_.loc[testStart : testEnd]



#       train
list_of_sentiments_score = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray([FTdf_.loc[date, 'Comp']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_train = np.asarray(list_of_sentiments_score)

#       test
list_of_sentiments_score = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray([FTdf_.loc[date, 'Comp']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_test = np.asarray(list_of_sentiments_score)

y_train = pd.DataFrame(train['Prices'])
y_test = pd.DataFrame(test['Prices'])


# =============================================================================
# Random Forrest predictor
# =============================================================================

rf = RandomForestRegressor()
rf.fit(numpy_dataframe_train, train['Prices'])

prediction=rf.predict(numpy_dataframe_test)


len(prediction)
# ****** UPDATED CODE **********
idx = pd.date_range(test_data_start, test_data_end) # ORIGINAL 



tt = test['Date'].tolist()
idx2 = pd.to_datetime(pd.Series(tt))



predictions_df = pd.DataFrame(data=prediction[0:], index = idx2, columns=['Prices'])
predictions_df['Prices'] = predictions_df['Prices']
newData = test['Prices'].to_numpy()
testNew = pd.DataFrame(data=newData, index = idx2, columns=['Prices'])
predictions_df['actual_value'] = testNew['Prices']
predictions_df.columns = ['predicted_price', 'actual_price']
predictions_df.plot(title = 'Random Forest Regressor plotting predicted vs actual prices')
predictions_df['predicted_price'] = predictions_df['predicted_price']
print(rf.score(numpy_dataframe_train, train['Prices']))


# =============================================================================
# Linear regression prediction
# =============================================================================

tt3 = test['Date'].tolist()
idx3 = pd.to_datetime(pd.Series(tt3))

regr = linear_model.LinearRegression()
regr.fit(numpy_dataframe_train, train['Prices'])   
prediction = regr.predict(numpy_dataframe_test)



# idx = pd.date_range(test_data_start, test_data_end)
predictions_df = pd.DataFrame(data=prediction[0:], index = idx3, columns=['Prices'])
predictions_df['Prices'] = predictions_df['Prices']
newData = test['Prices'].to_numpy()
testNew = pd.DataFrame(data=newData, index = idx3, columns=['Prices'])
predictions_df['actual_value'] = testNew['Prices']
predictions_df.columns = ['predicted_price', 'actual_price']
predictions_df.plot()
predictions_df['predicted_price'] = predictions_df['predicted_price']
print(rf.score(numpy_dataframe_train, train['Prices']))




# =============================================================================
#               PREPARING START/END DATES FOR PREDICTIONS
# =============================================================================
percentage = round(len(FTdf_)/100*70) 

len(FTdf_)-percentage

test=FTdf_.iloc[percentage:len(FTdf_),:]
train = FTdf_.iloc[:percentage]


train_data_start = train['Date'].iloc[0]
train_data_end = train['Date'].iloc[-1]
test_data_start = test['Date'].iloc[0]
test_data_end = test['Date'].iloc[-1]

trainStart = ((FTdf_[FTdf_['Date']==train_data_start].index.values))[0]
trainEnd = ((FTdf_[FTdf_['Date']==train_data_end].index.values))[0]
testStart = ((FTdf_[FTdf_['Date']==test_data_start].index.values))[0]
testEnd = ((FTdf_[FTdf_['Date']==test_data_end].index.values))[0]

train = FTdf_.loc[trainStart : trainEnd]
test = FTdf_.loc[testStart : testEnd]


FTdataframe = FTdf_



# =============================================================================
#       LISTS OF SENTIMENT SCORES
# =============================================================================
list_of_sentiments_score = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray([FTdataframe.loc[date, 'Comp'],FTdataframe.loc[date, 'Negative'],FTdataframe.loc[date, 'Neutral'],FTdataframe.loc[date, 'Positive']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_train = np.asarray(list_of_sentiments_score)


list_of_sentiments_score = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray([FTdataframe.loc[date, 'Comp'],FTdataframe.loc[date, 'Negative'],FTdataframe.loc[date, 'Neutral'],FTdataframe.loc[date, 'Positive']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_test = np.asarray(list_of_sentiments_score)


# =============================================================================
#           NO IDEA WHAT THIS IS DOING
# =============================================================================
rf = RandomForestRegressor(random_state=25)
rf.fit(numpy_dataframe_train, train['Prices'])


# =============================================================================
#        TAKING LIST OF SENTIMENTS AND GIVING STOCK PREDICTIONS
# =============================================================================

prediction = rf.predict(numpy_dataframe_test)
prediction_list = []
prediction_list.append(prediction)

# =============================================================================
#          CREATING DATAFRAME WITH PREDICTED AND ACTUAL STOCK PRICES
# =============================================================================

tt3 = test['Date'].tolist()
idx3 = pd.to_datetime(pd.Series(tt3))


predictions_dataframe_list = pd.DataFrame(data=prediction[0:], index = idx3, columns=['Prices'])

test_array = test["Prices"].tolist()
predictions_dataframe_list.insert(1, "actual_prices", test_array, True)

predictions_dataframe_list['Prices'] = predictions_dataframe_list['Prices'] + 0
predictions_dataframe_list

predictions_dataframe_list.columns = ['predicted_price','actual_price']

prediction = rf.predict(numpy_dataframe_train)

tt = train['Date'].tolist()
idx2 = pd.to_datetime(pd.Series(tt))

predictions_dataframe1 = pd.DataFrame(data=prediction[0:], index = idx2, columns=['Predicted Prices'])

test_array1 = train["Prices"].tolist()
predictions_dataframe1.insert(1, "actual_prices", test_array1, True)


predictions_dataframe1.columns=['Predicted Prices','Actual Prices']
predictions_dataframe1.plot(color=['orange','green'], title = 'Random Forest Regressor of Predicted vs Actual Stock Prices')




xx = predictions_dataframe1['Actual Prices'].apply(np.int64)
yy = predictions_dataframe1['Predicted Prices'].apply(np.int64)

print(accuracy_score(xx,yy))






