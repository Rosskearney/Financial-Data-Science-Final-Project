#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 21:22:18 2022

@author: rosskearney
"""

import numpy as np
import pandas as pd
from nltk.sentiment.util import *
import matplotlib.pyplot as mlpt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import linear_model
import csv
import yfinance as yf
import requests
import os
import json
import dateutil.parser
import time
import re
from datetime import date, timedelta


# =============================================================================
# # delete previously made 'data.csv' file
# =============================================================================
# os.remove('twitterData.csv')
# os.remove('twitterDataTSLA.csv')

# =============================================================================
#           ASSIGN TOKEN
# =============================================================================
os.environ['TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAGifbAEAAAAAUrJ5zdt4YLK9FnUwWyCGSJ2jXbg%3DPhujW2Fjr4UbmbJ2iFORV8L8CFYYwifPO0xfOFOI0qYhsm99ZT'

# =============================================================================
#           FUNCTION TO REMOVE URL FROM TWEETS
# =============================================================================
def remove_url(txt):
    """Remove URLs found in text string with nothing 
    Parameters
    ----------
    txt : string
        A text string that you want to parse and remove urls.
    Returns
    -------
    The same txt string with url's removed.
    """
    return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())


# =============================================================================
#  SOME FUNCTION CREATION AND QUERY PARAMETERS
# =============================================================================
# function which retrieves the token from the environment.
def auth():
    return os.getenv('TOKEN')

# function to authorise bearer token and return headers used to access the API.
def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

# Function with searching parameters
def create_url(keyword, start_date, end_date, max_results = 10):
    
    search_url = "https://api.twitter.com/2/tweets/search/all"

    #change params based on the endpoint
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
   
    return (search_url, query_params)


def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()



# =============================================================================
#                           INPUT SEARCH PARAMETERS
# =============================================================================
bearer_token = auth()
headers = create_headers(bearer_token)
keyword = "TSLA -is:retweet -is:nullcast lang:en"
# results per day wanted
max_results = 15



start_list_sdate = date(2021,2,1)   # start date
start_list_edate = date(2022,2,1)

end_list_sdate = date(2021,2,28)   # start date
end_list_edate = date(2022,2,28)

# =============================================================================
#                       CREATE LIST OF SEARCH DATES
# =============================================================================
start_list_dates = pd.date_range(start_list_sdate,start_list_edate-timedelta(days=1),freq='d').strftime('%Y-%m-%d').to_list()
end_list_dates = pd.date_range(end_list_sdate,end_list_edate-timedelta(days=1),freq='d').strftime('%Y-%m-%d').to_list()

start_list = []
end_list = []
# Taking max 500 tweets between each start and end date, i.e. each month
for i in range(len(start_list_dates)):
    start_list.append(start_list_dates[i] + 'T00:00:00.000Z')
    
for i in range(len(end_list_dates)):
    end_list.append(end_list_dates[i] + 'T00:00:00.000Z')    
    

start_time = start_list[0]
end_time = end_list[-1]
    
url = create_url(keyword, start_time,end_time, max_results)
json_response = connect_to_endpoint(url[0], headers, url[1])

print(json.dumps(json_response, indent=4, sort_keys=True))


    
# =============================================================================
#          FUNCTION: APPEND RESULTS TO CSV
# =============================================================================
def append_to_csv(json_response, fileName):

    #A counter variable
    counter = 0

    #Open OR create the target CSV file
    csvFile = open(fileName, "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    #Loop through each tweet
    for tweet in json_response['data']:

        created_at = dateutil.parser.parse(tweet['created_at'][0:10])

        text = remove_url(tweet['text'])
        
        # Assemble all data in a list
        res = [created_at, text]
        
        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1

    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response: ", counter) 



# =============================================================================
#           PREPARE CSV FILE TO BE WRITTED OVER
# =============================================================================
#Total number of tweets we collected from the loop
total_tweets = 0

# Create file
csvFile = open("twitterDataTSLA.csv", "a", newline="", encoding='utf-8') #/////////////////////////////
csvWriter = csv.writer(csvFile)

#Create headers for the data you want to save
csvWriter.writerow(['created_at','tweet']) 
csvFile.close()


# =============================================================================
#               FOR LOOP TO ACTUALLY PULL TWEETS
# =============================================================================
for i in range(0,len(start_list)):

    # Inputs
    count = 0 # Counting tweets per time period
    max_count = max_results # Max tweets per time period
    flag = True
    next_token = None
    
    # Check if flag is true
    while flag:
        # Check if max_count reached
        if count >= max_count:
            break
        print("-------------------")
        print("Token: ", next_token)
        url = create_url(keyword, start_list[i],end_list[i], max_results)
        json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
        result_count = json_response['meta']['result_count']

        if 'next_token' in json_response['meta']:
            # Save the token to use for next call
            next_token = json_response['meta']['next_token']
            print("Next Token: ", next_token)
            if result_count is not None and result_count > 0 and next_token is not None:
                print("Start Date: ", start_list[i])
                append_to_csv(json_response, "twitterDataTSLA.csv") #////////////////////////////////
                count += result_count
                total_tweets += result_count
                print("Total # of Tweets added: ", total_tweets)
                print("-------------------")
                time.sleep(5)                
        # If no next token exists
        else:
            if result_count is not None and result_count > 0:
                print("-------------------")
                print("Start Date: ", start_list[i])
                append_to_csv(json_response, "twitterDataTSLA.csv") #////////////////////////////////
                count += result_count
                total_tweets += result_count
                print("Total # of Tweets added: ", total_tweets)
                print("-------------------")
                time.sleep(5)
            
            #Since this is the final request, turn flag to false to move to the next time period.
            flag = False
            next_token = None
        time.sleep(5)
print("Total number of results: ", total_tweets)


# =============================================================================
#                   CREATE DATAFRAME OF TWEETS AND TIDYING DATES
# =============================================================================
# creating a data frame
df = pd.DataFrame()
df = pd.read_csv("twitterDataTSLA.csv")#///////////////////////////////////////////////////

df = df.rename(columns={"created_at": "Date", "tweet": "Tweets"})

df['Date'] = df['Date'].str[:10]

print(df)



# =============================================================================
#                      CREATING 'Tweets.csv' FILE ?
# =============================================================================
os.remove('Tweets.csv')                                                            

df.to_csv("Tweets.csv")
cdata=pd.DataFrame(columns=['Date','Tweets'])
total=100
index=0
for index,row in df.iterrows():
    stre=row["Tweets"]
    my_new_string = re.sub('[^ a-zA-Z0-9]', '', stre)
    temp_df = pd.DataFrame([[df["Date"].iloc[index], 
                            my_new_string]], columns = ['Date','Tweets'])
    cdata = pd.concat([cdata, temp_df], axis = 0).reset_index(drop = True)

# =============================================================================
#                  ADDING SENTIMENT COLUMNS
# =============================================================================

cdata["Comp"] = ''
cdata["Negative"] = ''
cdata["Neutral"] = ''
cdata["Positive"] = ''
cdata


nltk.download('vader_lexicon')

sentiment_i_a = SentimentIntensityAnalyzer()
for indexx, row in cdata.T.iteritems():
    try:
        sentence_i = unicodedata.normalize('NFKD', cdata.loc[indexx, 'Tweets'])
        sentence_sentiment = sentiment_i_a.polarity_scores(sentence_i)
        cdata['Comp'].iloc[indexx] = sentence_sentiment['compound']
        cdata['Negative'].iloc[indexx] = sentence_sentiment['neg']
        cdata['Neutral'].iloc[indexx] = sentence_sentiment['neu']
        cdata['Positive'].iloc[indexx] = sentence_sentiment['compound']

    except TypeError:
        print (indexx)


# =============================================================================
#           CREATE NEW COLUMN, PUT AVG SENTIMENT PER DAY FOR THAT DAY
# =============================================================================

cdata['SentAvg'] = ''

for i in cdata['Date'].unique():
   cdata.loc[cdata["Date"] == i, "SentAvg"] = cdata[cdata['Date']==i]['Comp'].mean()


# =============================================================================
#                  CREATING DATAFRAME WITH ONE ROW PER DAY
# =============================================================================


ccdata=pd.DataFrame(columns=['Date','Tweets','Comp','Negative','Neutral','Positive','SentAvg'])
# i = 0
indx=0
get_title=""
for i in range(0,len(cdata)-1):
    get_date=cdata.Date.iloc[i]
    next_date=cdata.Date.iloc[i+1]
    if(str(get_date)==str(next_date)):
        get_title=get_title+cdata.Tweets.iloc[i]+" "
    if(str(get_date)!=str(next_date)):
        get_title = cdata.Tweets.iloc[i] # **********
        sentimentAvg = cdata.SentAvg.iloc[i]
        CompScore = cdata['Comp'].iloc[i] 
        NegScore = cdata['Negative'].iloc[i]
        NeuScore =  cdata['Neutral'].iloc[i] 
        PosScore = cdata['Positive'].iloc[i] 
        temp_df = pd.DataFrame([[get_date, get_title, CompScore, NegScore,
                                 NeuScore, PosScore,                  
                                 sentimentAvg]], columns =['Date','Tweets','Comp','Negative','Neutral','Positive','SentAvg'])
        ccdata = pd.concat([ccdata, temp_df], axis = 0).reset_index(drop = True)
        get_title=" "



# =============================================================================
#                       GETTING STOCK DATA
# =============================================================================

StartDate = cdata['Date'].iloc[0]
EndDate = cdata['Date'].iloc[-1]

ticker1 = 'TSLA'

read_stock_p = yf.download(ticker1, start=StartDate, end=EndDate) 

# RESETTING INDEX AWAY FROM DATES
read_stock_p = read_stock_p.reset_index()



# =============================================================================
# ADDING 'Prices' COLUMN TO TWEETS DATA AND PUTTING CORRESPONDING STOCK PRICE EACH DAY
# =============================================================================

ccdata['Prices']=""


indx=0
for i in range (0,len(ccdata)):
    for j in range (0,len(read_stock_p)):
        get_title_date=ccdata.Date.iloc[i]

        get_stock_date=str(read_stock_p.Date.iloc[j])[0:10]
        if(str(get_stock_date)==str(get_title_date)):
            ccdata['Prices'].iloc[i] = read_stock_p.Close[j]
            
# =============================================================================
#       PUTTING PREV. CLOSE INTO DATES WITH CLOSED MARKETS
# =============================================================================

ccdata = ccdata.mask(ccdata == '')
    
ccdata['Prices'].fillna(method='ffill', inplace=True)
ccdata['Prices'].fillna(method='bfill', inplace=True)

# =============================================================================
#           CALCULATING POS/NEG TWEET PERCENTAGES
# =============================================================================

posi=0
nega=0
for i in range (0,len(ccdata)):
    get_val=ccdata.SentAvg[i]
    if(float(get_val)<(0)):
        nega=nega+1
    if(float(get_val>(0))):
        posi=posi+1
    
posper=(posi/(len(ccdata)))*100
negper=(nega/(len(ccdata)))*100
print("% of positive days= ",posper)
print("% of negative days= ",negper)
arr=np.asarray([posper,negper], dtype=int)
mlpt.pie(arr,labels=['positive','negative'])
mlpt.plot()



# =============================================================================
#               MACHINE LEARING FOR STOCK PREDICTION BEGINS HERE
# =============================================================================

#  create new datafram
df_=ccdata[['Date','Prices','Comp','Negative','Neutral','Positive']].copy()
df_

df_ = df_.mask(ccdata == '')
    
df_['Prices'].fillna(method='bfill', inplace=True)
df_['Prices'].fillna(method='ffill', inplace=True)


# split dataframe into training and testing
percentage = round(len(df_)/100*70) 

len(df_)-percentage

test=df_.iloc[percentage:len(df_),:]
train = df_.iloc[:percentage]



# Create training dataframe
sentiment_score_list = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray([df_.loc[date, 'Negative'],df_.loc[date, 'Positive']])
    sentiment_score_list.append(sentiment_score)
numpy_df_train = np.asarray(sentiment_score_list)

print(numpy_df_train)



# Create testing dataframe
sentiment_score_list = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray([df_.loc[date, 'Negative'],df_.loc[date, 'Positive']])
    sentiment_score_list.append(sentiment_score)
numpy_df_test = np.asarray(sentiment_score_list)

print(numpy_df_test)

y_train = pd.DataFrame(train['Prices'])

y_test = pd.DataFrame(test['Prices'])


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

reg.predict(numpy_df_test)


print(reg.score(numpy_df_train, y_train))
print(reg.predict(numpy_df_train))

# =============================================================================
# 
# =============================================================================


percentage = round(len(df_)/100*70) 

len(df_)-percentage

test=df_.iloc[percentage:len(df_),:]
train = df_.iloc[:percentage]


train_data_start = train['Date'].iloc[0]
train_data_end = train['Date'].iloc[-1]
test_data_start = test['Date'].iloc[0]
test_data_end = test['Date'].iloc[-1]

trainStart = ((df_[df_['Date']==train_data_start].index.values))[0]
trainEnd = ((df_[df_['Date']==train_data_end].index.values))[0]
testStart = ((df_[df_['Date']==test_data_start].index.values))[0]
testEnd = ((df_[df_['Date']==test_data_end].index.values))[0]

train = df_.loc[trainStart : trainEnd]
test = df_.loc[testStart : testEnd]



#       train
list_of_sentiments_score = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray([df_.loc[date, 'Comp']])
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_train = np.asarray(list_of_sentiments_score)

#       test
list_of_sentiments_score = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray([df_.loc[date, 'Comp']])
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
predictions_df.plot()
predictions_df['predicted_price'] = predictions_df['predicted_price']
print(rf.score(numpy_dataframe_train, train['Prices']))


# =============================================================================
# Linear Regression Prediction
# =============================================================================

tt3 = test['Date'].tolist()
idx3 = pd.to_datetime(pd.Series(tt3))

regr = linear_model.LinearRegression()
regr.fit(numpy_dataframe_train, train['Prices'])   
prediction = regr.predict(numpy_dataframe_test)


predictions_df = pd.DataFrame(data=prediction[0:], index = idx3, columns=['Prices'])
predictions_df['Prices'] = predictions_df['Prices']
newData = test['Prices'].to_numpy()
testNew = pd.DataFrame(data=newData, index = idx3, columns=['Prices'])
predictions_df['actual_value'] = testNew['Prices']
predictions_df.columns = ['predicted_price', 'actual_price']
predictions_df.plot()
predictions_df['predicted_price'] = predictions_df['predicted_price']
print(regr.score(numpy_dataframe_train, train['Prices']))

# =============================================================================
#               PREPARING START/END DATES FOR PREDICTIONS
# =============================================================================
percentage = round(len(df_)/100*70) 

len(df_)-percentage

test=df_.iloc[percentage:len(df_),:]
train = df_.iloc[:percentage]


train_data_start = train['Date'].iloc[0]
train_data_end = train['Date'].iloc[-1]
test_data_start = test['Date'].iloc[0]
test_data_end = test['Date'].iloc[-1]

trainStart = ((df_[df_['Date']==train_data_start].index.values))[0]
trainEnd = ((df_[df_['Date']==train_data_end].index.values))[0]
testStart = ((df_[df_['Date']==test_data_start].index.values))[0]
testEnd = ((df_[df_['Date']==test_data_end].index.values))[0]

train = df_.loc[trainStart : trainEnd]
test = df_.loc[testStart : testEnd]


FTdataframe = df_



# =============================================================================
#       LISTS OF SENTIMENT SCORES
# =============================================================================

def Average(lst):
    return sum(lst) / len(lst)

list_of_sentiments_score = []
for date, row in train.T.iteritems():
    sentiment_score = np.asarray([FTdataframe.loc[date, 'Comp'],FTdataframe.loc[date, 'Negative'],FTdataframe.loc[date, 'Neutral'],FTdataframe.loc[date, 'Positive']])
    for n in range(len(sentiment_score)):
        if sentiment_score[n] == 0:
            sentiment_score[n] = Average(sentiment_score)
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_train = np.asarray(list_of_sentiments_score)




list_of_sentiments_score = []
for date, row in test.T.iteritems():
    sentiment_score = np.asarray([FTdataframe.loc[date, 'Comp'],FTdataframe.loc[date, 'Negative'],FTdataframe.loc[date, 'Neutral'],FTdataframe.loc[date, 'Positive']])
    for n in range(len(sentiment_score)):
        if sentiment_score[n] == 0:
            sentiment_score[n] = Average(sentiment_score)
    list_of_sentiments_score.append(sentiment_score)
numpy_dataframe_test = np.asarray(list_of_sentiments_score)


# =============================================================================
#    adjusting random forest regressor setting
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
predictions_dataframe1.plot(color=['orange','green'],title = 'Random Forest Regressor Predicted vs Actual Stock Prices')


xx = predictions_dataframe1['Actual Prices'].apply(np.int64)
yy = predictions_dataframe1['Predicted Prices'].apply(np.int64)

# Accuracy score

print(accuracy_score(xx,yy))




# =============================================================================
#               CUMULATIVE SENTIMENT X STOCK PRICE
# =============================================================================

def Cumulative(lists):
    cu_list = []
    length = len(lists)
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)]
    return cu_list[1:]

CumSent =  (Cumulative(ccdata['SentAvg']))
Sent = ccdata['SentAvg']
SentMA = ccdata['SentAvg'].rolling(30).mean()
StockPrice = ccdata['Prices']

StockPrice = StockPrice.mask(StockPrice == '')
StockPrice.fillna(method='bfill', inplace=True)
StockPrice.fillna(method='ffill', inplace=True)

len(CumSent)
len(StockPrice)

sentXstock = pd.DataFrame(
    {'Cumulative sentiment': CumSent,
     'Sentiment': Sent,
     'Rolling':SentMA,
     'Stock Price': StockPrice})



# create figure and axis objects with subplots()
fig,ax = mlpt.subplots()

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()

# make both plot
ax.plot(sentXstock['Rolling'], color = 'blue')
ax2.plot(sentXstock['Stock Price'], color = 'red')

# set x-axis label
ax.set_xlabel("Days",fontsize=14)

# make plots with different y-axis using second axis object
ticker1 = 'Stock Price'
ticker2 = 'Article Sentiment'

ax2.set_ylabel(ticker1, color="blue",fontsize=14)
ax.set_ylabel(ticker2, color="red",fontsize=14)

mlpt.title('TSLA stock price vs 30 day rolling sentiment average')
mlpt.xticks(rotation = 70)
mlpt.show()









