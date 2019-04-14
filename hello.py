from flask import Flask, render_template, abort, request
import sys
from dict import Dictionary
import pandas as pd
from nltk.tokenize import word_tokenize
from googletrans import Translator
import progressbar
from time import sleep
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
from collections import Counter
import re, string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics import mean_squared_error
import tweepy

app = Flask(__name__)
class SentimentScore:
    def __init__(self, positive_tweets, negative_tweets, neutral_tweets):

        self.positive_tweets = positive_tweets
        self.negative_tweets = negative_tweets
        self.neutral_tweets = neutral_tweets

        self.neg = len(negative_tweets)
        self.pos = len(positive_tweets)
        self.neut = len(neutral_tweets)

dictionaryN  = Dictionary('negative.txt')
dictionaryP  = Dictionary('positive.txt')
dictionaryN2  = Dictionary('negative-words.txt')
dictionaryP2  = Dictionary('positive-words.txt')

def fetch(key):
	data={}
	auth = tweepy.auth.OAuthHandler("8IpOjg0S14jliFJ8f1gRvc9Y5" , "pg52bFarcIrtDJkfwdDdZkqAOi3hORw2l7v3amsK1YWocGIrUS")
	auth.set_access_token("609857856-2ze0BElTOm3AJbSIFrbwJFXi87ffkE0QmcOmyC1h", "7ny4oyk5Kx49P0mQTA8wK2o2wTtxhbfqnn6fNdsFFJCxm")

	api = tweepy.API(auth, wait_on_rate_limit = True)
	count = 0
	for tweet in tweepy.Cursor(api.search,
                           q = key,
                           since = "2019-04-05",
                           until = "2019-04-06",
                           tweet_mode = 'extended',
                           count=200,
                           lang = "id").items():
	if (not tweet.retweeted) and ('RT @' not in tweet.full_text) and (count<=100):
	    # Write a row to the CSV file. I use encode UTF-8
	    # csvWriter.writerow([tweet.created_at, tweet.full_text.encode('utf-8')])
	    data.extended(tweet.full_text)
	    count = count+1
	return data

def clean_tweet(tweet): 
        ''' 
        Utility function to clean tweet text by removing links, special characters 
        using simple regex statements. 

        '''
       
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|(&[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", tweet).split())

def preprocess(tweet):
	stopword_factory = StopWordRemoverFactory()
	stopword = stopword_factory.create_stop_word_remover()
	return stopword.remove(tweet)

def lexicon(tweet):
	negative_score = 0
	positive_score = 0
	negative_tweets = []
	positive_tweets = []
	neutral_tweets = []
	# bar = progressbar.ProgressBar(maxval=tweet.shape[0], \
 #    	widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	# bar.start()
	for x in range(tweet.shape[0]):
		# print(x)
		# print(tweet[x])
		token = word_tokenize(tweet[x])
		# print(token)
		for word in token:
			negative_score += dictionaryN.check(word)
			# print(word)
			# print(negative_score)
		for word in token:
			# print(word)
			positive_score += dictionaryP.check(word)
			# print(positive_score)	
		if negative_score > positive_score:
			res = 'negative'
			negative_tweets.append(tweet[x])
		elif negative_score == positive_score:
			res =  'neutral'
			neutral_tweets.append(tweet[x])
		else:
			res =  'positive'
			positive_tweets.append(tweet[x])
		# print(negative_score)
		# print(positive_score)
		negative_score = 0
		positive_score = 0
		# bar.update(x+1)
		# sleep(0.1)
		# print(res)
	# bar.finish()
	result_dict = {
		"positive" : len(positive_tweets),
		"negative" : len(negative_tweets),
		"neutral" : len(neutral_tweets)
	}
	# print(result_dict)
	return result_dict

def clustering(data):
	clu1=[]
	clu2=[]
	clu3=[]

	tfpickle = pd.read_pickle("tf.pickle")
	kmean = pd.read_pickle("kmeansmodel.pickle")

	Y =  tfpickle.transform(data)
	labels = kmean.predict(Y)

	print(Counter(kmean.labels_))

	if len(labels) == len(kmean.labels_):
		
		y_true = kmean.labels_
		y_pred = labels
		mse = mean_squared_error(y_true, y_pred)
		print(mse)

		resc={}
		for x in range (0,len(data)):
			resc[x]=labels[x]

		# print(type(modelKmeans.labels_))
		# print(resc)
		for x in range(0,len(resc)):
			if(resc[x] == 0):
				clu1.append(data[x])
			elif(resc[x] == 1):
				clu2.append(data[x])
			else:
				clu3.append(data[x])
		final={
			'clu1' : len(clu1),
			'clu2' : len(clu2),
			'clu3' : len(clu3),
			# 'isiclu1' : clu1,
			# 'isiclu2' : clu2,
			# 'isiclu3' : clu3
		}
		return final
		# return Counter(modelKmeans.labels_)
	else:
		resc={}
		for x in range (0,len(data)):
			resc[x]=labels[x]

		# print(type(modelKmeans.labels_))
		# print(resc)
		for x in range(0,len(resc)):
			if(resc[x] == 0):
				clu1.append(data[x])
			elif(resc[x] == 1):
				clu2.append(data[x])
			else:
				clu3.append(data[x])
		final={
			'clu1' : len(clu1),
			'clu2' : len(clu2),
			'clu3' : len(clu3),
			# 'isiclu1' : clu1,
			# 'isiclu2' : clu2,
			# 'isiclu3' : clu3
		}
		return final

@app.route("/", methods=["GET"])
def root():
	return render_template("index.html")


@app.route('/hello')
def hello_world():
    return 'Hello, World!'

@app.route('/result', methods=["POST"])
def analysis():
	temp={}
	res2={}
	name=[]
	res=[]
	prep = []
	f = request.files.getlist('inputfiles1')
	f.extend(request.files.getlist('inputfiles2'))
	# print (f)
	# print (len(f))
	for x in range(len(f)):
		# print (f[x])
		data = pd.read_csv(f[x])
		# print(data)
		tweets = data['Tweet'].values
		# print(tweets)
		for tweet in tweets:
			clean = clean_tweet(tweet).replace("b 	", '').lower()
			clear = preprocess(clean)
			prep.append(clear)
		# res[f[x].filename] = lexicon(tweets)
		# print("limittttt")
		preprocessed = np.array(prep, dtype=object)
		# print(preprocessed)
		temp[x] = lexicon(preprocessed)
		name.append(f[x].filename)
		# print(name)
		res2[x] = clustering(preprocessed)
		# print(res2)
		prep.clear()
		# print(preprocessed.shape)
	# print (type(preprocessed))


	lex1 = []
	lex1.append(temp[0]['positive'])
	lex1.append(temp[0]['negative'])
	lex1.append(temp[0]['neutral'])

	lex2 = []
	lex2.append(temp[1]['positive'])
	lex2.append(temp[1]['negative'])
	lex2.append(temp[1]['neutral'])

	clu1 = []
	clu1.append(res2[0]['clu1'])
	clu1.append(res2[0]['clu2'])
	clu1.append(res2[0]['clu3'])

	clu2 = []
	clu2.append(res2[1]['clu1'])
	clu2.append(res2[1]['clu2'])
	clu2.append(res2[1]['clu3'])

	# return ""
	return render_template('result.html', name=name, result=temp, res2=res2, result1=lex1, result2=lex2, clu1=clu1, clu2=clu2)

@app.route('/searchresult', methods=["POST"])
def search():
	temp={}
	res2={}
	name=[]
	res=[]
	prep=[]
	key = request.form['key1']
	key.extend(request.form['key2'])
	for x in range(len(key)):
		datas = fetch(key[x])
		for data in datas:
				clean = clean_tweet(data).replace("b 	", '').lower()
				clear = preprocess(data)
				prep.append(clear)
		preprocessed = np.array(prep, dtype=object)
		temp[x] = lexicon(preprocessed)
		res2[x] = clustering(preprocessed)
		prep.clear

	lex1 = []
	lex1.append(temp[0]['positive'])
	lex1.append(temp[0]['negative'])
	lex1.append(temp[0]['neutral'])

	lex2 = []
	lex2.append(temp[1]['positive'])
	lex2.append(temp[1]['negative'])
	lex2.append(temp[1]['neutral'])

	clu1 = []
	clu1.append(res2[0]['clu1'])
	clu1.append(res2[0]['clu2'])
	clu1.append(res2[0]['clu3'])

	clu2 = []
	clu2.append(res2[1]['clu1'])
	clu2.append(res2[1]['clu2'])
	clu2.append(res2[1]['clu3'])

	# return ""
	return render_template('result.html', name=name, result=temp, res2=res2, result1=lex1, result2=lex2, clu1=clu1, clu2=clu2)

	