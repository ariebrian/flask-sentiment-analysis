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

def translate(tweet):
	translator = Translator()
	return translator.translate(tweet, dest ='en').text


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
		for word in token:
	        # print(word)
			positive_score += dictionaryP.check(word)

		if negative_score > positive_score:
			res = 'negative'
			negative_tweets.append(tweet[x])
		elif negative_score == positive_score:
			res =  'neutral'
			neutral_tweets.append(tweet[x])
		else:
			res =  'positive'
			positive_tweets.append(tweet[x])
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
	kmean = pd.read_pickle("kmeans.pickle")

	Y =  tfpickle.transform(data)
	labels = kmean.predict(Y)

	print(labels)

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
	f = request.files.getlist('inputfiles[]')
	# print (f[0])
	for x in range(len(f)):
		# print (f[x])
		data = pd.read_csv(f[x])
		tweets = data['Tweet'].values	
		# res[f[x].filename] = lexicon(tweets)
		# print("limittttt")
		temp[x] = lexicon(tweets)
		name.append(f[x].filename)
		res2[x] = clustering(tweets)
	# print (temp)
	# print(res2)

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
	