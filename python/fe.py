import pandas as pd
import numpy as np
import nltk
import re
import csv
from sklearn.feature_extraction import text
from bs4 import BeautifulSoup


PUNC = ['(',')',':',';',',','-','!','.','?','/','"',"'",'*']
SYMBOL = ['<','>','@','#']
STOP_WORDS = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(STOP_WORDS)
sw = []
for stw in STOP_WORDS:
	sw.append("q"+stw)
	sw.append("z"+stw)
STOP_WORDS = text.ENGLISH_STOP_WORDS.union(sw)
CARRIAGE_RETURNS = ['\n','\r\n']
WORD_REGEX = "^[a-z']+$"

def clean_word(word):
	"""
	clean the word
	->lower->del punction/returns->match as word
	"""
	word = word.lower()
	for punc in PUNC+CARRIAGE_RETURNS:
		word = word.replace(punc,"").strip("'")
	return word if re.match(WORD_REGEX,word) else None

def text_preproc(line):
	text = []
	line = BeautifulSoup(line).get_text(" ")
	porter = nltk.PorterStemmer()
	words = nltk.word_tokenize(line)
	for w in words:
		cleanword = clean_word(w)
		if cleanword and cleanword not in STOP_WORDS and len(cleanword) > 1:
			text.append(porter.stem(cleanword))
	return text

def data_preproc(data,outfile):
	fo = csv.writer(open(outfile,'w'),lineterminator='\n')
	if 'median_relevance' in data:
		fo.writerow(['id','query','product_title','product_description','median_relevance','relevance_variance'])
		for i in xrange(len(data.id)):
			query = ' '.join(text_preproc(data['query'][i]))
			product_title = ' '.join(text_preproc(data['product_title'][i]))
			product_description = ' '.join(text_preproc(data['product_description'][i]))
			fo.writerow([data['id'][i],query,product_title,product_description,data['median_relevance'][i],data['relevance_variance'][i]])
	else:
		fo.writerow(['id','query','product_title','product_description'])
		for i in xrange(len(data.id)):
			query = ' '.join(text_preproc(data['query'][i]))
			product_title = ' '.join(text_preproc(data['product_title'][i]))
			product_description = ' '.join(text_preproc(data['product_description'][i]))
			fo.writerow([data['id'][i],query,product_title,product_description])

def count_word_feature(data):
	query_count = []
	p_title_count = []
	p_des_count = []
	for i in xrange(len(data)):
		query_count.append(len(data['query'][i].split(' ')))
		p_title_count.append(len(data['product_title'][i].split(' ')))
		p_des_count.append(len(data['product_description'][i].split(' ')))
	c_feature = pd.DataFrame({'query':data['query'].values,'query_count':query_count,'product_title_count':p_title_count,'product_description_count':p_des_count})
	return c_feature

def mr_of_train_query(train_data):
	query_set = set(list(train_data['query']))
	query_dict = {}
	for q in query_set:
		query_dict[q] = []
	for i in xrange(len(train_data)):
		query_dict[train_data['query'][i]].append(train_data['median_relevance'][i])
	fo = csv.writer(open('../query_label_stat.csv','w'),lineterminator='\n')
	fo.writerow(['query','mean','max','min','std'])
	for q in query_dict:
		ay = np.array(query_dict[q])
		fo.writerow([q,ay.mean(),ay.max(),ay.min(),ay.std()])
