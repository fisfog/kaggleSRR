import pandas as pd
import numpy as np
import nltk
import re
import csv
import util
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from bs4 import BeautifulSoup
from gensim import corpora,models,similarities
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
w2v_model = word2vec.Word2Vec.load('../300feature_40minwords_5context_all.model')
vocab = set(w2v_model.vocab.keys())

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


def d2dsim(d1,d2):
	l1 = len(d1)
	l2 = len(d2)
	s = dict()
	for w1 in d1:
		if w1 in vocab:
			if w1 not in s:
				s[w1] = []
			for w2 in d2:
				if w2 in vocab:
					s[w1].append(w2v_model.similarity(w1,w2))
	similar = 0
	for w in s:
		ar = np.array(s[w])
		if ar.size != 0:
			similar += ar.max()
	return (similar*1.0)/len(s) if len(s) != 0 else 0


def count_word_feature(data,flag=0):
	index = []
	query_count = []
	p_title_count = []
	p_des_count = []
	qw_inptitle = []
	qw_inptitle_p = []
	ptw_inquery = []
	ptw_inquery_p = []
	ptw_inquery_qp = []
	d2dsimilar = []
	for i in xrange(len(data)):
		if 'relevance_variance' in data:
			if flag == 1 and data['relevance_variance'][i] != 0:
				continue
			if flag == 2 and data['relevance_variance'][i] == 0:
				continue

		index.append(i)
		ql = data['query'][i].split(' ')
		ptl = data['product_title'][i].split(' ')
		pdl = data['product_description'][i].split(' ')
		query_count.append(len(ql))
		p_title_count.append(len(ptl))
		p_des_count.append(len(pdl))
		x = 0
		for qw in ql:
			if qw in ptl:
				x += 1
		qw_inptitle.append(x)
		qw_inptitle_p.append((x*1.0)/len(ql))
		x = 0
		for ptw in ptl:
			if ptw in ql:
				x += 1
		ptw_inquery.append(x)
		ptw_inquery_p.append((x*1.0/len(ptl)))
		ptw_inquery_qp.append((x*1.0)/len(ql))
		d2dsimilar.append(d2dsim(ql,ptl))

	if 'median_relevance' in data:
		fe_dict = {'query':data['query'].values[index],'query_count':query_count,'product_title_count':p_title_count,'product_description_count':p_des_count,'id':data['id'].values[index],'median_relevance':data['median_relevance'].values[index]}
		fe_dict['qw_inptitle'] = qw_inptitle
		fe_dict['qw_inptitle_p'] = qw_inptitle_p
		fe_dict['ptw_inquery'] = ptw_inquery
		fe_dict['ptw_inquery_p'] = ptw_inquery_p
		fe_dict['ptw_inquery_qp'] = ptw_inquery_qp
		fe_dict['d2dsim'] = d2dsimilar
		c_feature = pd.DataFrame(fe_dict)
	else:
		fe_dict = {'query':data['query'].values,'query_count':query_count,'product_title_count':p_title_count,'product_description_count':p_des_count,'id':data['id'].values}
		fe_dict['qw_inptitle'] = qw_inptitle
		fe_dict['qw_inptitle_p'] = qw_inptitle_p
		fe_dict['ptw_inquery'] = ptw_inquery
		fe_dict['ptw_inquery_p'] = ptw_inquery_p
		fe_dict['ptw_inquery_qp'] = ptw_inquery_qp
		fe_dict['d2dsim'] = d2dsimilar
		c_feature = pd.DataFrame(fe_dict)
	return c_feature

def mr_of_train_query(train_data):
	query_set = set(list(train_data['query']))
	query_dict = {}
	query_rel_v = {}
	for q in query_set:
		query_dict[q] = []
		query_rel_v[q] = []
	for i in xrange(len(train_data)):
		query_dict[train_data['query'][i]].append(train_data['median_relevance'][i])
		query_rel_v[train_data['query'][i]].append(train_data['relevance_variance'][i])
	fo = csv.writer(open('../query_label_stat.csv','w'),lineterminator='\n')
	fo.writerow(['query','mean','max','min','std','1p','2p','3p','4p',\
				'wilson1','wilson2','wilson3','wilson4',\
				'wp1','wp2','wp3','wp4','rv_mean','rv_max','rv_min','rv_std'])
	for q in query_dict:
		ay = np.array(query_dict[q])
		rv_ay = np.array(query_rel_v[q])
		fe = []
		fe.append(q)
		fe.append(ay.mean())
		fe.append(ay.max())
		fe.append(ay.min())
		fe.append(ay.std())
		for i in xrange(1,5):
			fe.append((np.sum(ay==i)*1.0)/ay.size)
		# wilson confidence lower limit
		for i in xrange(1,5):
			fe.append(util.wilson_lower_limit((np.sum(ay==i)*1.0)/ay.size,ay.size))
		# wl x p
		for i in xrange(1,5):
			fe.append(fe[4+i]*fe[8+i])
		fe.append(rv_ay.mean())
		fe.append(rv_ay.max())
		fe.append(rv_ay.min())
		fe.append(rv_ay.std())
		fo.writerow(fe)

def svd_tfidf(data,outfile,k=400):
	dl = list(data.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
	tfv = TfidfVectorizer(min_df=3,  max_features=None, 
			strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
			ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
			stop_words = 'english')
	tfv.fit(dl)
	X =  tfv.transform(dl)
	svd = TruncatedSVD(n_components=k)
	svd.fit(X)
	X = svd.transform(X)
	fo = csv.writer(open(outfile,'w'),lineterminator='\n')
	head = ['id']
	for i in xrange(k):
		head.append('f'+str(i+1))
	fo.writerow(head)
	for i in xrange(len(data)):
		fe = [data['id'][i]]+X[i].tolist()
		fo.writerow(fe)


def doc2vec_by_avg_word(w2v_dict,doc,n_features=50):
	d2v = np.zeros(n_features)
	k = 0
	for w in doc:
		if w in w2v_dict:
			d2v += w2v_dict[w]
			k += 1
	return np.divide(d2v,k)

def doc2vec_no_extra(w2v_model,doc,n_features=300):
	d2v = np.zeros(n_features)
	k = 0
	for w in doc:
		if w in w2v_model:
			d2v += w2v_model[w]
			k += 1
	return np.divide(d2v,k) if k != 0 else d2v