# coding = utf-8
""" 
Search Results Relevance @ Kaggle
author : littlekid
email : muyunlei@gmail.com
"""
import pandas as pd
import numpy as np
import util
import fe
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.stem.porter import *
from sklearn.feature_extraction import text
from bs4 import BeautifulSoup

# A model
train_data = '../train.csv'
test_data = '../test.csv'

train = pd.read_csv(train_data).fillna("")
test = pd.read_csv(test_data).fillna("")

idx_test = test.id.values.astype(int)
idx_train = train.id.values.astype(int)
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

y_train = train.median_relevance.values
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

# the infamous tfidf vectorizer
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')

# Fit TFIDF
tfv.fit(traindata)
X_train =  tfv.transform(traindata) 
X_test = tfv.transform(testdata)

svd = TruncatedSVD(n_components=300)
scl = StandardScaler()

clf = pipeline.Pipeline([('svd', svd),
						('scl',scl)])

X_train = clf.fit_transform(X_train)
X_test = clf.transform(X_test)

a_dict_train = {'id':idx_train}
a_dict_test = {'id':idx_test}
for i in xrange(X_train.shape[1]):
	a_dict_train['a_svd'+str(i)] = X_train[:,i]

for i in xrange(X_test.shape[1]):
	a_dict_test['a_svd'+str(i)] = X_test[:,i]

out_a_pd_train = pd.DataFrame(a_dict_train)
out_a_pd_test = pd.DataFrame(a_dict_test)

# B model
sw=[]
s_data = []
s_labels = []
t_data = []
t_labels = []
#stopwords tweak - more overhead
stop_words = ['http','www','img','border','0','1','2','3','4','5','6','7','8','9']
stop_words = text.ENGLISH_STOP_WORDS.union(stop_words)
for stw in stop_words:
	sw.append("q"+stw)
	sw.append("z"+stw)

stop_words = text.ENGLISH_STOP_WORDS.union(sw)

train = pd.read_csv(train_data).fillna("")
test = pd.read_csv(test_data).fillna("")

idx_test = test.id.values.astype(int)
idx_train = train.id.values.astype(int)

stemmer = PorterStemmer()
class stemmerUtility(object):
		"""Stemming functionality"""
		@staticmethod
		def stemPorter(review_text):
			porter = PorterStemmer()
			preprocessed_docs = []
			for doc in review_text:
				final_doc = []
				for word in doc:
					final_doc.append(porter.stem(word))
					#final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
				preprocessed_docs.append(final_doc)
			return preprocessed_docs

for i in range(len(train.id)):
	s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")]) + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
	s=re.sub("[^a-zA-Z0-9]"," ", s)
	s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
	s_data.append(s)
	s_labels.append(str(train["median_relevance"][i]))

for i in range(len(test.id)):
	s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
	s=re.sub("[^a-zA-Z0-9]"," ", s)
	s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
	t_data.append(s)

# clf = pipeline.Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
#     ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0))])

clf = pipeline.Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0))])

B_X_train = clf.fit_transform(s_data, s_labels)
B_X_test = clf.transform(t_data)

b_dict_train = {'id':idx_train}
b_dict_test = {'id':idx_test}
for i in xrange(B_X_train.shape[1]):
	b_dict_train['b_svd'+str(i)] = B_X_train[:,i]

for i in xrange(B_X_test.shape[1]):
	b_dict_test['b_svd'+str(i)] = B_X_test[:,i]

out_b_pd_train = pd.DataFrame(b_dict_train)
out_b_pd_test = pd.DataFrame(b_dict_test)

# C model
train_data = '../train_after_preproc.csv'
test_data = '../test_after_preproc.csv'

w2v_model = fe.w2v_model

train = pd.read_csv(train_data).fillna("")
test = pd.read_csv(test_data).fillna("")

idx_train = train.id.values.astype(int)
train = train.drop('id', axis=1)
idx_test = test.id.values.astype(int)
test = test.drop('id', axis=1)

y_train = train.median_relevance.values
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

traindata = np.array([fe.doc2vec_no_extra(w2v_model,doc.split()) for doc in traindata])
testdata = np.array([fe.doc2vec_no_extra(w2v_model,doc.split()) for doc in testdata])

c_dict_train = {'id':idx_train}
c_dict_test = {'id':idx_test}
for i in xrange(traindata.shape[1]):
	c_dict_train['cD2V'+str(i)] = traindata[:,i]

for i in xrange(testdata.shape[1]):
	c_dict_test['cD2V'+str(i)] = testdata[:,i]

out_c_pd_train = pd.DataFrame(c_dict_train)
out_c_pd_test = pd.DataFrame(c_dict_test)

# load data with pandas

train_data = '../train_after_preproc.csv'
test_data = '../test_after_preproc.csv'

train = pd.read_csv(train_data).fillna("")
test = pd.read_csv(test_data).fillna("")

cftrain = fe.count_word_feature(train)
cftest = fe.count_word_feature(test)

query_label_stat = pd.read_csv('../query_label_stat.csv')

# A+
features_train = pd.merge(cftrain,out_a_pd_train)
features_test = pd.merge(cftest,out_a_pd_test)

features_train = pd.merge(features_train,query_label_stat)
features_test = pd.merge(features_test,query_label_stat)

features_train = features_train.drop(['d2dsim'],axis=1)
features_test = features_test.drop(['d2dsim'],axis=1)

y_train = features_train.median_relevance.values
X_train = features_train.drop(['query','id','median_relevance'],axis=1).values
X_test = features_test.drop(['query','id'],axis=1).values
idx_train = features_train['id'].values.astype(int)
idx_test = features_test['id'].values.astype(int)

# svm
scl = StandardScaler()
svm_model = SVC(C=12)
clf = pipeline.Pipeline([('scl', scl),
						('svm', svm_model)])

clf.fit(X_train,y_train)
a_ypred_svm = clf.predict(X_test)


# B+
features_train = pd.merge(cftrain,out_b_pd_train)
features_test = pd.merge(cftest,out_b_pd_test)

features_train = pd.merge(features_train,query_label_stat)
features_test = pd.merge(features_test,query_label_stat)

y_train = features_train.median_relevance.values
X_train = features_train.drop(['query','id','median_relevance'],axis=1).values
X_test = features_test.drop(['query','id'],axis=1).values
idx_train = features_train['id'].values.astype(int)
idx_test = features_test['id'].values.astype(int)

# svm
scl = StandardScaler()
svm_model = SVC(C=12)
clf = pipeline.Pipeline([('scl', scl),
						('svm', svm_model)])

clf.fit(X_train,y_train)
b_ypred_svm = clf.predict(X_test)

# C+
features_train = pd.merge(cftrain,out_c_pd_train)
features_test = pd.merge(cftest,out_c_pd_test)

features_train = pd.merge(features_train,query_label_stat)
features_test = pd.merge(features_test,query_label_stat)

y_train = features_train.median_relevance.values
X_train = features_train.drop(['query','id','median_relevance'],axis=1).values
X_test = features_test.drop(['query','id'],axis=1).values
idx_train = features_train['id'].values.astype(int)
idx_test = features_test['id'].values.astype(int)

# svm
scl = StandardScaler()
svm_model = SVC(C=15)
clf = pipeline.Pipeline([('scl', scl),
						('svm', svm_model)])

clf.fit(X_train,y_train)
c_ypred_svm = clf.predict(X_test)

submission = pd.DataFrame({"id": idx_test, "prediction": np.floor((a_ypred_svm+b_ypred_svm+c_ypred_svm)/3).astype(int)})
submission.to_csv("../submission/Amodel_Bmodel_Cmodel_lk29_svm12_12_15.csv", index=False)
