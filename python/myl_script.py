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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.stem.porter import *
from sklearn.feature_extraction import text
from bs4 import BeautifulSoup
from gensim.models import word2vec
import xgboost as xgb

# offline validation
trainf = '../offline/offline_train.csv'
testf = '../offline/offline_test.csv'
trainfap = '../offline/offline_train_ap.csv'
testfap = '../offline/offline_test_ap.csv'
query_feature = '../query_label_stat.csv'

# online test
# trainf = '../train.csv'
# testf = '../test.csv'
# trainfap = '../train_after_preproc.csv'
# testfap = '../test_after_preproc.csv'
# query_feature = '../query_label_stst.csv'


# A model tdidf SVD+SVM
# load data with pandas
train = pd.read_csv(trainf).fillna("")
test = pd.read_csv(testf).fillna("")

# train_rv0 = train[train.relevance_variance==0]
# train_rv1 = train[train.relevance_variance!=0]

# abandon id

# idx_train_rv0 = train_rv0.id.values.astype(int)
# train_rv0 = train_rv0.drop('id', axis=1)
# idx_train_rv1 = train_rv1.id.values.astype(int)
# train_rv1 = train_rv1.drop('id', axis=1)

idx_train = train.id.values.astype(int)
train = train.drop('id', axis=1)
idx_test = test.id.values.astype(int)
test = test.drop('id', axis=1)

# create labels. drop useless columns
# y_train_rv0 = train_rv0.median_relevance.values
# train_rv0 = train_rv0.drop(['median_relevance', 'relevance_variance'], axis=1)
# y_train_rv1 = train_rv1.median_relevance.values
# train_rv1 = train_rv1.drop(['median_relevance', 'relevance_variance'], axis=1)

y_train = train.median_relevance.values
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

y_test = test.median_relevance.values

# do some lambda magic on text columns
# traindata_rv0 = list(train_rv0.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
# traindata_rv1 = list(train_rv1.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

# traindata = list(train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1))
# testdata = list(test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'],x['product_description']),axis=1))


# the infamous tfidf vectorizer
# tfv_rv0 = TfidfVectorizer(min_df=3,  max_features=None, 
#         strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
#         ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
#         stop_words = 'english')

# tfv_rv1 = TfidfVectorizer(min_df=3,  max_features=None, 
#         strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
#         ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
#         stop_words = 'english')


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
# Fit TFIDF
# tfv_rv0.fit(traindata_rv0)
# X_train_rv0 =  tfv_rv0.transform(traindata_rv0) 
# X_test_rv0 = tfv_rv0.transform(testdata)

# tfv_rv1.fit(traindata_rv1)
# X_train_rv1 =  tfv_rv1.transform(traindata_rv1) 
# X_test_rv1 = tfv_rv1.transform(testdata)

tfv.fit(traindata)
X_train =  tfv.transform(traindata) 
X_test = tfv.transform(testdata)

# Initialize SVD
svd = TruncatedSVD(n_components=300)

# Initialize the standard scaler 
scl = StandardScaler()

# model
svm_model = SVC(C=10,probability=True)

# clf_rv0 = pipeline.Pipeline([('svd', svd),
#     					('scl', scl),
#                     	('svm', svm_model)])

# clf_rv1 = pipeline.Pipeline([('svd', svd),
#     					('scl', scl),
#                     	('svm', svm_model)])

# clf_rv0.fit(X_train_rv0,y_train_rv0)
# clf_rv1.fit(X_train_rv1,y_train_rv1)
# ypred_rv0 = clf_rv0.predict(X_test_rv0)
# ypred_rv1 = clf_rv1.predict(X_test_rv1)

# print "RV0: %f"%(util.quadratic_weighted_kappa(y_test,ypred_rv0))
# print "RV1: %f"%(util.quadratic_weighted_kappa(y_test,ypred_rv1))
# print "RV0+RV1: %f"%(util.quadratic_weighted_kappa(y_test,np.floor((ypred_rv0+ypred_rv1)/2).astype(int)))

clf = pipeline.Pipeline([('svd', svd),
    					('scl', scl),
                    	('svm', svm_model)])
clf.fit(X_train,y_train)
ypred = clf.predict(X_test)
print "Kappa: %f"%(util.quadratic_weighted_kappa(y_test,ypred))

ppred = clf.predict_proba(X_train)
y_predp = clf.predict_proba(X_test)
out_a_pd_train = pd.DataFrame({'id':idx_train,'a_1p':ppred[:,0],'a_2p':ppred[:,1],'a_3p':ppred[:,2],'a_4p':ppred[:,3]})
out_a_pd_test = pd.DataFrame({'id':idx_test,'a_1p':y_predp[:,0],'a_2p':y_predp[:,1],'a_3p':y_predp[:,2],'a_4p':y_predp[:,3]})

clf = pipeline.Pipeline([('svd', svd)])

clf = pipeline.Pipeline([('svd', svd),
    					('scl', scl)])
X_train = clf.fit_transform(X_train,y_train)
X_test = clf.transform(X_test)

a_dict_train = {'id':idx_train}
a_dict_test = {'id':idx_test}
for i in xrange(X_train.shape[1]):
	a_dict_train['a_svd'+str(i)] = X_train[:,i]

for i in xrange(X_test.shape[1]):
	a_dict_test['a_svd'+str(i)] = X_test[:,i]

out_a_pd_train = pd.DataFrame(a_dict_train)
out_a_pd_test = pd.DataFrame(a_dict_test)

#########################A model end##########################################################################
#########################B model begin########################################################
# array declarations
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

train = pd.read_csv(trainf).fillna("")
test = pd.read_csv(testf).fillna("")

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
	t_labels.append(str(test["median_relevance"][i]))

clf = pipeline.Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
    ('svm', SVC(C=11, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])

clf = pipeline.Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)),
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True))])

clf = pipeline.Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0))])

clf.fit(s_data, s_labels)
ypred = clf.predict(t_data)
print util.quadratic_weighted_kappa(np.array(t_labels).astype(int),ypred.astype(float).astype(int))

ppred_train = clf.predict_proba(s_data)
ppred_test = clf.predict_proba(t_data)

ppred_train_dict = {'id':idx_train}
for i in xrange(ppred_train.shape[1]):

out_b_pd_train = pd.DataFrame({'id':idx_train,'b_1p':ppred_train[:,0],'b_2p':ppred_train[:,1],'b_3p':ppred_train[:,2],'b_4p':ppred_train[:,3]})
out_b_pd_test = pd.DataFrame({'id':idx_test,'b_1p':ppred_test[:,0],'b_2p':ppred_test[:,1],'b_3p':ppred_test[:,2],'b_4p':ppred_test[:,3]})

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



#########################B model end#############################################################################
#########################Word2Vec 50features#######################################################
# w2v_dict = util.loadWord2vec('../vocab.txt','../wordVectors.txt')
w2v_model = fe.w2v_model

train = pd.read_csv(trainfap).fillna("")
test = pd.read_csv(testfap).fillna("")

idx_train = train.id.values.astype(int)
train = train.drop('id', axis=1)
idx_test = test.id.values.astype(int)
test = test.drop('id', axis=1)

y_train = train.median_relevance.values
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

y_test = test.median_relevance.values

traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

traindata = np.array([fe.doc2vec_no_extra(w2v_model,doc.split()) for doc in traindata])
testdata = np.array([fe.doc2vec_no_extra(w2v_model,doc.split()) for doc in testdata])

# SVM
scl = StandardScaler()
svm_model = SVC(C=16)
clf = pipeline.Pipeline([('scl', scl),
						('svm', svm_model)])
clf.fit(traindata,y_train)

ypred_svm = clf.predict(testdata)
print util.quadratic_weighted_kappa(y_test,ypred_svm)



c_dict_train = {'id':idx_train}
c_dict_test = {'id':idx_test}
for i in xrange(traindata.shape[1]):
	c_dict_train['cD2V'+str(i)] = traindata[:,i]

for i in xrange(testdata.shape[1]):
	c_dict_test['cD2V'+str(i)] = testdata[:,i]

out_c_pd_train = pd.DataFrame(c_dict_train)
out_c_pd_test = pd.DataFrame(c_dict_test)


#########################Ensemble A and B########################################################################################
train = pd.read_csv(trainfap).fillna("")
test = pd.read_csv(testfap).fillna("")

cftrain = fe.count_word_feature(train)
cftest = fe.count_word_feature(test)

query_label_stat = pd.read_csv('../query_label_stat.csv')

features_train = pd.merge(cftrain,out_a_pd_train)
features_test = pd.merge(cftest,out_a_pd_test)

features_train = pd.merge(cftrain,out_b_pd_train)
features_test = pd.merge(cftest,out_b_pd_test)

features_train = pd.merge(features_train,out_b_pd_train)
features_test = pd.merge(features_test,out_b_pd_train)

features_train = pd.merge(cftrain,query_label_stat)
features_test = pd.merge(cftest,query_label_stat)

features_train = pd.merge(features_train,query_label_stat)
features_test = pd.merge(features_test,query_label_stat)

y_train = features_train.median_relevance.values
X_train = features_train.drop(['query','id','median_relevance'],axis=1).values
y_test = features_test.median_relevance.values
X_test = features_test.drop(['query','id','median_relevance'],axis=1).values
idx_train = features_train['id'].values.astype(int)
idx_test = features_test['id'].values.astype(int)

rf = RandomForestClassifier(n_jobs=2,n_estimators=70,min_samples_split=4,max_depth=8)
# gbc = GradientBoostingClassifier()


rf.fit(X_train,y_train)

# validation
ypred_rf = rf.predict(X_test)
print util.quadratic_weighted_kappa(y_test,ypred_rf)


# SVM
scl = StandardScaler()
svm_model = SVC(C=12)
clf = pipeline.Pipeline([('scl', scl),
						('svm', svm_model)])
clf.fit(X_train,y_train)

ypred_svm = clf.predict(X_test)
print util.quadratic_weighted_kappa(y_test,ypred_svm)

print util.quadratic_weighted_kappa(y_test,np.floor((ar+br)/2).astype(int))

# xgboost
param = {}
param['eta'] = 0.06
param['max_depth'] = 5
param['silent'] = 1
# param['min_child_weight'] = 100
param['objective'] = 'multi:softmax'
param['num_class'] = 4
param['nthread'] = 2

dtrain = xgb.DMatrix(X_train,label=y_train-1)
dtest = xgb.DMatrix(X_test)
bst = xgb.train(param,dtrain,80)

ypred_gbdt = bst.predict(dtest)+1
print util.quadratic_weighted_kappa(y_test,ypred_gbdt)

print "RF+GBDT: %f"%(util.quadratic_weighted_kappa(y_test,np.floor((ypred_rf+ypred_gbdt)/2).astype(int)))
print "SVM+GBDT: %f"%(util.quadratic_weighted_kappa(y_test,np.floor((ypred_svm+ypred_gbdt)/2).astype(int)))




##################Split With RV#################################################################
train = pd.read_csv(trainfap).fillna("")
test = pd.read_csv(testfap).fillna("")
query_label_stat = pd.read_csv('../query_label_stat.csv')

cftrain_rv0 = fe.count_word_feature(train,1)
cftrain_rv1 = fe.count_word_feature(train,2)
cftest = fe.count_word_feature(test)

features_train_rv0 = pd.merge(cftrain_rv0,query_label_stat)
features_train_rv1 = pd.merge(cftrain_rv1,query_label_stat)

features_test = pd.merge(cftest,query_label_stat)

y_train_rv0 = features_train_rv0.median_relevance.values
X_train_rv0 = features_train_rv0.drop(['query','id','median_relevance'],axis=1).values
y_train_rv1 = features_train_rv1.median_relevance.values
X_train_rv1 = features_train_rv1.drop(['query','id','median_relevance'],axis=1).values

y_test = features_test.median_relevance.values
X_test = features_test.drop(['query','id','median_relevance'],axis=1).values
idx_train_rv0 = features_train_rv0['id'].values.astype(int)
idx_train_rv1 = features_train_rv1['id'].values.astype(int)
idx_test = features_test['id'].values.astype(int)

# rv0 5,6,40
param_rv0 = {}
param_rv0['eta'] = 0.05
param_rv0['max_depth'] = 6
param_rv0['silent'] = 1
# param_rv0['min_child_weight'] = 100
param_rv0['objective'] = 'multi:softmax'
param_rv0['num_class'] = 4
param_rv0['nthread'] = 2

# rv1 4,6,90
param_rv1 = {}
param_rv1['eta'] = 0.04
param_rv1['max_depth'] = 6
param_rv1['silent'] = 1
# param_rv1['min_child_weight'] = 100
param_rv1['objective'] = 'multi:softmax'
param_rv1['num_class'] = 4
param_rv1['nthread'] = 2

dtrain_rv0 = xgb.DMatrix(X_train_rv0,label=y_train_rv0-1)
dtrain_rv1 = xgb.DMatrix(X_train_rv1,label=y_train_rv1-1)
dtest = xgb.DMatrix(X_test)
bst_rv0 = xgb.train(param_rv0,dtrain_rv0,40)
bst_rv1 = xgb.train(param_rv1,dtrain_rv1,90)

ypred_rv0 = bst_rv0.predict(dtest)+1
ypred_rv1 = bst_rv1.predict(dtest)+1

print "RV0: %f"%(util.quadratic_weighted_kappa(y_test,ypred_rv0))
print "RV1: %f"%(util.quadratic_weighted_kappa(y_test,ypred_rv1))
print "RV0+RV1: %f"%(util.quadratic_weighted_kappa(y_test,np.floor((ypred_rv0+ypred_rv1)/2).astype(int)))
