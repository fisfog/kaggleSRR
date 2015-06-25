# coding = utf-8
""" 
Search Results Relevance @ Kaggle
author : littlekid
email : muyunlei@gmail.com
"""
import pandas as pd
import numpy as np
import util
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.stem.porter import *

# start with svd benchmark
# load data with pandas
train = pd.read_csv('../train.csv').fillna("")
test = pd.read_csv('../test.csv').fillna("")

# abandon id
idx = test.id.values.astype(int)
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# create labels. drop useless columns
y = train.median_relevance.values
train = train.drop(['median_relevance', 'relevance_variance'], axis=1)

# do some lambda magic on text columns
traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))

# the infamous tfidf vectorizer
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')

# Fit TFIDF
tfv.fit(traindata)
X =  tfv.transform(traindata) 
X_test = tfv.transform(testdata)

# Initialize SVD
svd = TruncatedSVD()

# Initialize the standard scaler 
scl = StandardScaler()

# model
svm_model = SVC()


# Create the pipeline 
clf_svm = pipeline.Pipeline([('svd', svd),
						 ('scl', scl),
                	     ('svm', svm_model)])

# Create a parameter grid to search for best parameters for everything in the pipeline
param_grid = {'svd__n_components' : [200,300,400,500],
              'svm__C': [1,5,10]}

# Kappa Scorer 
kappa_scorer = metrics.make_scorer(util.quadratic_weighted_kappa, greater_is_better = True)

# Initialize Grid Search Model of SVM
model_svm = grid_search.GridSearchCV(estimator = clf_svm, param_grid=param_grid, scoring=kappa_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

# Fit Grid Search Model
model_svm.fit(X, y)
print("Best score: %0.3f" % model_svm.best_score_)
print("Best parameters set:")
best_parameters = model_svm.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
	print("\t%s: %r" % (param_name, best_parameters[param_name]))
	
best_model_svm = model_svm.best_estimator_

# LR
lr_model = LogisticRegression()
clf_lr = pipeline.Pipeline([('svd', svd),
						 ('scl', scl),
                	     ('lr', lr_model)])

param_grid = {'svd__n_components' : [300,400,500],
              'lr__C': [0.1,1,10]}
# Initialize Grid Search Model of LR
model_lr = grid_search.GridSearchCV(estimator = clf_lr, param_grid=param_grid, scoring=kappa_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

# Fit Grid Search Model
model_lr.fit(X, y)
print("Best score: %0.3f" % model_lr.best_score_)
print("Best parameters set:")
best_parameters = model_lr.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
	print("\t%s: %r" % (param_name, best_parameters[param_name]))
best_model_lr = model_lr.best_estimator_

# RF
rf_model = RandomForestClassifier(n_jobs = -1)
clf_rf = pipeline.Pipeline([('svd', svd),
                	     ('rf', rf_model)])

param_grid = {'svd__n_components' : [300,400,500],
              'rf__n_estimators': [100,150,200]}
# Initialize Grid Search Model of LR
model_rf = grid_search.GridSearchCV(estimator = clf_rf, param_grid=param_grid, scoring=kappa_scorer,
                                 verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

# Fit Grid Search Model
model_rf.fit(X, y)
print("Best score: %0.3f" % model_rf.best_score_)
print("Best parameters set:")
best_parameters = model_rf.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
	print("\t%s: %r" % (param_name, best_parameters[param_name]))

best_model_rf = model_rf.best_estimator_
