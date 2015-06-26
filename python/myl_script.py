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

# start with svd benchmark
# load data with pandas
train = pd.read_csv('../train_after_preproc.csv').fillna("")
test = pd.read_csv('../test_after_preproc.csv').fillna("")

cftrain = fe.count_word_feature(train)
cftest = fe.count_word_feature(test)

query_label_stat = pd.read_csv('../query_label_stat.csv')

features_train = pd.merge(cftrain,query_label_stat)
features_test = pd.merge(cfest,query_label_stat)

y = train.median_relevance.values
X_train = features_train.drop('query',axis=1).values
X_test = features_test.drop('query',axis=1).values
