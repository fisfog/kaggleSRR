
"""
Beating the Benchmark 
Search Results Relevance @ Kaggle
__author__ : Abhishek

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search
from nltk.stem.porter import *
import pandas as pd
import numpy as np
import re
import fe
import util
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import xgboost as xgb

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


# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


if __name__ == '__main__':
    # train = pd.read_csv('./train_after_preproc.csv').fillna("")
    # test = pd.read_csv('./test_after_preproc.csv').fillna("")
    # cftrain = fe.count_word_feature(train)
    # cftest = fe.count_word_feature(test)


    # Load the training file
    train = pd.read_csv('../train.csv')
    test = pd.read_csv('../test.csv')
    
    # we dont need ID columns
    idx = test.id.values.astype(int)
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)
    
    # create labels. drop useless columns
    y = train.median_relevance.values
    train = train.drop(['median_relevance', 'relevance_variance'], axis=1)
    
    # do some lambda magic on text columns
    traindata = list(train.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    testdata = list(test.apply(lambda x:'%s %s' % (x['query'],x['product_title']),axis=1))
    
    # the infamous tfidf vectorizer (Do you remember this one?)
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
    
    # We will use SVM here..
    svm_model = SVC()
    
    # Create the pipeline 
    clf = pipeline.Pipeline([('svd', svd),
    						 ('scl', scl),
                    	     ('svm', svm_model)])
    
    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'svd__n_components' : [300,350,380,400],
                  'svm__C': [1,5,10,15]}
    
    # Kappa Scorer 
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better = True)
    
    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
                                     verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
                                     
    # Fit Grid Search Model
    model.fit(X, y)
    print("Best score: %0.3f" % model.best_score_)
    a_best_s = model.best_score_
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
    	print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    # Get best model
    best_model = model.best_estimator_
    
    # Fit model with best parameters optimized for quadratic_weighted_kappa
    best_model.fit(X,y)
    preds = best_model.predict(X_test)
    A_pd = pd.DataFrame({'id':idx,'ap':preds})
    
    #load data
    train = pd.read_csv("../train.csv").fillna("")
    test  = pd.read_csv("../test.csv").fillna("")
    
    #remove html, remove non text or numeric, make query and title unique features for counts using prefix (accounted for in stopwords tweak)
    stemmer = PorterStemmer()
    ## Stemming functionality
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
        s=(" ").join(["q"+ z for z in BeautifulSoup(train["query"][i]).get_text(" ").split(" ")])\
                     + " " + (" ").join(["z"+ z for z in BeautifulSoup(train.product_title[i]).get_text(" ").split(" ")])\
                      + " " + BeautifulSoup(train.product_description[i]).get_text(" ")
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s_data.append(s)
        s_labels.append(str(train["median_relevance"][i]))
    for i in range(len(test.id)):
        s=(" ").join(["q"+ z for z in BeautifulSoup(test["query"][i]).get_text().split(" ")]) + " " + (" ").join(["z"+ z for z in BeautifulSoup(test.product_title[i]).get_text().split(" ")]) + " " + BeautifulSoup(test.product_description[i]).get_text()
        s=re.sub("[^a-zA-Z0-9]"," ", s)
        s= (" ").join([stemmer.stem(z) for z in s.split(" ")])
        t_data.append(s)
    #create sklearn pipeline, fit all, and predit test data
    clf = Pipeline([('v',TfidfVectorizer(min_df=5, max_df=500, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 2), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = 'english')), 
    ('svd', TruncatedSVD(n_components=200, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)), 
    ('scl', StandardScaler(copy=True, with_mean=True, with_std=True)), 
    ('svm', SVC(C=11, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None))])
    clf.fit(s_data, s_labels)
    t_labels = clf.predict(t_data)
    B_pd = pd.DataFrame({'id':idx,'bp':t_labels})


    # myl

    train = pd.read_csv('../train_after_preproc.csv').fillna("")
    test = pd.read_csv('../test_after_preproc.csv').fillna("")

    # cftrain_rv0 = fe.count_word_feature(train,1)
    # cftrain_rv1 = fe.count_word_feature(train,2)
    cftrain = fe.count_word_feature(train)
    cftest = fe.count_word_feature(test)

    query_label_stat = pd.read_csv('../query_label_stat.csv')

    # features_train_rv0 = pd.merge(cftrain_rv0,query_label_stat)
    # features_train_rv1 = pd.merge(cftrain_rv1,query_label_stat)
    features_train = pd.merge(cftrain,query_label_stat)
    features_test = pd.merge(cftest,query_label_stat)

    # y_rv0 = features_train_rv0.median_relevance.values
    # X_train_rv0 = features_train_rv0.drop(['query','id','median_relevance'],axis=1).values
    # y_rv1 = features_train_rv1.median_relevance.values
    # X_train_rv1 = features_train_rv1.drop(['query','id','median_relevance'],axis=1).values

    y = features_train.median_relevance.values
    X_train = features_train.drop(['query','id','median_relevance'],axis=1).values

    X_test = features_test.drop(['query','id'],axis=1).values
    c_idx = features_test['id'].values.astype(int)

    # RandomForestClassifier
    # rf = RandomForestClassifier(n_jobs=2,n_estimators=70,min_samples_split=4,max_depth=8)
    # clf = pipeline.Pipeline([('rf',rf)])
    # param_grid = {'rf__n_estimators': [50,60,70,80,90,100],
    #             'rf__min_samples_split':[3,4,5],
    #             'rf__max_depth':[3,4,5,6,7]}

    # kappa_scorer = metrics.make_scorer(util.quadratic_weighted_kappa, greater_is_better = True)

    # model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
    #                              verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

    # model.fit(X_train, y)
    # print("Best score: %0.3f" % model.best_score_)

    # print("Best parameters set:")
    # best_parameters = model.best_estimator_.get_params()
    # for param_name in sorted(param_grid.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))

    # best_model = model.best_estimator_

    # myl_ypreds_rf = best_model.predict(X_test)

    # scl = StandardScaler()
    # svm_model = SVC()
    # param_grid = {'svm__C':[5,6,7,8,9,10,11,12,13]}
    # clf = pipeline.Pipeline([('scl', scl),
    #                     ('svm', svm_model)])
    # model = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring=kappa_scorer,
    #                              verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)
    # model.fit(X_train, y)
    # print("Best score: %0.3f" % model.best_score_)

    # print("Best parameters set:")
    # best_parameters = model.best_estimator_.get_params()
    # for param_name in sorted(param_grid.keys()):
    #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
    # best_model = model.best_estimator_

    # myl_ypreds_svm = best_model.predict(X_test)

   
    #####################XGBoost############################
    # rv0 5,6,40
    # param_rv0 = {}
    # param_rv0['eta'] = 0.05
    # param_rv0['max_depth'] = 6
    # param_rv0['silent'] = 1
    # # param_rv0['min_child_weight'] = 100
    # param_rv0['objective'] = 'multi:softmax'
    # param_rv0['num_class'] = 4
    # param_rv0['nthread'] = 2    

    # # rv1 4,6,90
    # param_rv1 = {}
    # param_rv1['eta'] = 0.04
    # param_rv1['max_depth'] = 6
    # param_rv1['silent'] = 1
    # # param_rv1['min_child_weight'] = 100
    # param_rv1['objective'] = 'multi:softmax'
    # param_rv1['num_class'] = 4
    # param_rv1['nthread'] = 2


    param = {}
    param['eta'] = 0.04
    param['max_depth'] = 6
    param['silent'] = 1
    # param_rv0['min_child_weight'] = 100
    param['objective'] = 'multi:softmax'
    param['num_class'] = 4
    param['nthread'] = 2  

    # n_rv0 = 40
    # n_rv1 = 90
    num_round = 75

    # dtrain_rv0 = xgb.DMatrix(X_train_rv0,label=y_rv0-1)
    # dtrain_rv1 = xgb.DMatrix(X_train_rv1,label=y_rv1-1)
    dtrain = xgb.DMatrix(X_train,label=y-1)
    dtest = xgb.DMatrix(X_test)
    # bst_rv0 = xgb.train(param_rv0,dtrain_rv0,n_rv0)
    # bst_rv1 = xgb.train(param_rv1,dtrain_rv1,n_rv1)
    bst = xgb.train(param,dtrain,num_round)
    # myl_ypreds_rv0 = bst_rv0.predict(dtest)+1
    # myl_ypreds_rv1 = bst_rv1.predict(dtest)+1
    # myl_ypreds = np.floor((myl_ypreds_rv0+myl_ypreds_rv1)/2).astype(int)
    
    myl_ypreds_gbdt = bst.predict(dtest)+1

    # myl_ypreds = np.floor((myl_ypreds_gbdt+myl_ypreds_rf)/2).astype(int)
    C_pd = pd.DataFrame({'id':c_idx,'cp':myl_ypreds_gbdt})

    # D_pd = pd.read_csv('../submission/cf-submission.csv')
    # D_pd.columns = 'id','dp'

    final_preds = pd.merge(A_pd,B_pd)
    final_preds = pd.merge(final_preds,C_pd)
    # final_preds = pd.merge(final_preds,D_pd)

    # print "Cpred%d"%len(C_pd)
    # print len(final_preds)
    # print "len(idx)%d"%len(idx)
    import math
    p3 = []
    for i in range(len(final_preds)):
        # x = (int(t_labels[i]) + preds[i] + myl_ypreds[i])/3
        x = (final_preds['ap'][i]+int(final_preds['bp'][i])+final_preds['cp'][i])/3
        x = math.floor(x)
        p3.append(int(x))
        
        
    
    # p3 = (t_labels + preds)/2
    # p3 = p3.apply(lambda x:math.floor(x))
    # p3 = p3.apply(lambda x:int(x))
    
    # preds12 = 

    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": p3})
    submission.to_csv("../submission/tfidf_porter_mylXgb13.csv", index=False)