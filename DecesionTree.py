import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
from sklearn.externals import joblib
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib 
from sklearn import svm 

df_train = pd.read_csv('X_TRAIN_Similarity.csv')
M = df_train.as_matrix()


Cosine_dist_train = df_train['Cosine'].values.tolist()
Jacard_sim_train = df_train['jackard'].values.tolist()
wordnet_sim_train = df_train['WordnetSimilarity'].values.tolist()
is_duplicate_train = df_train['Is_duplicate'].values.tolist()

X_train = [ ([0] * 2) for row in range(len(M)) ]
#print X_train[:,0]
Y_train = []

i=0
while i < len(Cosine_dist_train):
       X_train[i][0] = Cosine_dist_train[i]
       X_train[i][1] = Jacard_sim_train[i]
#       X_train[i][2] = wordnet_sim_train[i]
       i=i+1
       
Y_train = is_duplicate_train[:]
print X_train
print Y_train       

df_test = pd.read_csv('TestMixData.csv')
M = df_test.as_matrix()


Cosine_dist_test = df_test['cosine'].values.tolist()
Jacard_sim_test = df_test['jackard'].values.tolist()
#wordnet_sim_test = df_test['WordnetSimilarity'].values.tolist()


X_test = [ ([0] * 2) for row in range(len(M)) ]
Y_test = []
i=0
while i < len(Cosine_dist_test):
       X_test[i][0] = Cosine_dist_test[i]
       X_test[i][1] = Jacard_sim_test[i]
#       X_test[i][2] = wordnet_sim_test[i]
       i=i+1




'''Decesion Tree'''
print "Decesion Tree"
clf = DecisionTreeClassifier(max_depth=10,min_samples_split=10,min_samples_leaf=5)
clf = clf.fit(X_train, Y_train)
joblib.dump(clf, 'DecisionTreeClassifier.pkl') 

clf = joblib.load('DecisionTreeClassifier.pkl') 
Y_test = clf.predict(X_test)
print Y_test
fileptr = open('DecisionTreeClassifier.csv','a') 
fileptr.write("test_id" + "," + "is_duplicate" +"\n")
for i in range(len(Y_test)):   
    fileptr.write(str(i) + "," + str(Y_test[i]) +"\n")
fileptr.close 




