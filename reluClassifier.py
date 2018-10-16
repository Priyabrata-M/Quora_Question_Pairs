import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

df_train = pd.read_csv('X_TRAIN_Similarity.csv')
M = df_train.as_matrix()

Cosine_dist_train = df_train['Cosine'].values.tolist()
Jacard_sim_train = df_train['jackard'].values.tolist()
wordnet_sim_train = df_train['WordnetSimilarity'].values.tolist()
Cosine_Doc2Vec  = df_train['Cosine_Doc2Vec'].values.tolist()
word1stSimilarity = df_train['word1stSimilarity'].values.tolist()
wordlastSimilarity = df_train['wordlastSimilarity'].values.tolist()
bigramMatchingCounts = df_train['bigramMatchingCounts'].values.tolist()
levenshteinDistance  = df_train['levenshteinDistance'].values.tolist()
hammingDistance = df_train['hammingDistance'].values.tolist()
jaroWinkler = df_train['jaroWinkler'].values.tolist()
is_duplicate_train = df_train['Is_duplicate'].values.tolist()

X_train = [ ([0] * 10) for row in range(len(M)) ]
#print X_train[:,0]
Y_train = []

i=0
while i < len(Cosine_dist_train):
       X_train[i][0] = Cosine_dist_train[i]
       X_train[i][1] = Jacard_sim_train[i]
       X_train[i][2]=wordnet_sim_train[i]
       X_train[i][3]=Cosine_Doc2Vec[i]
       X_train[i][4]=word1stSimilarity[i]
       X_train[i][5]=wordlastSimilarity[i]
       X_train[i][6]=bigramMatchingCounts[i]
       X_train[i][7]=levenshteinDistance[i]
       X_train[i][8]=hammingDistance[i]
       X_train[i][9]=jaroWinkler[i]
       i=i+1
       
Y_train = is_duplicate_train[:]
     

X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.15,random_state=0)

#===========relu =neural net==================
count=500
nn = MLPClassifier(activation='relu',hidden_layer_sizes=(100,50,10),learning_rate= 'constant',learning_rate_init=0.0005, random_state=1, max_iter = count)
nn.fit(X_train,Y_train)
joblib.dump(nn, 'RELUClassifier.pkl')
pred = nn.predict(X_test)
#print "Accuracy:" + str(float(count)/len(pred))
print " Accuracy of neural net "+str(metrics.accuracy_score(Y_test, pred))

