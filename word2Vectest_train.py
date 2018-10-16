#Word2Vec on merged test and train
import gensim.models as gensim
import logging
import pandas as pd
import numpy as np
from sklearn.externals import joblib 
import nltk
from gensim import corpora, models, similarities



#fileptr = open("final_inputWord2Vec_test_train.txt",'a');
fileptr2 = open("cosinSimilarityWord2Vec_train.txt",'a');
#fileptr3 = open("cosinSimilarityWord2Vec_test.txt",'a');
##
df = pd.read_csv('train.csv')
#
#for i in range(0,len(df)):
#    fileptr.write(str(df["question1"][i])+" "+str(df["question2"][i])+" ")
#    
df1 = pd.read_csv('test.csv')
#
#for i in range(0,len(df1)):
#    fileptr.write(str(df1["question1"][i])+" "+str(df1["question2"][i])+" ")
#    
#
#
#print "file merged"
#
#fileptr1 = open("final_inputWord2Vec_test_train.txt",'r');
#sentences = fileptr1.read()

#q1_train = df['question1'].values.tolist()
#q1_test = df1['question1'].values.tolist()
#q2_train = df['question2'].values.tolist()
#q2_test = df1['question2'].values.tolist()
#
#corpus= q1_train + q1_test + q2_train + q2_test
#  
#tok_corp= [nltk.word_tokenize(sent.decode('utf-8')) for sent in corpus]
#       
#           
#model = gensim.Word2Vec(tok_corp, size=100, window=5, min_count=1, workers=4)
#
#model.save('Word2Vec_test_train')
#model = gensim.Word2Vec.load('Word2Vec_test_train')
#model.most_similar('india')
#model.most_similar([vector])

#model = g.Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
#joblib.dump(model, 'word2Vec_test_train.pkl') 
#
#print "model pickled"
#pickled_model = joblib.load('word2Vec_test_train.pkl')

#print model.wv['what']

pic_model = joblib.load("modelW2V.pkl")


for i in range(0,len(df)):
    Q1 = str(df["question1"][i])
    Q2 = str(df["question2"][i])
    print "Q1:"
    print Q1
    print "Q2:"
    print Q2
    lsQ1 = Q1.split();
    lsQ2 = Q2.split();    
    sumSimilarity = [];
    for j in range(0,len(lsQ1)):
        maxSimilarity = [];
        for k in range(0,len(lsQ2)):
            print str(lsQ1[j]) +" VS "+ str(lsQ2[k])
            if str(lsQ1[j]).endswith('?'):
                str_Q1 = str(lsQ1[j]).replace('?','')
            else:
                str_Q1 = str(lsQ1[j])
           
            if str(lsQ2[k]).endswith('?'):
                str_Q2 = str(lsQ2[k]).replace('?','')
            else:
                str_Q2 = str(lsQ2[k])
            print str_Q1 +" VS "+ str_Q2  
            maxSimilarity.append(pic_model.wv.similarity(str_Q1,str_Q2))
        sumSimilarity.append(max(maxSimilarity))
        print sumSimilarity
    fileptr2.write(sum(sumSimilarity))
    
#for i in range(0,len(df1)):
#    Q1 = str(df1["question1"][i])
#    Q2 = str(df1["question2"][i])
#    lsQ1 = Q1.split();
#    lsQ2 = Q2.split();    
#    sumSimilarity = [];
#    for j in range(0,len(lsQ1)):
#        maxSimilarity = [];
#        for k in range(0,len(lsQ2)):
#            maxSimilarity.append(model.wv.similarity(str(lsQ1[j]),str(lsQ2[k])))
#        sumSimilarity.append(max(maxSimilarity))
#    fileptr3.write(sum(sumSimilarity))
#            
#            
