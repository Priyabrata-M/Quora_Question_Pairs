import gensim.models as g
import logging
import pandas as pd
import numpy as np

d2v_model = g.Doc2Vec.load('doc2vecMOdel_Test.doc2vec')

fileptr1 = open("CosinSimmilarity_test_1800000_1900000.txt",'a');

df = pd.read_csv('test.csv')

    
for i in range(1800000,1900000):
    fileptr1.write(str(i)+","+str(d2v_model.docvecs.similarity_unseen_docs(d2v_model, str(df["question1"][i]),str(df["question2"][i]), alpha=0.01, min_alpha=0.001, steps=300)));
    fileptr1.write("\n")
