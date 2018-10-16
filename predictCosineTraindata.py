import gensim.models as g
import logging
import pandas as pd
import numpy as np

d2v_model = g.Doc2Vec.load('doc2vecMOdel_Train.doc2vec')

fileptr1 = open("CosinSimmilarity_train.txt",'a');

df = pd.read_csv('train.csv')

    
for i in range(404209,len(df)):
    fileptr1.write(str(i)+","+str(d2v_model.docvecs.similarity_unseen_docs(d2v_model, str(df["question1"][i]),str(df["question2"][i]), alpha=0.01, min_alpha=0.001, steps=300)));
    fileptr1.write("\n")
