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

df = pd.read_csv('train.csv')
question1_train = df['question1'].values.tolist()
question2_train = df['question2'].values.tolist()


df1 = pd.read_csv('test.csv')
question1_test = df1['question1'].values.tolist()
question2_test = df1['question2'].values.tolist()



corpus = question1_train + question2_train + question1_test + question2_test

tok_crop = [nltk.word_tokenize(str(sent).decode('utf-8')) for sent in corpus]
model = gensim.models.Word2Vec(tok_crop,size=100, window=10, min_count=1, workers=4)

## trim memory
#model.init_sims(replace=True)
### creta a dict 
##w2v = dict(zip(model.index2word, model.syn0))
##print "Number of tokens in Word2Vec:", len(w2v.keys())
#
joblib.dump(model, 'modelW2V.pkl') 

# save model
#model.save('data/3_word2vec.mdl')
#model.save_word2vec_format('data/3_word2vec.bin', binary=True)


pic_model = joblib.load("modelW2V.pkl")
print pic_model.wv.similarity('man','woman')

print model.wv['what']
#print pic_model.wv.similarity('What is the step by step guide to invest in share market in india?', 'What is the step by step guide to invest in share market?')
