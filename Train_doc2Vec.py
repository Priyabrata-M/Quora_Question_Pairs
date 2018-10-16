import gensim.models as g
import logging
import pandas as pd
import numpy as np


#doc2vec parameters
vector_size = 100
window_size = 10
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 
worker_count = 1 

'''train_corpus = "final_inputDoc2Vec_train.txt"

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#train doc2vec model
docs = g.doc2vec.TaggedLineDocument(train_corpus)
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, 
                  negative=negative_size, dbow_words=1, dm_concat=1, iter=train_epoch)

#save model
model.save('doc2vecMOdel_Train.doc2vec')'''

d2v_model = g.Doc2Vec.load('doc2vecMOdel_Train.doc2vec')

lsStr1 = "How do I read and find my YouTube comments?".split();
lsStr2 = "How can I see all my Youtube comments?".split();

print d2v_model.docvecs.similarity_unseen_docs(d2v_model, lsStr1, lsStr2, alpha=0.01, min_alpha=0.001, steps=300);
