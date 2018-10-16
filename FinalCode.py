# -*- coding: utf-8 -*-
import HTMLParser
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
import string
import nltk
import pickle
from sklearn.externals import joblib
from nltk.corpus import wordnet as wn
import csv
import pandas as pd
import numpy as np
from sklearn import metrics
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
string_input="When do you use it's シ instead of し?I luv my &lt;3 iphone &amp; you’re awsm apple. DisplayIsAwesome, sooo happppppy �, �"

df = pd.read_csv('demo.csv')
M = df.as_matrix()
Y=M[:,5]

## get the csv columns:

id = df['id'].values.tolist()
qid1 = df['qid1'].values.tolist()
qid2 = df['qid2'].values.tolist()
question1 = df['question1'].values.tolist()
question2 = df['question2'].values.tolist()
is_duplicate = df['is_duplicate'].values.tolist()

##====================================================
corpus = question1 + question2
tok_crop = [nltk.word_tokenize(str(sent).decode('utf-8')) for sent in corpus]
model = gensim.models.Word2Vec(tok_crop, workers=16, iter=10, negative=20)

### save the model:


joblib.dump(model, 'modelD2V.pkl') 

## save model
pic_model = joblib.load("model_W2V.pkl")
#print pic_model.wv.similarity('man','woman')
#print pic_model.wv.similarity('What is the step by step guide to invest in share market in india?', 'What is the step by step guide to invest in share market?')
#print pic_model.most_similar(['man'])
print model.wmdistance(['man'])

## test train split

X_train, X_test, y_train, y_test = train_test_split(
M, Y, test_size=0.15, random_state=42)


## create final data set with out stop words
M_backup=M; 



#def sent2vec_(sent1,sent2):
#    pass
    

def preprocess(string_input): 
    string_input=string_input.lower()
#    with punctuation accuracy score is actually better 
#    string_input=string_input.translate(None, string.punctuation)        
    ### decode 1====================================>>>    
    ans= string_input.decode("utf8").encode('ascii','ignore')
    string_input =ans
#    print("unocode removed string :"+string)   
    
    
    ## html parser====================================>>>
    html_parser = HTMLParser.HTMLParser()
    string_input = html_parser.unescape(string_input)
#    print string    
    
    #Removal of Stop-words: ====================================>>>
    #'them','their','him','his','her','i','am'
    stopwordList=['the','are', 'is','a','an','and','are','in','is','it','its','of','on','with','do']
#    stopwordList=['i','the','are', 'is','be','being','to','a','an','and','are','as','at','be','by','for','in','is','it','its','of','on','that','to','was','were','will','with','have','had','can','could','may','might','bellow','below','under','do','somebodys','someother','someone',]
    ans=''
    word_list=string_input.split()
    
    
    ### lemmatization process================================>>>>
    from nltk.stem.wordnet import WordNetLemmatizer 
    lem = WordNetLemmatizer()
    for i in range (0,len(word_list)):
        word_list[i]=lem.lemmatize(word_list[i], "v")       
    
    ###stopword removal=========================================>>>>>>
#    stopwordList=stopwords.words('english')
#    stopwordList.remove('not')
    filtered_word_list = word_list[:] #make a copy of the word_list
    for word in word_list: # iterate over word_list
      if word in stopwordList: 
        filtered_word_list.remove(word) # remove word from filtered_word_list if it is a stopword
    ans= ' '.join(filtered_word_list) 
#    for i in filtered_word_list:
#        ans+=i.strip()+' '
    string_input=ans
    
#    print("stopwords removed string :"+str(filtered_word_list))
     #Standardizing words:====================================>>>
    import itertools 
    string_input = ''.join(''.join(s)[:2] for _, s in itertools.groupby(string_input))
#    print string
    
   
      
    
    return string_input



#================================WORDNET==================================
def wordnetSentenceSimilarity(sent1,sent2):
    score=0
    for st1 in sent1.split():
        for st2 in sent2.split():
           score+= get_best_synset_pair(st1,st2)
    return score        

def get_best_synset_pair(word_1, word_2):
    """ 
    Choose the pair with highest path similarity among all pairs. 
    Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               if sim > max_sim:
                   max_sim = sim
                   best_pair = synset_1, synset_2
        return max_sim


### sentence to vectore ==============================================>>>
    
import math
from collections import Counter
def get_cosine(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in common])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
   
    if not denominator:
        return 0.0 
    else:
        return float(numerator) / denominator

def text_to_vector(text): 
    words = text.split() 
    return Counter(words)


def levenshtein(s1,s2): 
    if len(s1) > len(s2):
        s1,s2 = s2,s1 
    distances = range(len(s1) + 1) 
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1]) 
            else:
                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1]))) 
        distances = newDistances 
    return distances[-1]

##===================================================================>>>>>



## preprocess the data:
wordnetSimilarity=[]    
cosineDistance=[]
levenshteinDistance=[]
labeledSentence=np.zeros((len(X_train),2),dtype=object)
colourArray=[]     
for i in range (0,len(X_train)):
    
    if y_train[i]==str(1):
        colourArray.append('red')
    else:
        colourArray.append('blue')      

    X_train[i][3]=preprocess(str(X_train[i][3]))
#    print (M[i][3])
    X_train[i][4]=preprocess(str(X_train[i][4]))
#    print (M[i][4])    
    ## compute the cosine distance between them    
    ## calculate word net similarity
    
    wordnetSimilarity.append( wordnetSentenceSimilarity(X_train[i][3],X_train[i][4]))
    print wordnetSimilarity 
   ## doc to vec
    ## labeled sentence creation==================================================>>>
    #have to format sentences into [['word1', 'word2', 'word3', 'lastword'], ['label1']]
    labeledSentence[i][0]=(X_train[i][3]+' '+X_train[i][4]).split()
    labeledSentence[i][1]=str(X_train[i][5]).split() 
    print labeledSentence[i]
#   print labeledSentence[i][1]
    vector1 = text_to_vector(X_train[i][3]) 
    vector2 = text_to_vector(X_train[i][4]) 
    cosineDistance.append(get_cosine(vector1, vector2))
    levenshteinDistance.append(levenshtein(X_train[i][3],X_train[i][4]))
#print cosineDistance  
#print levenshteinDistance  
 

##================================================+++>>>>>>    

## pickle file:
    
from sklearn.externals import joblib
joblib.dump(X_train, 'PreProcessed_Xtrain_pickle_file.pkl')    
joblib.dump(cosineDistance, 'PreProcessed_cosineDistance_pickle_file.pkl')    
joblib.dump(levenshteinDistance, 'PreProcessed_levenshteinDistance_pickle_file.pkl')    
with open('preprocessed_data.csv', 'w') as csvfile:
    fieldnames = ['question_1', 'question_2','Is_duplicate','cosineDistance','levenshteinDistance']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range (0,len(X_train)):        
        writer.writerow({'question_1': X_train[i][3], 'question_2': X_train[i][4],'Is_duplicate':X_train[i][5],'cosineDistance':cosineDistance[i],'levenshteinDistance':levenshteinDistance[i]})
csvfile.close()  

##logistic regression implementation  

#featureMetrix=X_train[:,3:5]

featureMetrix=np.column_stack((cosineDistance,levenshteinDistance))
#print featureMetrix

### Visualize the data set=========================================>>
## plot the points ========================================

plt.figure()
plt.scatter(featureMetrix[:,0],featureMetrix[:,1],s=25,color=colourArray[:])
plt.savefig('dataVisualization.png')
plt.show()



######TRAIN THE MODEL##########################################>>>>>

#LOGISTIV REGRESSION
clf=linear_model.LogisticRegression()  
clf.fit(featureMetrix,y_train)
#SVM#############################################################>>>>>
## predict
clf_svm= svm.SVC(kernel='linear')
clf_svm.fit(featureMetrix,y_train)

clf_rbfsvm= svm.SVC()
clf_rbfsvm.fit(featureMetrix,y_train)

##get the cosine and lavenstine distance from the test file
cosineDistance_test=[]
levenshteinDistance_test=[]
for i in range (0,len(X_test)):
    vector1 = text_to_vector(X_test[i][3]) 
    vector2 = text_to_vector(X_test[i][4]) 
    cosineDistance_test.append(get_cosine(vector1, vector2))
    levenshteinDistance_test.append(levenshtein(X_test[i][3],X_test[i][4]))
    
featureMetrix_test=np.column_stack((cosineDistance_test,levenshteinDistance_test))
## predicts models
Y_predict_logistic=clf.predict(featureMetrix_test)
Y_predict_svm=clf_svm.predict(featureMetrix_test)
Y_predict_rbfsvm=clf_rbfsvm.predict(featureMetrix_test)
#print Y_predict
#print y_test
logistic_acc=metrics.accuracy_score(y_test, Y_predict_logistic)
svm_acc=metrics.accuracy_score(y_test, Y_predict_svm)
rbfsvm_acc=metrics.accuracy_score(y_test, Y_predict_rbfsvm)
print 'logistic regression accuracy score:'+str(logistic_acc)
print 'Linear SVM accuracy score:'+str(svm_acc)
print 'RBF SVM accuracy score:'+str(rbfsvm_acc)
    