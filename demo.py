import numpy as np
import numpy
import HTMLParser
from sklearn import linear_model
from sklearn import svm
import matplotlib.pyplot as plt
import string
import nltk
from sklearn.externals import joblib
from nltk.corpus import wordnet as wn
import csv
import string
from string import punctuation
from gensim.models import Word2Vec
import jellyfish
import gensim.models as g
import math
from sklearn.feature_extraction.text import  TfidfVectorizer


from sklearn import metrics
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
# from gensim.parsing import PorterStemmer

# global_stemmer = PorterStemmer()

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
    
    stopwordList=['the','are', 'is','a','an','is','do']
#    stopwordList=['i','the','are', 'is','be','being','to','a','an','and','are','as','at','be','by','for','in','is','it','its','of','on','that','to','was','were','will','with','have','had','can','could','may','might','bellow','below','under','do','somebodys','someother','someone',]
    ans=''
    word_list=string_input.split()
    
    
    ### lemmatization process================================>>>>
#    from nltk.stem.wordnet import WordNetLemmatizer 
#    lem = WordNetLemmatizer()
#    for i in range (0,len(word_list)):
#        word_list[i]=lem.lemmatize(word_list[i], "v")       
#    
    ###stopword removal=========================================>>>>>>

    filtered_word_list = word_list[:] #make a copy of the word_list
    for word in word_list: # iterate over word_list
      if word in stopwordList: 
        filtered_word_list.remove(word) # remove word from filtered_word_list if it is a stopword
    ans= ' '.join(filtered_word_list) 

    string_input=ans
    
#    print("stopwords removed string :"+str(filtered_word_list))
     #Standardizing words:====================================>>>
#    import itertools 
#    string_input = ''.join(''.join(s)[:2] for _, s in itertools.groupby(string_input))
##    print string
    
   
      
    
    return string_input


# wordnet
    
def get_best_synset_pair(word_1, word_2):
    """ 
    Choose the pair with highest path similarity among all pairs. 
    Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return 0
    else:
        max_sim = -1.0
#        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               if sim > max_sim:
                   max_sim = sim
#                   best_pair = synset_1, synset_2
                 
        return max_sim


#================================WORDNET==================================
def wordnetSentenceSimilarity(sent1,sent2):
    
    if len(sent1)<=len(sent2):
        sent1_A=np.array(sent1.split())        
        sent2_A=np.array(sent2.split() )
    else:
        sent1_A=np.array(sent2.split())
        sent2_A=np.array(sent1.split() )
    score=0
    for st1 in sent1_A:
        maxScore=0
        maxSimilarword=""
        for st2 in sent2_A:
            temp=get_best_synset_pair(st1,st2)
            if temp>maxScore:
                maxScore=temp
                maxSimilarword=st2
        score= score+maxScore
        index = np.argwhere(sent2_A==maxSimilarword)
        np.delete(sent2_A,index)
    return score/((len(sent1)+len(sent2))/2.0)        



### sentence to vectore ==============================================>>>
    



# 1st last bigram count

def first_last_bigram(q1,q2):
    q1.replace(".","")
    q2.replace(".","")
    
    q1.replace("?","")
    q2.replace("?","")
    
    if q1.split(" ")[0]==q2.split(" ")[0]:
       first=1
    else:
       first=0
       
    if q1.split(" ")[-1]==q2.split(" ")[-1]:
       last=1      
    else:
       last=0  
       
   # bigram matching counts------------------------------------->>>>
    count=0
    for p in range (0,len(q1.split())-1):
       for q in range (0,len(q2.split())-1):
           if (q1.split()[p]+q1.split()[p+1])==(q2.split()[q]+q2.split()[q+1]):
               count=count+1
               break
        
    return  first,last,count   


def stem(sentence):
    list = sentence.split(" ")
    new_list = []
    newSentence = ""
    for word in list:
        word = word.lower()
        new_list.append(word)
        newSentence += word + " "
    return newSentence, new_list


def csvRead(filename):
    list = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        flag = False
        i=0
        for row in reader:
            if flag:
                list.append(row)
            else:
                flag = True
            # i+=1
            # if i ==100:
            #     break
        print ("read done")
    return list




def arrayMaker(sentencex, sentencey, length,widf):
    toreturn = numpy.zeros(shape=(length, 2), dtype=numpy.float32)
    model = Word2Vec.load("cosword2vec.bin")
    # model.save("word2vec.bin")
    # vocab = model.wv.vocab.keys()
    # print (vocab,len(vocab))
    for i in range(len(sentencex)):
        tf1=calculateTf(sentencex[i])
        tf2=calculateTf(sentencey[i])
        a1 = numpy.zeros(shape=(len(sentencex[i]), 100), dtype=numpy.float32)
        a2 = numpy.zeros(shape=(len(sentencey[i]), 100), dtype=numpy.float32)
        for j, word1 in zip(range(len(sentencex[i])), sentencex[i]):
            a1[j] = model[word1]
            tfidf=tf1[word1]*widf[word1]
            # print (a1[j],tfidf)
            a1[j]=a1[j].dot(tfidf)
        for j, word2 in zip(range(len(sentencex[i])), sentencey[i]):
            a2[j] = model[word2]
            tfidf=tf2[word2]*widf[word2]
            a2[j]=a2[j].dot(tfidf)

        a = numpy.mean(a1, axis=0, dtype=numpy.float32)
        b = numpy.mean(a2, axis=0, dtype=numpy.float32)
        toreturn[i][0] = cos_sim(a, b)
        print (i,cos_sim(a, b))
        print (toreturn[i][0])
    return toreturn


def createModel(list, llist):
    sentences = []
    sentencex = []
    sentencey = []
    sentencex1 = []
    sentencey1 = []
    # csentence =[]
    print("model cretion started",len(list),len(llist))
    for i, row in zip(range(len(list)), list):
        row[3], list1 = stem(row[3].translate(None, string.punctuation))
        row[4], list2 = stem(row[4].translate(None, string.punctuation))

        sentences.append(list1)
        sentences.append(list2)
        sentencex.append(list1)
        sentencey.append(list2)
        # csentence.append(' '.join(list1))
        # csentence.append(' '.join(list2))
        # print (row[3])
        print ("first",i)
    print("model cretion started",len(list),len(llist))

    for i, row in zip(range(len(llist)), llist):
        row[1], list1 = stem(row[1].translate(None, string.punctuation))
        row[2], list2 = stem(row[2].translate(None, string.punctuation))

        sentences.append(list1)
        sentences.append(list2)
        sentencex1.append(list1)
        sentencey1.append(list2)
        # print (row[3])
        #print ("first1",i)
    print ("over")
    #model = Word2Vec(sentences, min_count=1, size=100)
    #model.save("cosword2vec.bin")
    #print("word2vec created")
    ## idf  words and value
    wordsIdf = {}
    for sentence in sentences:
        wordsIdf=countWordInDoc(sentence,wordsIdf)
    wordsIdf=calculteIdf(wordsIdf,len(sentences))
    print ("idf done")
    #return csentence,sentencex, sentencey,sentencex1, sentencey1,wordsIdf
    return [],sentencex, sentencey,sentencex1, sentencey1,wordsIdf
    #return [],[], [],[], [],[]


def createCommonList(list):
    sentencex = []
    sentencey = []
    for i, row in zip(range(len(list)), list):
        row[1], list1 = stem(row[2].translate(None, string.punctuation))
        row[2], list2 = stem(row[2].translate(None, string.punctuation))

        sentencex.append(list1)
        sentencey.append(list2)

    return sentencex, sentencey

def cos_sim(a, b):
    dot_product = numpy.dot(a, b)
    norm_a = numpy.linalg.norm(a)
    norm_b = numpy.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def csv_write(y,flag=False):
    if flag:
        data = open('TestCosineData.csv', 'w')
    else:
        data = open('TrainCosineData.csv', 'w')
    csvwriter = csv.writer(data)
    print ("entered")
    csvwriter.writerow(["cosine", "jackard"])
    ############
    for i in range(len(y)):
        # j=i+1
        print (y[i][0])
        raw = [y[i][0], int(y[i][1])]
        csvwriter.writerow(raw)
        # raw_input()
    data.close()


def csv_write1(y):
    data = open('unk.csv', 'w')
    csvwriter = csv.writer(data)
    print ("entered")
    # csvwriter.writerow(["cosine","jackard"])
    ############
    y = numpy.array(set(y))
    for i in range(y.size):
        # j=i+1
        # print (y[i][0])
        raw = [y[i]]
        csvwriter.writerow(raw)
        # raw_input()
    data.close()

def countWordInDoc(sentence,wordsIdf):
    d=[]
    for word in sentence:
        if word not in d:
            wordsIdf[word] = wordsIdf.get(word, 0) + 1
            d.append(word)
    return wordsIdf

def calculateTf(sentence):
    d={}
    for word in sentence:
        d[word] = d.get(word, 0) + 1
    length=len(sentence)
    for key in d:
        d[key]/=float(length)
    return d


def jackardSimilarity(str1,str2):
    str11=set(str1)
    # print (type(str1))
    str21=set(str2)
    str3=set(str1+str2)
    union=len(str3)
    intersection=len(str11)+len(str21)-len(str3)
    toreturn=float(intersection) / float(union)
    #
    # print (union,intersection)
   #print (toreturn)
    # raw_input("fds")

    return toreturn


def calculteIdf(wordsIdf,noOfDoc):
    for key in wordsIdf:
        wordsIdf[key]=math.log(noOfDoc/float(wordsIdf[key]))
    return wordsIdf

#mylist = csvRead("train.csv")
#mylist1 = csvRead("test.csv")
#q,q1, q2,tq1, tq2,widf = createModel(mylist, mylist1)
#calculated_data = arrayMaker(q1, q2, len(mylist),widf)
#csv_write(calculated_data)
#calculated_data = arrayMaker(tq1, tq2, len(mylist1),widf)
#csv_write(calculated_data,True)
# csv_write1(unk)
#odel = Word2Vec.load("cosword2vec.bin")

d2v_model = g.Doc2Vec.load('doc2vecMOdel_Train.doc2vec')
flag=True
while (flag):
                
        sentence1= raw_input("enter sentence1 :")
        sentence2= raw_input("enter sentence2 :")
        
        #tf1=calculateTf(sentence1)
        #tf2=calculateTf(sentence2)
#        a1 = numpy.zeros(shape=(len(sentence1), 100), dtype=numpy.float32)
#        a2 = numpy.zeros(shape=(len(sentence2), 100), dtype=numpy.float32)
#        for j, word1 in zip(range(len(sentence1)), sentence1):
#            a1[j] = model[word1]
#            tfidf=tf1[word1]*widf[word1]
#            a1[j]=a1[j].dot(tfidf)
#        for j, word2 in zip(range(len(sentence2)), sentence2):
#            a2[j] = model[word2]
#            tfidf=tf2[word2]*widf[word2]
#            a2[j]=a2[j].dot(tfidf)
#        
#        a = numpy.mean(a1, axis=0, dtype=numpy.float32)
#        b = numpy.mean(a2, axis=0, dtype=numpy.float32)
#        cosfeature=cos_sim(a, b)
        cosfeature=0.8
        jaccardfeature=jackardSimilarity(sentence1,sentence2)
        ldfeature=jellyfish.levenshtein_distance(sentence1,sentence2)
        hammingdistance=jellyfish.hamming_distance(sentence1,sentence2)
        jarofeature=jellyfish.jaro_winkler(sentence1,sentence2)
        
        
        lsStr1 = sentence1.split();
        lsStr2 = sentence2.split();
        wfetaure=d2v_model.docvecs.similarity_unseen_docs(d2v_model, lsStr1, lsStr2, alpha=0.01, min_alpha=0.001, steps=300);
        
        question_1=preprocess(sentence1)
        question_2=preprocess(sentence2)
        wordnetscore=wordnetSentenceSimilarity(question_1, question_2)
        first,last,bigram=first_last_bigram(sentence1,sentence2)

                
        X_train = numpy.zeros(shape=(1,10))#] * 10) for row in range(10) ]
        i=0
        
        while i < 1:
               X_train[i][0]= cosfeature
               X_train[i][1]= jaccardfeature
               X_train[i][2]=wordnetscore
               X_train[i][3]=wfetaure
               X_train[i][4]=first
               X_train[i][5]=last
               X_train[i][6]=bigram
               X_train[i][7]=ldfeature
               X_train[i][8]=hammingdistance
               X_train[i][9]=jarofeature
               i=i+1
               
        clf=joblib.load("randomForest.pkl")
#        print ("jaccardfeature" , jaccardfeature)
#        print ("wordnetscore",wordnetscore)
#        print ("hammingdistance",hammingdistance)
        pred = clf.predict(X_train)
        print"prediction is"+str(pred)
#        if pred==1:
#            print "questions are similar"
#        else:
#            print "questions are not similar"        
#            

        