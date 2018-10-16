import numpy
import csv
import string
from string import punctuation
from gensim.models import Word2Vec
import gensim
import math
from sklearn.feature_extraction.text import  TfidfVectorizer


# from gensim.parsing import PorterStemmer

# global_stemmer = PorterStemmer()


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


# def jackardSimilarity(str1,str2):

def arrayMakerUsingGoogleWord2vec(list):
    toreturn = numpy.zeros(shape=(len(list), 2), dtype=numpy.float32)
    # translation = string.maketrans("", "", string.punctuation);
    # translation = string.maketrans("", punctuation)
    # new = words.translate(translation)
    # model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    sentences = []
    sentencex = []
    sentencey = []
    for i, row in zip(range(len(list)), list):
        # print (row[3], row[4], row[5])
        row[3], list1 = stem(row[3].translate(None, string.punctuation))
        row[4], list2 = stem(row[4].translate(None, string.punctuation))

        sentences.append(list1)
        sentences.append(list2)
        # sentences+=row[3]+". "+row[4]+". "
        sentencex.append(list1)
        sentencey.append(list2)
        #print (row[3])
        # if (i > 100):
        #     break
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
                                                            binary=True)  # Word2Vec(sentences, min_count=1, size=100)
    # vocab = model.wv.vocab.keys()
    # print (vocab,len(vocab))

    #    print (model['raches'])
    # print (model['india'])
    # quit()
    unk = []
    for i in range(len(sentencex)):
        a1 = numpy.zeros(shape=(len(sentencex[i]), 300), dtype=numpy.float32)
        a2 = numpy.zeros(shape=(len(sentencey[i]), 300), dtype=numpy.float32)
        for j, word1 in zip(range(len(sentencex[i])), sentencex[i]):
            # for k in range(len(word1)):
            #     a1[j][k] = model[word1][k]
            #     a2[j][k] = model[word2][k]
            # print (a1[j])
            # print (a2[j])
            # print (model[word2])
            # a2[j] = model[word2]
            # print ("word1 :"+word1)
            try:
                a1[j] = model[word1]
            except:
                print ("word1 :" + word1)
                unk.append(word1)
                # print("a1 : ",a1[j])
                # print (a2[j])
                # quit()
        for j, word2 in zip(range(len(sentencex[i])), sentencey[i]):
            # print ("word2 :"+word2)
            try:
                a2[j] = model[word2]
            except:
                print ("word2 :" + word2)
                unk.append(word2)
        a = numpy.mean(a1, axis=0, dtype=numpy.float32)
        b = numpy.mean(a2, axis=0, dtype=numpy.float32)
        toreturn[i][0] = cos_sim(a, b)
    return toreturn, unk


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
    model = Word2Vec(sentences, min_count=1, size=100)
    model.save("cosword2vec.bin")
    print("word2vec created")
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

def calculteIdf(wordsIdf,noOfDoc):
    for key in wordsIdf:
        wordsIdf[key]=math.log(noOfDoc/float(wordsIdf[key]))
    return wordsIdf

mylist = csvRead("train.csv")
mylist1 = csvRead("test.csv")
q,q1, q2,tq1, tq2,widf = createModel(mylist, mylist1)
# q1, q2 = createCommonList(mylist)
# print (widf)
# quit()
# tfidfvec = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0)
# ret=tfidfvec.fit_transform(q)
# print (type(ret))
# quit()
# print (widf['would'])
calculated_data = arrayMaker(q1, q2, len(mylist),widf)
csv_write(calculated_data)
calculated_data = arrayMaker(tq1, tq2, len(mylist1),widf)
csv_write(calculated_data,True)
# csv_write1(unk)