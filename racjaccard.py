import numpy
import csv
import string
from string import punctuation
from gensim.models import Word2Vec
import gensim
from gensim.parsing import PorterStemmer

global_stemmer = PorterStemmer()


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
        for row in reader:
            if flag:
                list.append(row)
            else:
                flag = True
    return list


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
    print (toreturn)
    # raw_input("fds")

    return toreturn

def arrayMaker(list):
    toreturn = numpy.zeros(shape=(len(list), 2), dtype=numpy.float32)
    # translation = string.maketrans("", "", string.punctuation);
    # translation = string.maketrans("", punctuation)
    # new = words.translate(translation)
    # model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #sentences = []
    #sentencex = []
    #sentencey = []
    for i, row in zip(range(len(list)), list):
        #print (row[1], row[2])
        row[1], list1 = stem(row[1].translate(None, string.punctuation))
        row[2], list2 = stem(row[2].translate(None, string.punctuation))

        toreturn[i][1]=jackardSimilarity(list1,list2)
    return toreturn

def cos_sim(a, b):
    dot_product = numpy.dot(a, b)
    norm_a = numpy.linalg.norm(a)
    norm_b = numpy.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def csv_write(y):
    data = open('TestJackardData.csv', 'w')
    csvwriter = csv.writer(data)
    print ("entered")
    csvwriter.writerow(["cosine","jackard"])
    ############
    for i in range(len(y)):
        # j=i+1
        print (y[i][1])
        raw = [y[i][0],y[i][1]]
        csvwriter.writerow(raw)
        # raw_input()
    data.close()
mylist = csvRead("test.csv")
calculated_data=arrayMaker(mylist)
csv_write(calculated_data)