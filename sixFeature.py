import jellyfish
import numpy
import csv
import string
from string import punctuation
from gensim.models import Word2Vec
import gensim
from gensim.parsing import PorterStemmer

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

def arrayMaker(list):
    data = open('TestSixFeature.csv', 'w')
    csvwriter = csv.writer(data)
    csvwriter.writerow(["levenshtein distance","jaro distance","damerau levenshtein distance","hamming distance","jaro winkler","match rating comparison"])
    y = numpy.zeros(6, dtype=numpy.float32)
    for i, row in zip(range(len(list)), list):
        # print (row[3], row[4], row[5])
        # row[3], list1 = stem(row[3].translate(None, string.punctuation))
        # row[4], list2 = stem(row[4].translate(None, string.punctuation))
        sentence1 = unicode(row[1].translate(None, string.punctuation), "utf-8")
        sentence2 = unicode(row[2].translate(None, string.punctuation), "utf-8")

        y[0]=jellyfish.levenshtein_distance(sentence1,sentence2)
        y[1]=jellyfish.jaro_distance(sentence1,sentence2)
        y[2]=jellyfish.damerau_levenshtein_distance(sentence1,sentence2)
        y[3]=jellyfish.hamming_distance(sentence1,sentence2)
        y[4]=jellyfish.jaro_winkler(sentence1,sentence2)
        y[5]=jellyfish.match_rating_comparison(sentence1,sentence2)
        print (i)
        raw = [y[0],y[1],y[2],y[3],y[4],y[5]]
        csvwriter.writerow(raw)
    data.close()
    return 0#toreturn

def csv_write(y):
    data = open('TestSixFeature.csv', 'w')
    csvwriter = csv.writer(data)
    print ("entered")
    csvwriter.writerow(["levenshtein distance","jaro distance","damerau levenshtein distance","hamming distance","jaro winkler","match rating comparison"])
    ############
    for i in range(len(y)):
        raw = [y[i][0],y[i][1],y[i][2],y[i][3],y[i][4],y[i][5]]
        csvwriter.writerow(raw)
        # raw_input()
    data.close()
mylist = csvRead("test.csv")
calculated_data=arrayMaker(mylist)
# csv_write(calculated_data)
