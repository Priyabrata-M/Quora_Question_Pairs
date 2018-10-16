import csv

def csvRead(filename,flag=False):
    list = []
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        # i=0
        for row in reader:
            # print row
            # quit()
            if flag:
                list.append(row)
            else:
                flag = True
            # if i == 5:
            #     break
            # i+=1
    return list

def listMaker(list1,list2,list3,list4,list5):
    list=[]
    print (len(list1),len(list2),len(list3),len(list4),len(list5))
    for a,b,c,d,e in zip(list1,list2,list3,list4,list5):
        temp=[]
        temp.append(c[2])
        temp.append(d[0])
        temp.append(c[1])
        temp.append(e[1])
        temp.append(a[2])
        temp.append(a[3])
        temp.append(a[4])
        temp.append(b[0])
        temp.append(b[3])
        temp.append(b[4])
        # print temp
        # quit()
        list.append(temp)
    return list

def csv_write(y):
    data = open('TestFeatures.csv', 'w')
    csvwriter = csv.writer(data)
    csvwriter.writerow(["WordnetSimilarity","Cosine","jackard","Cosine_Doc2Vec","word1stSimilarity","wordlastSimilarity","bigramMatchingCounts","levenshteinDistance","hammingDistance","jaroWinkler"])
    for i in range(len(y)):
        raw = [y[i][0],y[i][1],y[i][2],y[i][3],y[i][4],y[i][5],y[i][6],y[i][7],y[i][8],y[i][9]]
        csvwriter.writerow(raw)
    data.close()

list1=csvRead("Test_wordMatching.csv") #extract 2,3,4 "word1stSimilarity","wordlastSimilarity","bigramMatchingCounts"
list2=csvRead("TestSixFeature.csv")  #extract 0,3,4 "levenshteinDistance","hammingDistance","jaroWinkler"
list3=csvRead("FinalFeatureFile_Test.csv")  #extract 2,1 "WordnetSimilarity","jackard"
list4=csvRead("TestCosineData.csv") #extract 0 "cosine"
list5=csvRead("Doc2VecTest",True)   #extract 1 "Cosine_Doc2Vec"
list=listMaker(list1,list2,list3,list4,list5)
# print list
csv_write(list)
