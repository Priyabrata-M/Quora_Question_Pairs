import argparse
import math
import pandas as pd
import numpy
import h5py
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from mnist import MNIST
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--is_train", type=str)

args = parser.parse_args()
model_store="neuralQuestionPair3.pkl"
model_store1="neuralQuestionPair1.pkl"

def readTrain():
    df_train = pd.read_csv('X_TRAIN_Similarity.csv')
    M = df_train.as_matrix()

    WordnetSimilarity_train = df_train['WordnetSimilarity'].values.tolist()
    Cosine_train = df_train['Cosine'].values.tolist()
    jackard_train = df_train['jackard'].values.tolist()
    Cosine_Doc2Vec_train = df_train['Cosine_Doc2Vec'].values.tolist()
    word1stSimilarity_train = df_train['word1stSimilarity'].values.tolist()
    wordlastSimilarity_train = df_train['wordlastSimilarity'].values.tolist()
    bigramMatchingCounts_train = df_train['bigramMatchingCounts'].values.tolist()
    levenshteinDistance_train = df_train['levenshteinDistance'].values.tolist()
    hammingDistance_train = df_train['hammingDistance'].values.tolist()
    jaroWinkler_train = df_train['jaroWinkler'].values.tolist()
    is_duplicate_train = df_train['Is_duplicate'].values.tolist()
    X_train = [([0] * 10) for row in range(len(M))]
    Y_train = []

    i = 0
    while i < len(Cosine_train):
        X_train[i][0] = WordnetSimilarity_train[i]
        X_train[i][1] = Cosine_train[i]
        X_train[i][2] = jackard_train[i]
        X_train[i][3] = Cosine_Doc2Vec_train[i]
        X_train[i][4] = word1stSimilarity_train[i]
        X_train[i][5] = wordlastSimilarity_train[i]
        X_train[i][6] = bigramMatchingCounts_train[i]
        X_train[i][7] = levenshteinDistance_train[i]
        X_train[i][8] = hammingDistance_train[i]
        X_train[i][9] = jaroWinkler_train[i]
        i = i + 1
    Y_train = is_duplicate_train[:]
    return X_train,Y_train

def readTest():
    df_train = pd.read_csv('TestFeatures.csv')
    M = df_train.as_matrix()

    WordnetSimilarity_train = df_train['WordnetSimilarity'].values.tolist()
    Cosine_train = df_train['Cosine'].values.tolist()
    jackard_train = df_train['jackard'].values.tolist()
    Cosine_Doc2Vec_train = df_train['Cosine_Doc2Vec'].values.tolist()
    word1stSimilarity_train = df_train['word1stSimilarity'].values.tolist()
    wordlastSimilarity_train = df_train['wordlastSimilarity'].values.tolist()
    bigramMatchingCounts_train = df_train['bigramMatchingCounts'].values.tolist()
    levenshteinDistance_train = df_train['levenshteinDistance'].values.tolist()
    hammingDistance_train = df_train['hammingDistance'].values.tolist()
    jaroWinkler_train = df_train['jaroWinkler'].values.tolist()
    X_train = [([0] * 10) for row in range(len(M))]

    i = 0
    while i < len(Cosine_train):
        X_train[i][0] = WordnetSimilarity_train[i]
        X_train[i][1] = Cosine_train[i]
        X_train[i][2] = jackard_train[i]
        X_train[i][3] = Cosine_Doc2Vec_train[i]
        X_train[i][4] = word1stSimilarity_train[i]
        X_train[i][5] = wordlastSimilarity_train[i]
        X_train[i][6] = bigramMatchingCounts_train[i]
        X_train[i][7] = levenshteinDistance_train[i]
        X_train[i][8] = hammingDistance_train[i]
        X_train[i][9] = jaroWinkler_train[i]
        i = i + 1
    return X_train


def draw(activation,accuracy):
    # l = len(value[0])
    # for i in range(k):
    #     val=value[i]
    #     itr= [j for j in range(l)]
    #     print (itr,len(val))
    plt.ylabel('accuracy')
    plt.xlabel('activation')
    plt.title('(Accuracy vs activation)')
    plt.plot(activation,accuracy)
    plt.savefig("neuralNetwork.jpg")
    # plt.clf()

def gridSearch(trainx,trainy,testx,testy):
    # iteration = [100,200,500]
    # layers=[(100, 55, 20),(50, 55, 100),(100, 70, 120)]
    # activations=['logistic','relu',"tanh"]
    # solver=['adam','sgd',"lbfgs"]
    iteration = [150]
    layers=[(100, 55, 20)]
    activations=["logistic",'relu','tanh']
    solver=['sgd']
    flag =False
    maxscore=72.1753182508
    towrite=[]
    for itr in iteration:
        accuracy=[]
        for layer in layers:
            for acti in activations:
                for sol in solver:
                    clf = MLPClassifier(hidden_layer_sizes=layer, max_iter=itr, activation=acti, solver=sol)
                    clf.fit(trainx, trainy)
                    ascore=accuracy_score(testy,clf.predict(testx))*100
                    print (ascore)
                    accuracy.append(ascore)
                    name = "layer" + str(layer) + "," + "activation " + str(acti) + "," + "solver " + str(
                        sol)+", itr "+str(itr)+", accuracy "+str(ascore)

                    towrite.append(name)
                    if flag or maxscore<ascore:
                        maxscore=ascore
                        flag=False
                        joblib.dump(clf, model_store)
        draw(activations,accuracy)
    return towrite


if args.is_train == "yes":
    towrite=[]
    x,y=readTrain()
    draw(['logistic','relu',"tanh"],[72.0615394763,71.1381175384,72.1753182508])
    xtrain,xtest,ytrain, ytest= train_test_split(x,y, test_size=0.15, random_state=40)
    towrite.extend(gridSearch(xtrain,ytrain,xtest, ytest))
    f = open("neuralQuestionPairReport", 'w')
    for i in towrite:
        f.write(str(i) + "\n\r")
    f.close()
else:
    x,y=readTrain()

    clf = MLPClassifier(hidden_layer_sizes=(100, 55, 20), max_iter=500, activation='relu', solver="lbfgs")
    clf.fit(x,y)
    joblib.dump(clf, model_store)
    print accuracy_score(y,clf.predict(x))
    x=readTest()
    clf = joblib.load(model_store1)
    y=clf.predict(x)
    fileptr = open('toUpload.csv', 'w')
    fileptr.write("test_id" + "," + "is_duplicate" + "\n")
    for i in range(len(y)):
        fileptr.write(str(i) + "," + str(y) + "\n")
    fileptr.close