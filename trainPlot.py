from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import csv,numpy

def readFile():
    toreturn=[]
    with open('X_TRAIN_Similarity.csv', 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
                toreturn.append(row)

    x = numpy.ones(shape=(len(toreturn), 10))
    y = numpy.ones(len(toreturn))
    for pos,row in zip(range(len(toreturn)),toreturn):
        for i,name in zip(range(len(row)),row):
            # print dat
            print name
            if name == "Is_duplicate":
                # print name
                y[pos] = row[name]
            elif name == "WordnetSimilarity":
                x[pos][0]=row[name]
            elif name == "Cosine":
                x[pos][1]=row[name]
            elif name == "jackard":
                x[pos][2]=row[name]
            elif name == "Cosine_Doc2Vec":
                x[pos][3]=row[name]
            elif name == "word1stSimilarity":
                x[pos][4]=row[name]
            elif name == "wordlastSimilarity":
                x[pos][5]=row[name]
            elif name == "bigramMatchingCounts":
                x[pos][6]=row[name]
            elif name == "levenshteinDistance":
                x[pos][7]=row[name]
            elif name == "hammingDistance":
                x[pos][8]=row[name]
            elif name == "jaroWinkler":
                x[pos][9]=row[name]
    return x,y


def plot_with_data(x, y):
    embed = TSNE(n_components=2).fit_transform(x)
    xaxis = embed[:, 0]
    yaxis = embed[:, 1]
    plt.scatter(xaxis, yaxis, s=30, c=y)
    plt.savefig("analysisReal.jpg")
    plt.show()
    plt.close()

x,y=readFile()
plot_with_data(x,y)
