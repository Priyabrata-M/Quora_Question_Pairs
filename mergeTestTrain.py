#Merge test and train files for preparing corpus for doc2Vec

import pandas as pd
import numpy as np

fileptr = open("final_inputDoc2Vec_test.txt",'a');

df = pd.read_csv('test.csv')

for i in range(0,len(df)):
    fileptr.write(str(df["question1"][i])+" "+str(df["question2"][i])+" ")
    
    
fileptr1 = open("final_inputDoc2Vec_test.txt",'r');
message = fileptr1.read()
print(message)
fileptr1.close()
    
#df_test = pd.read_csv('test.csv')
#
#for i in range(0,len(df_test)):
#    fileptr.write(str(df_test["question1"][i]))
#    fileptr.write("\n")         
#
#for i in range(0,len(df_test)):
#    fileptr.write(str(df_test["question2"][i]))
#    fileptr.write("\n")         