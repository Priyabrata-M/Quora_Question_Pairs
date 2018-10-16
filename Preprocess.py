# -*- coding: utf-8 -*-

import HTMLParser
string="I luv my iphone you’re awsm apple. DisplayIsAwesome, sooo happppppy 🙂"


####-------------------------------------------------------------------------------
#Escaping HTML characters
#html_parser = HTMLParser.HTMLParser()
#string = html_parser.unescape(string)
#print("HTML character removed string :"+string)
#print 
#Decoding data


#Apostrophe Lookup
APPOSTOPHES = {"'s" : " is", "'re" : " are"} ## Need a huge dictionary
words = string.split()
reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]
reformed = " ".join(reformed)
string=reformed#########Check#############################
print("Apostrophe removed string :"+string)

#string="When do you use シ instead of し?"
ans= string.decode("utf8").encode('ascii','ignore')
string =ans
print("unocode removed string :"+string)

#
#Removal of Stop-words: 
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
stop_words = set(stopwords.words('english'))
#stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']) # remove it if you need punctuation 
ist_of_words = [i.lower() for i in wordpunct_tokenize(string) if i.lower() not in stop_words]
string=[]
string= ' '.join(ist_of_words)    
print("stopwords removed string :"+string)

#Removal of Punctuations


#Removal of Expressions




#Standardizing words: 
import itertools 
string = ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))
print("sStandardizing words processed string :"+string)


#Split Attached Words:


print(" before split attahed word processed string :"+string)
import re
ans=""
for a in re.findall('[A-Z][^A-Z]*',string):
   ans+=a.strip()+' '
string=ans
print("split attahed word processed string :"+string)



#Slangs lookup

#Removal of URLs: 