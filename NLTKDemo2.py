from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
 
mytext = "Hello Mr. Adam, how are you? I hope everything is going well. Today is a good day, see you dude."

print ("Plan Text:") 
print (mytext)

''' sentence tokenize '''
print ("")
print ("Sentence Tokenize:") 
print(sent_tokenize(mytext))

''' word tokenize '''
print ("")
print ("Word Tokenize:") 
print(word_tokenize(mytext))
