import urllib.request
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords


''' Download the webpage '''
response = urllib.request.urlopen('http://php.net/')

''' Get the HTML raw data '''
html = response.read()
#print ("html file:")
#print (html)

''' Get the processed clean text '''
soup = BeautifulSoup(html)
#print ("soup:")
#print (soup)
text = soup.get_text(strip=True)
#print ("processed text:")
#print (text)

''' Split text to tokens '''
tokens = [t for t in text.split()]
#print ("tokens:")
#print (tokens)

''' Remove stop words '''	
clean_tokens = tokens[:]

# nltk.download('stopwords') # just download for the 1st time
sr = stopwords.words('english')
 
for token in tokens:
    if token in stopwords.words('english'):
        clean_tokens.remove(token)

		

''' Count Word Frequence '''
freq = nltk.FreqDist(tokens)
for key,val in freq.items():
    print (str(key) + ':' + str(val))

freq.plot(20, cumulative=False)
