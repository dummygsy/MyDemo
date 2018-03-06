import urllib.request
from bs4 import BeautifulSoup
import nltk

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


''' Count Word Frequence '''
freq = nltk.FreqDist(tokens)
for key,val in freq.items():
    print (str(key) + ':' + str(val))

freq.plot(20, cumulative=False)
