import urllib.request
from bs4 import BeautifulSoup
 
response = urllib.request.urlopen('http://php.net/')
 
html = response.read()
print ("html file:")
print (html)

soup = BeautifulSoup(html)
print ("soup:")
print (soup)
text = soup.get_text(strip=True)
print ("processed text:")
print (text)
