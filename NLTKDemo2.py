# https://likegeeks.com/nlp-tutorial-using-python-nltk/

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
 
myentext = "Hello Mr. Adam, how are you? I hope everything is going well. Today is a good day, see you dude."

print("Plan Text:") 
print(myentext)

''' 
Tokenization process means splitting bigger parts to small parts. 
'''
''' sentence tokenize '''
print("")
print("Sentence Tokenize:") 
print(sent_tokenize(myentext))

''' word tokenize '''
print("")
print("Word Tokenize:") 
print(word_tokenize(myentext))


 
''' get word definition ''' 
syn = wordnet.synsets("pain")
print("")
print(syn[0].definition()) 
print(syn[0].examples())

''' get synonymous words '''
synonyms = [] 
for syn in wordnet.synsets('Computer'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print("")
print(synonyms)

''' get antonyms words '''
antonyms = [] 
for syn in wordnet.synsets('Small'):
    for lemma in syn.lemmas():
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())
print("")
print(antonyms)


'''
Word stemming means removing affixes from words and return the root word. Ex: The stem of the word working => work.
NLTK has a class called PorterStemmer which uses Porter stemming algorithm.
'''

stemmer = PorterStemmer()
print("")
print("The stem word of working is", stemmer.stem('working'))
print("The stem word of works is", stemmer.stem('works'))
print("The stem word of worked is", stemmer.stem('worked'))


'''
Word lemmatizing is similar to stemming, but the difference is the result of lemmatizing is a real word.
'''
print("")
print("The stem word of increases is", stemmer.stem('increases'))

lemmatizer = WordNetLemmatizer() 
print("The lemmatizing word of increases is", lemmatizer.lemmatize('increases'))

''' Set the result as verb, noun, adjective, or adverb '''
print("The verb lemmatizing word of increases is", lemmatizer.lemmatize('playing', pos="v"))
print("The noun lemmatizing word of increases is", lemmatizer.lemmatize('playing', pos="n"))
print("The adj lemmatizing word of increases is", lemmatizer.lemmatize('playing', pos="a"))
print("The adv lemmatizing word of increases is", lemmatizer.lemmatize('playing', pos="r"))


''' Stemming and Lemmatization Difference '''
print('----------------------')
print(stemmer.stem('stones'))
print(stemmer.stem('speaking'))
print(stemmer.stem('bedroom'))
print(stemmer.stem('jokes'))
print(stemmer.stem('lisa'))
print(stemmer.stem('purple'))
 
print('----------------------')
print(lemmatizer.lemmatize('stones'))
print(lemmatizer.lemmatize('speaking'))
print(lemmatizer.lemmatize('bedroom'))
print(lemmatizer.lemmatize('jokes'))
print(lemmatizer.lemmatize('lisa'))
print(lemmatizer.lemmatize('purple'))
