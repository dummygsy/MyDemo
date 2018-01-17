'''
This method uses Bayesian Inference Method to perform Spelling Correction
'''
import re, collections


# splitwords() will split all words from the input file, and save the output as a word list
def splitwords(text): 
    return re.findall('[a-z]+', text.lower())
#    return re.findall(r'\w+', text.lower())



# train() will build a dictionary with k-v pair, where key is the word and value is the count of that word
def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

# WORDS is the list of splitted words, like ['c', 'for', 'c', 'in', 's']
WORDS = splitwords(open('Spelling_Correction_Bayesian_Inference.txt').read())

# NWORDS is the list(class 'collections.defaultdict') of splitted words and their count, 
# like {'wiki': 2, 'xrange': 2, 'ord': 4, 'min': 2}
NWORDS = train(WORDS)


# edits1() will return all words that are one edit away from `word`
def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


# edits2() will return all words that are two edit away from `word`
def edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1))


# known_edits2() will return all words that are two edit away from `word` in known word
def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)


# known() will return the subset of `words` that appear in the dictionary of WORDS.
def known(words): 
    return set(w for w in words if w in WORDS)

# candidates() will generate possible spelling corrections for word.
# first check if the input word is in vocabulary or not, if so, return the word
# If the input word is not in vocabulary, then check one edit candidates in vocabulary
# If no one edit candidates in vocabulary, then check two edit candidates in vocabulary
# Otherwise, return the orginal input word directly (no spelling correction)
def candidates(word): 
#   return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
    return (known([word]) or known(edits1(word)) or known_edits2(word)) or [word])

# correction will return the most probable spelling correction for word.
def correction(word): 
    return max(candidates(word), key=NWORDS.get)


'''
>>> known(['abcde'])
set()
>>> known(edits1('abcde'))
{'abide', 'abode'}
>>> known_edits2('abcde')
{'aide', 'abbe', 'abe', 'abc', 'bide', 'ace', 'bache', 'aside', 'acne', 'abuse', 'above', 'anode', 'abide', 'acre', 'able', 'acme', 'abodes', 'arcade', 'abate', 'bade', 'acte', 'ache', 'abode'}

>>> known(['a'])
{'a'}
>>> known(['a','b'])
{'a', 'b'}

>>> known_edits2('abcdef')
{'abide', 'abodes', 'abode'}
>>>
>>>
>>> known(edits1('abcdef'))
set()

'''
