import nltk, string
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt') # just download for the 1st time


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

# Split the sentence into words and get the stem of each word by Porter Stemmer Algorithm
#
# >>> stemmer.stem("creates")
# 'creat'
# >>> stemmer.stem("created")
# 'creat'
# >>> stemmer.stem("create")
# 'creat'
#
# >>> stem_tokens(nltk.word_tokenize("I love a little bird!!"))
# ['I', 'love', 'a', 'littl', 'bird', '!', '!']
#
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

# Remove punctuation, lowercase, stem
# 
# Split the sentence into word tokens
# >>> nltk.word_tokenize("I love a little bird!!")
# ['I', 'love', 'a', 'little', 'bird', '!', '!']
#
# lower() become uppercase into lowercase (I become into i)
# >>> nltk.word_tokenize("I love a little bird!!".lower())
# ['i', 'love', 'a', 'little', 'bird', '!', '!']
#
# translate(remove_punctuation_map) remove punctuation (remove !)
# >>> nltk.word_tokenize("I love a little bird!!".lower().translate(remove_punctuation_map))
# ['i', 'love', 'a', 'little', 'bird']
#
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    # Learn vocabulary and idf, return term-document matrix. This is equivalent to fit followed by transform, but more efficiently implemented.
    tfidf = vectorizer.fit_transform([text1, text2]) 
    return ((tfidf * tfidf.T).A)[0,1]


# >>> tfidf = vectorizer.fit_transform(['a little bird', 'little dog'])
# >>> tfidf
# <2x3 sparse matrix of type '<class 'numpy.float64'>'
#        with 4 stored elements in Compressed Sparse Row format>
# >>> print(tfidf)
#   (0, 2)        0.5797386715376657
#   (0, 0)        0.8148024746671689
#   (1, 2)        0.5797386715376657
#   (1, 1)        0.8148024746671689
# >>> print(tfidf.T)
#   (2, 0)        0.5797386715376657
#   (0, 0)        0.8148024746671689
#   (2, 1)        0.5797386715376657
#   (1, 1)        0.8148024746671689
# >>> print(tfidf * tfidf.T)
#   (0, 1)        0.3360969272762574
#   (0, 0)        0.9999999999999999
#   (1, 1)        0.9999999999999999
#   (1, 0)        0.3360969272762574
# >>> print((tfidf * tfidf.T).A)
# [[1.         0.33609693]
#  [0.33609693 1.        ]]
# >>> print(((tfidf * tfidf.T).A)[0,1])
# 0.3360969272762574	


print(cosine_sim('a little bird', 'a little bird'))
print(cosine_sim('a little bird', 'a little bird chirps'))
print(cosine_sim('a little bird', 'a big dog barks'))

des1 = ['Plaster', 'Paints', 'Tiles', 'Windows', 'Floors', 'Vinyl', 'Boards', 'Emulsion Paint', 'Ceramic Tiles', 'Metal Frame Windows', 'Porcelain Tiles', 'Natural Stone Tiles', 'Mineral Plaster', 'Groundwork and Mining Materials', 'Framework Materials', 'Groundwork products', 'Sand', 'Gravel', 'Concrete products', 'FILIGRAN composite ceiling plates', 'Precast Staircase', 'Ready-Mix Concrete C 20/25', 'Metal products', 'Reinforcement Steel Mesh 2x3m', 'Reinforcement Steel  Bar 3m', 'Masonry products', 'Vertically Perforated Bricks', 'Hollow Block Bricks', 'Timber products', 'Construction Timber', 'General Goods', 'Footware', 'Working Gloves', 'Helmets', 'Sanitary Ware', 'Shower Faucet', 'Basin Faucet', 'Fittings', 'Sanitary Ware Fittings', 'Fittings', 'Sanitary Ware', 'Shower Faucet', 'Basin Faucet', 'Sanitary Ware Fittings', 'Tiles', 'Living Room Tile', 'Bath Room Tile', 'Kitchen Tile', 'Living Room Tile', 'Bath Room Tile', 'Kitchen Tile', 'Tiles', 'Structural and Steel Products', 'Concrete & Clay Products', 'Quarry and Premix Products', 'Timber', 'Electrical', 'LV Cable', 'HV Cable', 'Busbar', 'Cable', 'Pipe', 'Ducting', 'ACMV Equipment', 'Chiller', 'Cooling Tower', 'ACMV Valve', 'ACMV Pump', 'ACMV Insulation', '100x100mm duct', 'Hollow Core', 'Solid', 'Walls', 'Beams', 'Balconies', 'Slab', 'Starils', 'Columns', 'Rebar', 'Blocks']

des2 = ['Adhesives, mortars, plasters','Painting products', 'Tiles', 'Windows', 'Floor coverings', 'Balcony, precast concrete, installation', 'Earth Construction Equipment', 'Groundwork products', 'Frame Products', 'Sand', 'Gravel', 'Concrete products', 'FILIGRAN composite ceiling plates', 'Precast Staircase', 'Ready-Mix Concrete C 20/25', 'Metal products', 'Reinforcement Steel Mesh 2x3m', 'Reinforcement Steel  Bar 3m', 'Masonry products', 'Vertically Perforated Bricks', 'Hollow Block Bricks', 'Timber products', 'Construction Timber', 'Personal Protective Equipment', 'Footware', 'Working Gloves', 'Helmets', 'Accessories', 'Tile', 'Finishes', 'Civil ( Structural & Archtectural)', 'Electrical', 'Mechanical', 'Precast Elements']

# Get the cos similarity value 2D matrix
result = []
for i in range(0, len(des1)):
    result.append([])
    for j in range(0, len(des2)):
        result[i].append(cosine_sim(des1[i], des2[j]))
		
	
# Get the index of the largest cos similarity value
# TODO: if the resultarray[i] are all zeros, which means that no mapping, then the 1st index "0" (Adhesives, mortars, plasters) will be returned.
resultarray = numpy.array(result)
index = resultarray.argmax(axis=1)
print(index)

# Print the auto mapping result
print("The auto mapping result:")
for i in range(0, len(des1)):
    print(des1[i] + " == " + des2[index[i]])
