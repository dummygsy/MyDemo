import nltk, string
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


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
a = numpy.array(result)
index = a.argmax(axis=1)
