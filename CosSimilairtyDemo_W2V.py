import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class DocSim(object):
    def __init__(self, w2v_model , stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords
    
	# This methord "vectorize" return the vector of the input parameter 'doc'.
	# The vector of a word is a 300-dimentinal vector. 
	# The vector of a doc is a 300-dimentinal vector, which is a mean of all its word vector. 
	#
	# If one word of the doc is not in the model, then the word vector will not be added into the word_vecs list. For example, if the doc is "ready-mixed concrete", it will be splited into 2 words "ready-mixed" and "concrete". "ready-mixed" is not in the model while "concrete" is in the model, so the word_vecs list only includes one element which is the word vector of "concrete". The return value is the vector value of "concrete"
	# If all the word in the doc are not in the model, word_vecs list is empty. The return value is nan.
    def vectorize(self, doc):
        ''' Identify the vector values for each word in the given document '''
        doc = doc.lower()
        words = [w for w in doc.split(" ") if w not in self.stopwords]
        word_vecs = []
        for word in words:
            try:
                vec = self.w2v_model[word]
                word_vecs.append(vec)
            except KeyError:
                ''' Ignore, if the word doesn't exist in the vocabulary '''
                pass
    
        ''' Assuming that document vector is the mean of all the word vectors '''
        # TODO: DocVec is the mean value of all word vectors. Is there any better way to calculate the doc vector?
        vector = np.mean(word_vecs, axis=0)
        return vector
    
    
    def _cosine_sim(self, vecA, vecB):
        '''Find the cosine similarity distance between two vectors.'''
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim
    
    def calculate_similarity(self, source_doc, target_docs=[], threshold=0):
        '''Calculates & returns similarity scores between given source document & all
        the target documents.'''
        if isinstance(target_docs, str):
            target_docs = [target_docs]
    
        source_vec = self.vectorize(source_doc)
        results = []
        for target_doc in target_docs:
            target_vec = self.vectorize(target_doc)
            sim_score = self._cosine_sim(source_vec, target_vec)
            if sim_score >= threshold:
                results.append({
                    'score' : sim_score,
                    'target_doc' : target_doc
                })
            ''' Sort results by score in desc order '''
            results.sort(key=lambda k : k['score'] , reverse=True)
    
        return results


# This model is 3.4G, need long time to load and run.
# Add a parameter 'limit': Sets a maximum number of word-vectors to read from the file. The default, None, means read all.

# When model_loading_limit = 20000, the word "Rebar" is not in the model. So the cosine similarity values are all 0s.
# While when model_loading_limit = 200000, the word "Rebar" is in the model. So the cosine similarity values are NOT all 0s.
'''
source_doc[ 78 ]: Rebar
[{'score': 0, 'target_doc': 'Adhesives, mortars, plasters'}, {'score': 0, 'target_doc': 'Painting products'}, {'score': 0, 'target_doc': 'Tiles'}, {'score': 0, 'target_doc': 'Windows'}, {'score': 0, 'target_doc': 'Floor coverings'}, {'score': 0, 'target_doc': 'Balcony, precast concrete, installation'}, {'score': 0, 'target_doc': 'Earth Construction Equipment'}, {'score': 0, 'target_doc': 'Groundwork products'}, {'score': 0, 'target_doc': 'Frame Products'}, {'score': 0, 'target_doc': 'Sand'}, {'score': 0, 'target_doc': 'Gravel'}, {'score': 0, 'target_doc': 'Concrete products'}, {'score': 0, 'target_doc': 'FILIGRAN composite ceiling plates'}, {'score': 0, 'target_doc': 'Precast Staircase'}, {'score': 0, 'target_doc': 'Ready-Mix Concrete C 20/25'}, {'score': 0, 'target_doc': 'Metal products'}, {'score': 0, 'target_doc': 'Reinforcement Steel Mesh 2x3m'}, {'score': 0, 'target_doc': 'Reinforcement Steel  Bar 3m'}, {'score': 0, 'target_doc': 'Masonry products'}, {'score': 0, 'target_doc': 'Vertically Perforated Bricks'}, {'score': 0, 'target_doc': 'Hollow Block Bricks'}, {'score': 0, 'target_doc': 'Timber products'}, {'score': 0, 'target_doc': 'Construction Timber'}, {'score': 0, 'target_doc': 'Personal Protective Equipment'}, {'score': 0, 'target_doc': 'Footware'}, {'score': 0, 'target_doc': 'Working Gloves'}, {'score': 0, 'target_doc': 'Helmets'}, {'score': 0, 'target_doc': 'Accessories'}, {'score': 0, 'target_doc': 'Tile'}, {'score': 0, 'target_doc': 'Finishes'}, {'score': 0, 'target_doc': 'Civil ( Structural & Archtectural)'}, {'score': 0, 'target_doc': 'Electrical'}, {'score': 0, 'target_doc': 'Mechanical'}, {'score': 0, 'target_doc': 'Precast Elements'}]

source_doc[ 78 ]: Rebar
[{'score': 0.57177895, 'target_doc': 'Reinforcement Steel Mesh 2x3m'}, {'score': 0.5511488, 'target_doc': 'Reinforcement Steel  Bar 3m'}, {'score': 0.47597337, 'target_doc': 'Precast Staircase'}, {'score': 0.4522987, 'target_doc': 'Precast Elements'}, {'score': 0.43476433, 'target_doc': 'Balcony, precast concrete, installation'}, {'score': 0.42718527, 'target_doc': 'Construction Timber'}, {'score': 0.4161016, 'target_doc': 'Metal products'}, {'score': 0.40608576, 'target_doc': 'Concrete products'}, {'score': 0.40518692, 'target_doc': 'Masonry products'}, {'score': 0.3983916, 'target_doc': 'FILIGRAN composite ceiling plates'}, {'score': 0.38799053, 'target_doc': 'Vertically Perforated Bricks'}, {'score': 0.36670727, 'target_doc': 'Tile'}, {'score': 0.35847, 'target_doc': 'Ready-Mix Concrete C 20/25'}, {'score': 0.35665724, 'target_doc': 'Gravel'}, {'score': 0.35602576, 'target_doc': 'Sand'}, {'score': 0.32999325, 'target_doc': 'Hollow Block Bricks'}, {'score': 0.31820607, 'target_doc': 'Tiles'}, {'score': 0.31761798, 'target_doc': 'Earth Construction Equipment'}, {'score': 0.31652904, 'target_doc': 'Timber products'}, {'score': 0.2586682, 'target_doc': 'Floor coverings'}, {'score': 0.2582532, 'target_doc': 'Electrical'}, {'score': 0.2188361, 'target_doc': 'Frame Products'}, {'score': 0.18543155, 'target_doc': 'Civil ( Structural & Archtectural)'}, {'score': 0.17717923, 'target_doc': 'Windows'}, {'score': 0.16259964, 'target_doc': 'Adhesives, mortars, plasters'}, {'score': 0.14870383, 'target_doc': 'Mechanical'}, {'score': 0.13635552, 'target_doc': 'Painting products'}, {'score': 0.13530707, 'target_doc': 'Finishes'}, {'score': 0.13331111, 'target_doc': 'Helmets'}, {'score': 0.13241489, 'target_doc': 'Personal Protective Equipment'}, {'score': 0.12080674, 'target_doc': 'Accessories'}, {'score': 0.10411295, 'target_doc': 'Groundwork products'}, {'score': 0.10372519, 'target_doc': 'Working Gloves'}, {'score': 0, 'target_doc': 'Footware'}]
'''


model_loading_limit = 20000
model_path = './GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit = model_loading_limit)
		
ds = DocSim(w2v_model)

source_doc = 'Ready-Mixed Concrete'
target_docs = ['Ready-Mixed', 'Concrete', 'Concrete Products']
# For source_doc 'Ready-Mixed Concrete', since ready-mixed is not included in the Google News Vector Model, it will be considered as the same as 'Concrete';
# For target_doc 'Ready-Mixed', since ready-mixed is not included in the Google News Vector Model, its vector is nan;
# For target_doc 'Concrete Products', its vector is the mean value of vector of 'Concrete' and 'Products';
#
# The cos similarity score of 'Ready-Mixed Concrete' and 'Concrete' is 1, since source_doc 'Ready-Mixed Concrete' has been be considered as the same as 'Concrete';
# The cos similarity score of 'Ready-Mixed Concrete' and 'Concrete Products' is 0.755, since the vector of 'Concrete Products' is the mean value of word vector 'Concrete' and word vector 'Products';
# The cos similarity score of 'Ready-Mixed Concrete' and 'Ready-Mixed' is 0, since the vector of 'Ready-Mixed' is nan;

''' This will return 3 target docs with similarity score '''
sim_scores = ds.calculate_similarity(source_doc, target_docs)

print("Sample:")
print("source_doc:", source_doc)
print(sim_scores)



# Similarity Checking
des1 = ['Plaster', 'Paints', 'Tiles', 'Windows', 'Floors', 'Vinyl', 'Boards', 'Emulsion Paint', 'Ceramic Tiles', 'Metal Frame Windows', 'Porcelain Tiles', 'Natural Stone Tiles', 'Mineral Plaster', 'Groundwork and Mining Materials', 'Framework Materials', 'Groundwork products', 'Sand', 'Gravel', 'Concrete products', 'FILIGRAN composite ceiling plates', 'Precast Staircase', 'Ready-Mix Concrete C 20/25', 'Metal products', 'Reinforcement Steel Mesh 2x3m', 'Reinforcement Steel  Bar 3m', 'Masonry products', 'Vertically Perforated Bricks', 'Hollow Block Bricks', 'Timber products', 'Construction Timber', 'General Goods', 'Footware', 'Working Gloves', 'Helmets', 'Sanitary Ware', 'Shower Faucet', 'Basin Faucet', 'Fittings', 'Sanitary Ware Fittings', 'Fittings', 'Sanitary Ware', 'Shower Faucet', 'Basin Faucet', 'Sanitary Ware Fittings', 'Tiles', 'Living Room Tile', 'Bath Room Tile', 'Kitchen Tile', 'Living Room Tile', 'Bath Room Tile', 'Kitchen Tile', 'Tiles', 'Structural and Steel Products', 'Concrete & Clay Products', 'Quarry and Premix Products', 'Timber', 'Electrical', 'LV Cable', 'HV Cable', 'Busbar', 'Cable', 'Pipe', 'Ducting', 'ACMV Equipment', 'Chiller', 'Cooling Tower', 'ACMV Valve', 'ACMV Pump', 'ACMV Insulation', '100x100mm duct', 'Hollow Core', 'Solid', 'Walls', 'Beams', 'Balconies', 'Slab', 'Starils', 'Columns', 'Rebar', 'Blocks']

des2 = ['Adhesives, mortars, plasters','Painting products', 'Tiles', 'Windows', 'Floor coverings', 'Balcony, precast concrete, installation', 'Earth Construction Equipment', 'Groundwork products', 'Frame Products', 'Sand', 'Gravel', 'Concrete products', 'FILIGRAN composite ceiling plates', 'Precast Staircase', 'Ready-Mix Concrete C 20/25', 'Metal products', 'Reinforcement Steel Mesh 2x3m', 'Reinforcement Steel  Bar 3m', 'Masonry products', 'Vertically Perforated Bricks', 'Hollow Block Bricks', 'Timber products', 'Construction Timber', 'Personal Protective Equipment', 'Footware', 'Working Gloves', 'Helmets', 'Accessories', 'Tile', 'Finishes', 'Civil ( Structural & Archtectural)', 'Electrical', 'Mechanical', 'Precast Elements']

print("~~~~~~~~~~~~")
result = []

# des1[76], "Starils", the socre of "Starils" are all 0s. To avoid issue (when threshold is default value 0, and all socres are 0, then no results in the return value of calculate_similarity() ), change "if sim_score > threshold:" to "if sim_score >= threshold:".

# Why the socre of "Starils" are all 0s? The answer is that "Starils", this word is not in the w2v model. "vec = self.w2v_model[word]" will return a KeyError, and the return value of vectorize("Starils") is nan. The cosine simalirty value of a nan vector and other vector is always 0.

'''
sim_scores = ds.calculate_similarity(des1[76], des2)
print(sim_scores)
'''

'''
[{'target_doc': 'Adhesives, mortars, plasters', 'score': 0}, {'target_doc': 'Painting products', 'score': 0}, {'target_doc': 'Tiles', 'score': 0}, {'target_doc': 'Windows', 'score': 0}, {'target_doc': 'Floor coverings', 'score': 0}, {'target_doc': 'Balcony, precast concrete, installation', 'score': 0}, {'target_doc': 'Earth Construction Equipment', 'score': 0}, {'target_doc': 'Groundwork products', 'score': 0}, {'target_doc': 'Frame Products', 'score': 0}, {'target_doc': 'Sand', 'score': 0}, {'target_doc': 'Gravel', 'score': 0}, {'target_doc': 'Concrete products', 'score': 0}, {'target_doc': 'FILIGRAN composite ceiling plates', 'score': 0}, {'target_doc': 'Precast Staircase', 'score': 0}, {'target_doc': 'Ready-Mix Concrete C 20/25', 'score': 0}, {'target_doc': 'Metal products', 'score': 0}, {'target_doc': 'Reinforcement Steel Mesh 2x3m', 'score': 0}, {'target_doc': 'Reinforcement Steel  Bar 3m', 'score': 0}, {'target_doc': 'Masonry products', 'score': 0}, {'target_doc': 'Vertically Perforated Bricks', 'score': 0}, {'target_doc': 'Hollow Block Bricks', 'score': 0}, {'target_doc': 'Timber products', 'score': 0}, {'target_doc': 'Construction Timber', 'score': 0}, {'target_doc': 'Personal Protective Equipment', 'score': 0}, {'target_doc': 'Footware', 'score': 0}, {'target_doc': 'Working Gloves', 'score': 0}, {'target_doc': 'Helmets', 'score': 0}, {'target_doc': 'Accessories', 'score': 0}, {'target_doc': 'Tile', 'score': 0}, {'target_doc': 'Finishes', 'score': 0}, {'target_doc': 'Civil ( Structural & Archtectural)', 'score': 0}, {'target_doc': 'Electrical', 'score': 0}, {'target_doc': 'Mechanical', 'score': 0}, {'target_doc': 'Precast Elements', 'score': 0}]
'''


# Get the highest score target doc from all target docs
for i in range(0, len(des1)):
    sim_scores = ds.calculate_similarity(des1[i], des2)
    result.append(sim_scores[0].get('target_doc'))
    
    print("source_doc[", i, "]:", des1[i])
    print(sim_scores)


# Print the auto mapping result
print("The auto mapping result:")
for i in range(0, len(des1)):
    print(des1[i] + " == " + result[i])

