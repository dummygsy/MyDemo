import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class DocSim(object):
    def __init__(self, w2v_model , stopwords=[]):
        self.w2v_model = w2v_model
        self.stopwords = stopwords

    def vectorize(self, doc):
        """Identify the vector values for each word in the given document"""
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
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def calculate_similarity(self, source_doc, target_docs=[], threshold=0):
        """Calculates & returns similarity scores between given source document & all
        the target documents."""
        if isinstance(target_docs, str):
            target_docs = [target_docs]

        source_vec = self.vectorize(source_doc)
        results = []
        for target_doc in target_docs:
            target_vec = self.vectorize(target_doc)
            sim_score = self._cosine_sim(source_vec, target_vec)
            if sim_score > threshold:
                results.append({
                    'score' : sim_score,
                    'target_doc' : target_doc
                })
            ''' Sort results by score in desc order '''
            results.sort(key=lambda k : k['score'] , reverse=True)

        return results


# This model is 3.4G, need long time to load and run.
model_path = './GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
		
ds = DocSim(w2v_model)

source_doc = 'how to delete an invoice'
target_docs = ['delete a invoice', 'how do i remove an invoice', 'purge an invoice']

''' This will return 3 target docs with similarity score '''
sim_scores = ds.calculate_similarity(source_doc, target_docs)

print("source_doc:", source_doc)
print(sim_scores)
