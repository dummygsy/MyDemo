from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_sim_s(text_files):
   documents = [open(f) for f in text_files]
   tfidf = TfidfVectorizer().fit_transform(documents)
   # no need to normalize, since Vectorizer will return normalized tf-idf
   pairwise_similarity = tfidf * tfidf.T
   print(pairwise_similarity)

vect = TfidfVectorizer(min_df=1)
tfidf = vect.fit_transform(["I'd like an apple", "An apple a day keeps the doctor away", "Never compare an apple to an orange", "I prefer scikit-learn to Orange"])
print((tfidf * tfidf.T).A)
