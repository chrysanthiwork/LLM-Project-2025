from preprocessing import texts, doc_labels
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
vocabulary = vectorizer.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray()

# Φτιάχνουμε σωστά τη mapping: κάθε TF-IDF σε κάθε άρθρο
tfidf_representation_stemmed = {}
for label, tfidf in zip(doc_labels, tfidf_array):
    if label not in tfidf_representation_stemmed:
        tfidf_representation_stemmed[label] = []
    tfidf_representation_stemmed[label].append(dict(zip(vocabulary, tfidf)))

# Τώρα tfidf_representation_stemmed περιέχει:
# { 'sports': [άρθρο1, άρθρο2, ...], 'health': [...], ... }
#print(tfidf_representation_stemmed['_Alita__opening_weekend'])