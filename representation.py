from preprocessing import texts, doc_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import texts, doc_labels
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)
vocabulary = vectorizer.get_feature_names_out() #κάνει όλο το preprocessing αυτόματα για το vocabulary
tfidf_array = tfidf_matrix.toarray()

# Φτιάχνουμε σωστά τη mapping: κάθε TF-IDF σε κάθε άρθρο
tfidf_representation_stemmed = {}
label_counts = {}

for label, tfidf in zip(doc_labels, tfidf_array):
    # Αύξηση μετρητή εγγράφων για κάθε label
    if label not in label_counts:
        label_counts[label] = 0
    doc_id = f"{label}__{label_counts[label]}"
    label_counts[label] += 1

    # Αποθήκευση με μοναδικό ID
    tfidf_representation_stemmed[doc_id] = dict(zip(vocabulary, tfidf))


#print(tfidf_representation_stemmed['_Alita__opening_weekend'])



# Step 1: Χρήση CountVectorizer με binary=True
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts).toarray()  # shape: (num_docs, vocab_size)

# Πάρε το λεξιλόγιο
vocabulary = vectorizer.get_feature_names_out()
vocab_size = len(vocabulary)

# Step 2: Ομαδοποίηση ανά κατηγορία
categories = sorted(set(doc_labels))
category_to_vectors = {cat: [] for cat in categories}

for i, label in enumerate(doc_labels):
    category_to_vectors[label].append(X[i])

# Step 3: Εύρεση μέγιστου αριθμού εγγράφων ανά κατηγορία
max_docs_per_cat = max(len(docs) for docs in category_to_vectors.values())

# Step 4: Δημιουργία κενής 3D δομής (categories, docs_per_cat, vocab_size)
boolean_3d_array = np.zeros((len(categories), max_docs_per_cat, vocab_size))


# Γέμισμα της 3D δομής
for i, cat in enumerate(categories):
    docs = category_to_vectors[cat]
    for j, vec in enumerate(docs):
        boolean_3d_array[i, j, :] = vec

# Πληροφορίες για επιβεβαίωση
#print("3D Boolean array shape:", boolean_3d_array.shape)
for i, cat in enumerate(categories):
    print(f"{cat}: {len(category_to_vectors[cat])} documents")

# Προαιρετικά: Αν θέλεις να αποθηκεύσεις το array σε αρχείο .npy
# np.save("boolean_3d_array.npy", boolean_3d_array)
#print(boolean_3d_array)