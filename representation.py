from preprocessing import texts, doc_labels, texts_touched
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import texts, doc_labels
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ TF-IDF REPRESENTATION ============ #

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

# ============ BOOLEAN REPRESENTATION ============ #

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

#print(boolean_3d_array)

# ============ BM 25 ============ #




# ============ Singular Value Decomposition ============ #
vectorizer_svd = CountVectorizer()
doc_term_matrix = vectorizer_svd.fit_transform(texts_touched).toarray()
vocab = vectorizer_svd.get_feature_names_out() #without stem , untouched texts 
#print("Document-Term Matrix (A):") 
#print(doc_term_matrix)
#print(vocab)

# Step 3: Perform Singular Value Decomposition (SVD)
U, Lambda, Vt = np.linalg.svd(doc_term_matrix, full_matrices=False)

# U: Left singular vectors (7x7, represents documents)
#print("\nLeft Singular Matrix (U):")
#print(U)

# Lamda: Singular values (1D array of length 7)
#print("\nSingular Values (Lambda):")
#print(Lambda)

# Vt: Right singular vectors (15x7, represents terms)
#print("\nRight Singular Matrix (Vt):")
#print(Vt)

VT_times_V = np.dot(Vt, Vt.transpose()) #v_tarnspose
#print(VT_times_V)

# Step 4: Reconstruct the matrix using U, Lambda, and Vt
Lambda_diag = np.diag(Lambda)
A_reconstructed = np.dot(U, np.dot(Lambda_diag, Vt))

print("\nReconstructed Matrix (A_reconstructed):")
# print(A_reconstructed)
print(np.round(A_reconstructed, 2))

# Verify reconstruction accuracy
reconstruction_error = np.linalg.norm(doc_term_matrix - A_reconstructed)
print("\nReconstruction Error (Frobenius Norm):")
print(reconstruction_error)

print("Shape of doc_term_matrix:", doc_term_matrix.shape)
print("Shape of Vt.T:", Vt.T.shape)
print("Length of vocab:", len(vocab))
print(vocab)
# Convert Vt to a DataFrame for easier visualization
Vt_df = pd.DataFrame(Vt, columns=vocab)
print(Vt_df)

Vt_df.index.name = "Topic"
Vt_df = Vt_df.abs()  # Take absolute values to measure term importance

# Identify top terms contributing to each concept
n_top_terms = 5
concepts = {}
for concept_idx in range(Vt_df.shape[0]):
    top_terms = Vt_df.iloc[concept_idx].nlargest(n_top_terms).index.tolist()
    concepts[f"Concept {concept_idx + 1}"] = top_terms

print("\nKey Terms Contributing to Each Concept:")
for concept, terms in concepts.items():
    print(f"{concept}: {', '.join(terms)}")


# Plot singular values
plt.figure(figsize=(8, 5))
plt.bar(range(len(Lambda)), Lambda, color='skyblue', edgecolor='black', label="Singular Values")
plt.title("Singular Values and Concept Contributions")
plt.xlabel("Concept Index")
plt.ylabel("Singular Value")
plt.legend()
plt.grid()
plt.show()

# ============ LSA (Truncated SVD) ============ #