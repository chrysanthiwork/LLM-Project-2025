import numpy as np
import math
from representation import tfidf_representation_stemmed

def cosine_similarity(doc1, doc2):
    """
    Computes the cosine similarity between two documents represented as weighted dictionaries.

    Parameters:
        doc1 (dict): The first document with terms as keys and weights as values.
        doc2 (dict): The second document with terms as keys and weights as values.

    Returns:
        float: The cosine similarity between the two documents (range: 0 to 1).
    """
    # Compute the dot product of the two documents
    dot_product = sum(doc1.get(term, 0) * doc2.get(term, 0) for term in set(doc1) | set(doc2))

    # Compute the magnitude of each document
    magnitude_doc1 = math.sqrt(sum(weight ** 2 for weight in doc1.values()))
    magnitude_doc2 = math.sqrt(sum(weight ** 2 for weight in doc2.values()))

    # Handle edge case: if one of the documents is zero vector, similarity is 0
    if magnitude_doc1 == 0 or magnitude_doc2 == 0:
        return 0.0

    # Compute the cosine similarity
    similarity = dot_product / (magnitude_doc1 * magnitude_doc2)
    return similarity

def compute_pairwise_similarities(documents):
    # Create a list of document ids
    doc_ids = list(documents.keys())

    # Initialize an empty similarity matrix (square matrix)
    num_docs = len(documents)
    similarity_matrix = np.zeros((num_docs, num_docs))

    # Compute similarity for each pair of documents
    for i in range(num_docs):
        for j in range(i, num_docs):  # Symmetric matrix, so only calculate upper triangle
            doc1_id = doc_ids[i]
            doc2_id = doc_ids[j]
            similarity = cosine_similarity(documents[doc1_id], documents[doc2_id])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetric matrix

    return similarity_matrix

similarity_matrix = compute_pairwise_similarities(tfidf_representation_stemmed) #γίνεται υπολογισμός όλα με όλα πχ _Alita0 vs _Alita1 κλπ.
#print(similarity_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_similarity_heatmap(similarity_matrix):
    """
    Δημιουργεί ένα heatmap για τις ομοιότητες των εγγράφων χωρίς τα doc_ids.

    Parameters:
        similarity_matrix (np.array): Πίνακας ομοιοτήτων μεταξύ των εγγράφων.
    """
    plt.figure(figsize=(10, 8))

    # Χρησιμοποιούμε τη βιβλιοθήκη seaborn για να σχεδιάσουμε το heatmap
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)

    # Τίτλος και ετικέτες άξονα
    plt.title('Document Similarity Heatmap (Without Doc IDs)')
    plt.xlabel('Document Index')
    plt.ylabel('Document Index')

    # Εμφανίζουμε το γράφημα
    plt.show()

plot_similarity_heatmap(similarity_matrix)