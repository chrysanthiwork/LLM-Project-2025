from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from preprocessing import texts, doc_labels, texts_touched
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from representation import tfidf_representation_stemmed
from similarities import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kmeans_clusters(n_clusters, max_iter):
        # --- Βήμα 1: Μετατροπή dict-of-dicts σε matrix ---
    docs = list(tfidf_representation_stemmed.keys())
    all_terms = sorted({term for doc in tfidf_representation_stemmed.values() for term in doc.keys()})

    X = np.array([
        [doc.get(term, 0.0) for term in all_terms]
        for doc in tfidf_representation_stemmed.values()
    ])

    # --- Βήμα 2: Scaling ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Βήμα 3: KMeans ---
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', max_iter=max_iter, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    # --- Βήμα 4: Εκτύπωση clusters ---
    for idx, cluster_label in enumerate(labels):
        print(f"Article {idx+1} ({docs[idx]}) in cluster: {cluster_label}")

    # --- Βήμα 5: t-SNE Visualization ---
    tsne = TSNE(perplexity=7, n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', marker='o')
    plt.title('t-SNE Visualization of Clusters')
    plt.show()

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
#print(Vt_df)

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

num_concepts_available = len(Lambda)
if num_concepts_available < 2:
    print(f"Προσοχή: Διαθέσιμες μόνο {num_concepts_available} έννοιες. Οι 2D απεικονίσεις μπορεί να μην είναι δυνατές ή πλήρεις.")


# Για 2D απεικόνιση, θα χρησιμοποιήσουμε τις πρώτες δύο έννοιες (αν υπάρχουν).
# k_plot = min(2, num_concepts_available)


# # ==================== PLOT 1: Μπάρες Ιδιοτιμών (Singular Values) ====================
# #plt.figure(figsize=(10, 6))
# # plt.bar(range(len(Lambda)), Lambda, color='skyblue', edgecolor='black', label="Singular Values")
# # plt.title("Ιδιοτιμές (Singular Values - Σημαντικότητα Εννοιών)")
# # plt.xlabel("Δείκτης Έννοιας (Ταξινομημένες κατά Σπουδαιότητα)")
# # plt.ylabel("Τιμή Ιδιοτιμής (Μέγεθος)")
# # plt.xticks(range(len(Lambda))) # Εμφάνιση όλων των δεικτών εννοιών
# # plt.legend()
# # plt.grid(axis='y', linestyle='--')
# # plt.tight_layout()
# # plt.show()

# print("\nΕπεξήγηση Γραφήματος Ιδιοτιμών:")
# print("Κάθε μπάρα αντιστοιχεί σε μια 'λανθάνουσα έννοια' ή 'θέμα' που ανακαλύφθηκε από το SVD.")
# print("Το ύψος της μπάρας (ιδιοτιμή) δείχνει τη σπουδαιότητα ή τη 'δύναμη' αυτής της έννοιας στην εξήγηση της διακύμανσης των δεδομένων.")
# print("Οι έννοιες είναι ταξινομημένες κατά φθίνουσα σειρά σπουδαιότητας.")
# print("Αυτό το γράφημα βοηθά στην απόφαση για το πόσες έννοιες να διατηρηθούν για μείωση διαστατικότητας (π.χ., μέθοδος 'αγκώνα').")
# print("-" * 70)


# # ==================== PLOT 2: Προβολή Εγγράφων στον 2D Χώρο Εννοιών ====================
# if k_plot >= 2: # Απεικόνιση μόνο αν έχουμε τουλάχιστον 2 έννοιες
#     # Συντεταγμένες εγγράφων στον χώρο των εννοιών: U_k * S_k (όπου S_k διαγώνιος πίνακας των Lambda_k)
#     # doc_concept_vectors είναι (αριθμός_εγγράφων, αριθμός_διαθέσιμων_εννοιών)
#     doc_concept_vectors = U @ np.diag(Lambda) # U[:, :k] @ np.diag(Lambda[:k])

#     # plt.figure(figsize=(12, 8))
#     # # Χρήση των δύο πρώτων στηλών για 2D γράφημα
#     # plt.scatter(doc_concept_vectors[:, 0], doc_concept_vectors[:, 1], alpha=0.7, c='dodgerblue', s=120, edgecolors='w')

#     # Ετικέτες εγγράφων
#     doc_labels = [f"Doc {i+1}" for i in range(len(texts_touched))]
#     # Εναλλακτικά, μπορείτε να χρησιμοποιήσετε μέρη του αρχικού κειμένου αν είναι σύντομα:
#     # doc_labels = [text[:25]+"..." if len(text)>25 else text for text in texts_untouched]

#     # for i, label in enumerate(doc_labels):
#     #     plt.annotate(label, (doc_concept_vectors[i, 0], doc_concept_vectors[i, 1]),
#     #                  xytext=(8, 0), textcoords='offset points', fontsize=9)

#     # plt.xlabel(f"Έννοια 1 (Ιδιοτιμή: {Lambda[0]:.2f})", fontsize=12)
#     # plt.ylabel(f"Έννοια 2 (Ιδιοτιμή: {Lambda[1]:.2f})", fontsize=12)
#     # plt.title("Προβολή Εγγράφων στον 2D Χώρο Εννοιών (LSA)", fontsize=14)
#     # plt.axhline(0, color='grey', lw=0.7, linestyle=':')
#     # plt.axvline(0, color='grey', lw=0.7, linestyle=':')
#     # plt.grid(True, linestyle='--', alpha=0.7)
#     # plt.tight_layout()
#     # plt.show()

#     print("\nΕπεξήγηση Γραφήματος Προβολής Εγγράφων:")
#     print("Κάθε σημείο αντιπροσωπεύει ένα έγγραφο, προβεβλημένο στις δύο πιο σημαντικές έννοιες.")
#     print("Ο οριζόντιος άξονας είναι η Έννοια 1 και ο κάθετος άξονας είναι η Έννοια 2.")
#     print("Έγγραφα που βρίσκονται κοντά σε αυτόν τον χώρο θεωρούνται σημασιολογικά παρόμοια σύμφωνα με το LSA.")
#     print("Έγγραφα που είναι μακριά θεωρούνται σημασιολογικά διαφορετικά.")
#     print("Η αρχή των αξόνων (0,0) μπορεί να θεωρηθεί ως ένα 'ουδέτερο' σημείο.")
#     print("-" * 70)
# else:
#     print("\nΠαράλειψη 2D Γραφήματος Προβολής Εγγράφων: Δεν υπάρχουν αρκετές διαθέσιμες έννοιες (χρειάζονται τουλάχιστον 2).")


# # ==================== PLOT 3: Προβολή Όρων στον 2D Χώρο Εννοιών ====================
# if k_plot >= 2 and len(vocab) >=1 : # Απεικόνιση μόνο αν έχουμε τουλάχιστον 2 έννοιες και λεξιλόγιο
#     # Συντεταγμένες όρων στον χώρο των εννοιών: V_k * S_k (ή (S_k * Vt_k)^T)
#     # Vt είναι (έννοιες, όροι). Χρειαζόμαστε (όροι, έννοιες). Άρα Vt.T
#     # term_concept_vectors είναι (αριθμός_όρων, αριθμός_διαθέσιμων_εννοιών)
#     term_concept_vectors = Vt.T @ np.diag(Lambda)

#     # plt.figure(figsize=(14, 10))
#     # # Χρήση των δύο πρώτων στηλών για 2D γράφημα
#     # plt.scatter(term_concept_vectors[:, 0], term_concept_vectors[:, 1], alpha=0.7, c='crimson', s=70, edgecolors='w')

#     # # Ετικέτες όρων
#     # for i, term in enumerate(vocab):
#     #     plt.annotate(term, (term_concept_vectors[i, 0], term_concept_vectors[i, 1]),
#     #                  xytext=(6, -6), textcoords='offset points',
#     #                  fontsize=8) # Προσαρμόστε το μέγεθος γραμματοσειράς

#     # plt.xlabel(f"Έννοια 1 (Ιδιοτιμή: {Lambda[0]:.2f})", fontsize=12)
#     # plt.ylabel(f"Έννοια 2 (Ιδιοτιμή: {Lambda[1]:.2f})", fontsize=12)
#     # plt.title("Προβολή Όρων στον 2D Χώρο Εννοιών (LSA)", fontsize=14)
#     # plt.axhline(0, color='grey', lw=0.7, linestyle=':')
#     # plt.axvline(0, color='grey', lw=0.7, linestyle=':')
#     # plt.grid(True, linestyle='--', alpha=0.7)
#     # plt.tight_layout()
#     # plt.show()

#     print("\nΕπεξήγηση Γραφήματος Προβολής Όρων:")
#     print("Κάθε σημείο αντιπροσωπεύει έναν όρο, προβεβλημένο στις δύο πιο σημαντικές έννοιες.")
#     print("Όροι που βρίσκονται κοντά σε αυτόν τον χώρο τείνουν να συν-εμφανίζονται ή χρησιμοποιούνται σε παρόμοια περιβάλλοντα.")
#     print("Όροι που έχουν μεγάλη 'φόρτιση' (μακριά από την αρχή) σε μια έννοια είναι σημαντικοί για τον ορισμό αυτής της έννοιας.")
#     print("Π.χ., όροι κοντά μεταξύ τους και μακριά από την αρχή κατά μήκος του άξονα της Έννοιας 1 είναι σημασιολογικά σχετικοί και σημαντικοί για την Έννοια 1.")
#     print("-" * 70)
# else:
#     print("\nΠαράλειψη 2D Γραφήματος Προβολής Όρων: Δεν υπάρχουν αρκετές διαθέσιμες έννοιες ή όροι στο λεξιλόγιο.")

# # ==================== PLOT 4: Σημαντικότεροι Όροι ανά Έννοια (Bar Plots) ====================
# if num_concepts_available > 0 and len(vocab) > 0:
#     # Ο Vt έχει σχήμα (αριθμός_εννοιών, αριθμός_όρων)
#     # Δημιουργία DataFrame για ευκολότερο χειρισμό
#     # Οι στήλες είναι οι όροι, οι γραμμές είναι οι έννοιες
#     Vt_df_concepts_terms = pd.DataFrame(Vt, columns=vocab)
#     Vt_df_concepts_terms.index = [f"Έννοια {i+1}" for i in range(Vt.shape[0])]

#     print("\nΣημαντικότεροι Όροι που Συνεισφέρουν σε Κάθε Έννοια (Απόλυτες Φορτίσεις από Vt):")
#     n_top_terms = min(7, len(vocab)) # Εμφάνιση των top 7 όρων ή λιγότερων αν το λεξιλόγιο είναι μικρό

#     num_concepts_to_plot_terms = min(k_plot, num_concepts_available, 3) # Γράφημα για έως 3 έννοιες ή k_plot

#     for concept_idx in range(Vt_df_concepts_terms.shape[0]): # Επανάληψη για όλες τις διαθέσιμες έννοιες
#         concept_label = Vt_df_concepts_terms.index[concept_idx]
#         # Λήψη της γραμμής για την τρέχουσα έννοια (φορτίσεις όλων των όρων σε αυτή την έννοια)
#         concept_loadings = Vt_df_concepts_terms.iloc[concept_idx]

#         # Κορυφαίοι N όροι βάσει απόλυτης φόρτισης
#         top_terms_series = concept_loadings.abs().nlargest(n_top_terms)

#         print(f"\n{concept_label} (Ιδιοτιμή: {Lambda[concept_idx]:.2f}):")
#         for term, loading_abs in top_terms_series.items():
#             original_loading = concept_loadings[term] # Η αρχική φόρτιση με το πρόσημο
#             print(f"  - {term} (απόλ. φόρτ.: {loading_abs:.3f}, αρχική φόρτ.: {original_loading:.3f})")

#         # Δημιουργία γραφήματος μόνο για τις πρώτες num_concepts_to_plot_terms έννοιες
#         if concept_idx < num_concepts_to_plot_terms:
#             # Λήψη των πραγματικών φορτίσεων (με πρόσημο) για αυτούς τους κορυφαίους όρους
#             # και ταξινόμηση για καλύτερη οπτικοποίηση στο γράφημα
#             term_values_for_plot = concept_loadings[top_terms_series.index].sort_values(ascending=False)

#             # plt.figure(figsize=(10, max(4, n_top_terms * 0.6))) # Προσαρμογή ύψους βάσει n_top_terms
#             # term_values_for_plot.plot(kind='barh', color=['mediumseagreen' if x > 0 else 'lightcoral' for x in term_values_for_plot])
#             # plt.title(f"Top {n_top_terms} Όροι για την {concept_label} (Ιδιοτιμή: {Lambda[concept_idx]:.2f})", fontsize=14)
#             # plt.xlabel("Φόρτιση στην Έννοια", fontsize=12)
#             # plt.ylabel("Όρος", fontsize=12)
#             # plt.gca().invert_yaxis() # Εμφάνιση του κορυφαίου όρου στην κορυφή
#             # plt.grid(axis='x', linestyle='--', alpha=0.7)
#             # plt.tight_layout()
#             # plt.show()
# else:
#     print("\nΠαράλειψη Ανάλυσης Σημαντικότερων Όρων: Δεν υπάρχουν διαθέσιμες έννοιες ή όροι.")



print("\nΟλοκληρώθηκαν οι οπτικοποιήσεις και η ανάλυση SVD.")



# --- ΒΗΜΑ Α: SVD & Δημιουργία Διανυσμάτων ---
print("--- ΒΗΜΑ Α: Εκτέλεση Singular Value Decomposition (SVD) ---")
vectorizer_svd = CountVectorizer()
doc_term_matrix = vectorizer_svd.fit_transform(texts_touched).toarray()
U, Lambda, Vt = np.linalg.svd(doc_term_matrix, full_matrices=False)

# Δημιουργία των διανυσμάτων που θα χρησιμοποιηθούν για clustering
doc_concept_vectors = U @ np.diag(Lambda)
print(f"Δημιουργήθηκαν {doc_concept_vectors.shape[0]} διανύσματα εγγράφων με {doc_concept_vectors.shape[1]} έννοιες/διαστάσεις.\n")


# --- ΒΗΜΑ Β: Κανονικοποίηση Διανυσμάτων (Η ΔΙΟΡΘΩΣΗ) ---
print("--- ΒΗΜΑ Β: Κανονικοποίηση Διανυσμάτων (StandardScaler) ---")
print("Πριν την κανονικοποίηση, οι στήλες έχουν διαφορετικές κλίμακες (τυπ. αποκλίσεις):")
print(np.std(doc_concept_vectors, axis=0).round(2))

n_clusters = 250
max_iter = 5
kmeans_clusters(n_clusters, max_iter)






