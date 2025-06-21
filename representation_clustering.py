from preprocessing import texts, doc_labels, texts_untouched
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import texts, doc_labels
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
# Set the styles to Seaborn
sns.set()
# Import the KMeans module so we can perform k-means clustering with sklearn
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull # Απαραίτητο για το "αγκάλιασμα" των clusters
from sklearn.datasets import make_blobs


def kmeans(doc_vector, n_clusters=250):
    """
    Εκτελεί KMeans clustering και δημιουργεί μια απλή οπτικοποίηση των clusters.

    Args:
        doc_vector (np.array): Τα δεδομένα προς ομαδοποίηση (π.χ. διανύσματα από κείμενα).
        n_clusters (int): Ο επιθυμητός αριθμός των clusters.
    """
    # Βήμα 1: Εκτέλεση του KMeans
    # Χρησιμοποιούμε n_init='auto' για συμβατότητα και random_state για αναπαραγώγιμα αποτελέσματα.
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    
    # Η fit_predict() κάνει και το fit και το predict με μία κλήση.
    identified_clusters = kmeans.fit_predict(doc_vector)

    # Βήμα 2: Προετοιμασία για οπτικοποίηση (Χειρισμός διαστάσεων)
    # Ένα scatter plot χρειάζεται 2 διαστάσεις (x, y). Αν τα δεδομένα μας έχουν περισσότερες,
    # χρησιμοποιούμε PCA για να τις "συμπιέσουμε" σε 2 για το γράφημα.
    
    plot_data = doc_vector
    xlabel = "Feature 1"
    ylabel = "Feature 2"

    # Βήμα 3: Δημιουργία του γραφήματος
    plt.style.use('seaborn-v0_8-whitegrid') # Για πιο όμορφο γράφημα
    plt.figure(figsize=(10, 7))
    
    # Το 'c=identified_clusters' χρωματίζει αυτόματα κάθε σημείο ανάλογα με το cluster του.
    scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], 
                          c=identified_clusters, 
                          cmap='viridis', # Μια ωραία παλέτα χρωμάτων
                          s=50, alpha=0.8)

    plt.title(f'Οπτικοποίηση KMeans με {n_clusters} Clusters', fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Προσθέτουμε ένα legend για να ξέρουμε ποιο χρώμα είναι ποιο cluster
    plt.legend(handles=scatter.legend_elements()[0], 
               labels=[f'Cluster {i}' for i in range(n_clusters)],
               title="Clusters")
               
    plt.show()
     

'''
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


print(tfidf_representation_stemmed['_Alita__opening_weekend__0'])
'''
'''
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

print(boolean_3d_array)
'''

# ============ Singular Value Decomposition ============ #
vectorizer_svd = CountVectorizer()
doc_term_matrix = vectorizer_svd.fit_transform(texts_untouched).toarray()
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
k_plot = min(2, num_concepts_available)


# ==================== PLOT 1: Μπάρες Ιδιοτιμών (Singular Values) ====================
#plt.figure(figsize=(10, 6))
# plt.bar(range(len(Lambda)), Lambda, color='skyblue', edgecolor='black', label="Singular Values")
# plt.title("Ιδιοτιμές (Singular Values - Σημαντικότητα Εννοιών)")
# plt.xlabel("Δείκτης Έννοιας (Ταξινομημένες κατά Σπουδαιότητα)")
# plt.ylabel("Τιμή Ιδιοτιμής (Μέγεθος)")
# plt.xticks(range(len(Lambda))) # Εμφάνιση όλων των δεικτών εννοιών
# plt.legend()
# plt.grid(axis='y', linestyle='--')
# plt.tight_layout()
# plt.show()

print("\nΕπεξήγηση Γραφήματος Ιδιοτιμών:")
print("Κάθε μπάρα αντιστοιχεί σε μια 'λανθάνουσα έννοια' ή 'θέμα' που ανακαλύφθηκε από το SVD.")
print("Το ύψος της μπάρας (ιδιοτιμή) δείχνει τη σπουδαιότητα ή τη 'δύναμη' αυτής της έννοιας στην εξήγηση της διακύμανσης των δεδομένων.")
print("Οι έννοιες είναι ταξινομημένες κατά φθίνουσα σειρά σπουδαιότητας.")
print("Αυτό το γράφημα βοηθά στην απόφαση για το πόσες έννοιες να διατηρηθούν για μείωση διαστατικότητας (π.χ., μέθοδος 'αγκώνα').")
print("-" * 70)


# ==================== PLOT 2: Προβολή Εγγράφων στον 2D Χώρο Εννοιών ====================
if k_plot >= 2: # Απεικόνιση μόνο αν έχουμε τουλάχιστον 2 έννοιες
    # Συντεταγμένες εγγράφων στον χώρο των εννοιών: U_k * S_k (όπου S_k διαγώνιος πίνακας των Lambda_k)
    # doc_concept_vectors είναι (αριθμός_εγγράφων, αριθμός_διαθέσιμων_εννοιών)
    doc_concept_vectors = U @ np.diag(Lambda) # U[:, :k] @ np.diag(Lambda[:k])

    # plt.figure(figsize=(12, 8))
    # # Χρήση των δύο πρώτων στηλών για 2D γράφημα
    # plt.scatter(doc_concept_vectors[:, 0], doc_concept_vectors[:, 1], alpha=0.7, c='dodgerblue', s=120, edgecolors='w')

    # Ετικέτες εγγράφων
    doc_labels = [f"Doc {i+1}" for i in range(len(texts_untouched))]
    # Εναλλακτικά, μπορείτε να χρησιμοποιήσετε μέρη του αρχικού κειμένου αν είναι σύντομα:
    # doc_labels = [text[:25]+"..." if len(text)>25 else text for text in texts_untouched]

    # for i, label in enumerate(doc_labels):
    #     plt.annotate(label, (doc_concept_vectors[i, 0], doc_concept_vectors[i, 1]),
    #                  xytext=(8, 0), textcoords='offset points', fontsize=9)

    # plt.xlabel(f"Έννοια 1 (Ιδιοτιμή: {Lambda[0]:.2f})", fontsize=12)
    # plt.ylabel(f"Έννοια 2 (Ιδιοτιμή: {Lambda[1]:.2f})", fontsize=12)
    # plt.title("Προβολή Εγγράφων στον 2D Χώρο Εννοιών (LSA)", fontsize=14)
    # plt.axhline(0, color='grey', lw=0.7, linestyle=':')
    # plt.axvline(0, color='grey', lw=0.7, linestyle=':')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    print("\nΕπεξήγηση Γραφήματος Προβολής Εγγράφων:")
    print("Κάθε σημείο αντιπροσωπεύει ένα έγγραφο, προβεβλημένο στις δύο πιο σημαντικές έννοιες.")
    print("Ο οριζόντιος άξονας είναι η Έννοια 1 και ο κάθετος άξονας είναι η Έννοια 2.")
    print("Έγγραφα που βρίσκονται κοντά σε αυτόν τον χώρο θεωρούνται σημασιολογικά παρόμοια σύμφωνα με το LSA.")
    print("Έγγραφα που είναι μακριά θεωρούνται σημασιολογικά διαφορετικά.")
    print("Η αρχή των αξόνων (0,0) μπορεί να θεωρηθεί ως ένα 'ουδέτερο' σημείο.")
    print("-" * 70)
else:
    print("\nΠαράλειψη 2D Γραφήματος Προβολής Εγγράφων: Δεν υπάρχουν αρκετές διαθέσιμες έννοιες (χρειάζονται τουλάχιστον 2).")


# ==================== PLOT 3: Προβολή Όρων στον 2D Χώρο Εννοιών ====================
if k_plot >= 2 and len(vocab) >=1 : # Απεικόνιση μόνο αν έχουμε τουλάχιστον 2 έννοιες και λεξιλόγιο
    # Συντεταγμένες όρων στον χώρο των εννοιών: V_k * S_k (ή (S_k * Vt_k)^T)
    # Vt είναι (έννοιες, όροι). Χρειαζόμαστε (όροι, έννοιες). Άρα Vt.T
    # term_concept_vectors είναι (αριθμός_όρων, αριθμός_διαθέσιμων_εννοιών)
    term_concept_vectors = Vt.T @ np.diag(Lambda)

    # plt.figure(figsize=(14, 10))
    # # Χρήση των δύο πρώτων στηλών για 2D γράφημα
    # plt.scatter(term_concept_vectors[:, 0], term_concept_vectors[:, 1], alpha=0.7, c='crimson', s=70, edgecolors='w')

    # # Ετικέτες όρων
    # for i, term in enumerate(vocab):
    #     plt.annotate(term, (term_concept_vectors[i, 0], term_concept_vectors[i, 1]),
    #                  xytext=(6, -6), textcoords='offset points',
    #                  fontsize=8) # Προσαρμόστε το μέγεθος γραμματοσειράς

    # plt.xlabel(f"Έννοια 1 (Ιδιοτιμή: {Lambda[0]:.2f})", fontsize=12)
    # plt.ylabel(f"Έννοια 2 (Ιδιοτιμή: {Lambda[1]:.2f})", fontsize=12)
    # plt.title("Προβολή Όρων στον 2D Χώρο Εννοιών (LSA)", fontsize=14)
    # plt.axhline(0, color='grey', lw=0.7, linestyle=':')
    # plt.axvline(0, color='grey', lw=0.7, linestyle=':')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    print("\nΕπεξήγηση Γραφήματος Προβολής Όρων:")
    print("Κάθε σημείο αντιπροσωπεύει έναν όρο, προβεβλημένο στις δύο πιο σημαντικές έννοιες.")
    print("Όροι που βρίσκονται κοντά σε αυτόν τον χώρο τείνουν να συν-εμφανίζονται ή χρησιμοποιούνται σε παρόμοια περιβάλλοντα.")
    print("Όροι που έχουν μεγάλη 'φόρτιση' (μακριά από την αρχή) σε μια έννοια είναι σημαντικοί για τον ορισμό αυτής της έννοιας.")
    print("Π.χ., όροι κοντά μεταξύ τους και μακριά από την αρχή κατά μήκος του άξονα της Έννοιας 1 είναι σημασιολογικά σχετικοί και σημαντικοί για την Έννοια 1.")
    print("-" * 70)
else:
    print("\nΠαράλειψη 2D Γραφήματος Προβολής Όρων: Δεν υπάρχουν αρκετές διαθέσιμες έννοιες ή όροι στο λεξιλόγιο.")

# ==================== PLOT 4: Σημαντικότεροι Όροι ανά Έννοια (Bar Plots) ====================
if num_concepts_available > 0 and len(vocab) > 0:
    # Ο Vt έχει σχήμα (αριθμός_εννοιών, αριθμός_όρων)
    # Δημιουργία DataFrame για ευκολότερο χειρισμό
    # Οι στήλες είναι οι όροι, οι γραμμές είναι οι έννοιες
    Vt_df_concepts_terms = pd.DataFrame(Vt, columns=vocab)
    Vt_df_concepts_terms.index = [f"Έννοια {i+1}" for i in range(Vt.shape[0])]

    print("\nΣημαντικότεροι Όροι που Συνεισφέρουν σε Κάθε Έννοια (Απόλυτες Φορτίσεις από Vt):")
    n_top_terms = min(7, len(vocab)) # Εμφάνιση των top 7 όρων ή λιγότερων αν το λεξιλόγιο είναι μικρό

    num_concepts_to_plot_terms = min(k_plot, num_concepts_available, 3) # Γράφημα για έως 3 έννοιες ή k_plot

    for concept_idx in range(Vt_df_concepts_terms.shape[0]): # Επανάληψη για όλες τις διαθέσιμες έννοιες
        concept_label = Vt_df_concepts_terms.index[concept_idx]
        # Λήψη της γραμμής για την τρέχουσα έννοια (φορτίσεις όλων των όρων σε αυτή την έννοια)
        concept_loadings = Vt_df_concepts_terms.iloc[concept_idx]

        # Κορυφαίοι N όροι βάσει απόλυτης φόρτισης
        top_terms_series = concept_loadings.abs().nlargest(n_top_terms)

        print(f"\n{concept_label} (Ιδιοτιμή: {Lambda[concept_idx]:.2f}):")
        for term, loading_abs in top_terms_series.items():
            original_loading = concept_loadings[term] # Η αρχική φόρτιση με το πρόσημο
            print(f"  - {term} (απόλ. φόρτ.: {loading_abs:.3f}, αρχική φόρτ.: {original_loading:.3f})")

        # Δημιουργία γραφήματος μόνο για τις πρώτες num_concepts_to_plot_terms έννοιες
        if concept_idx < num_concepts_to_plot_terms:
            # Λήψη των πραγματικών φορτίσεων (με πρόσημο) για αυτούς τους κορυφαίους όρους
            # και ταξινόμηση για καλύτερη οπτικοποίηση στο γράφημα
            term_values_for_plot = concept_loadings[top_terms_series.index].sort_values(ascending=False)

            # plt.figure(figsize=(10, max(4, n_top_terms * 0.6))) # Προσαρμογή ύψους βάσει n_top_terms
            # term_values_for_plot.plot(kind='barh', color=['mediumseagreen' if x > 0 else 'lightcoral' for x in term_values_for_plot])
            # plt.title(f"Top {n_top_terms} Όροι για την {concept_label} (Ιδιοτιμή: {Lambda[concept_idx]:.2f})", fontsize=14)
            # plt.xlabel("Φόρτιση στην Έννοια", fontsize=12)
            # plt.ylabel("Όρος", fontsize=12)
            # plt.gca().invert_yaxis() # Εμφάνιση του κορυφαίου όρου στην κορυφή
            # plt.grid(axis='x', linestyle='--', alpha=0.7)
            # plt.tight_layout()
            # plt.show()
else:
    print("\nΠαράλειψη Ανάλυσης Σημαντικότερων Όρων: Δεν υπάρχουν διαθέσιμες έννοιες ή όροι.")



print("\nΟλοκληρώθηκαν οι οπτικοποιήσεις και η ανάλυση SVD.")


# ===================================================================
# 1. IMPORTS & SETUP
# ===================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Υποθέτουμε ότι οι παρακάτω μεταβλητές είναι ήδη διαθέσιμες
# από το αρχείο preprocessing.py
from preprocessing import texts_untouched

# Set the styles to Seaborn
sns.set()
plt.style.use('seaborn-v0_8-whitegrid')

# ===================================================================
# 2. FUNCTION DEFINITION
# ===================================================================
def dbscan_clustering(doc_vector, eps, min_samples=2):
    """
    Εκτελεί DBSCAN clustering και δημιουργεί μια οπτικοποίηση των clusters.
    Args:
        doc_vector (np.array): Τα δεδομένα προς ομαδοποίηση (π.χ. κανονικοποιημένα διανύσματα από SVD).
        eps (float): Η μέγιστη απόσταση μεταξύ δύο δειγμάτων.
        min_samples (int): Ο ελάχιστος αριθμός δειγμάτων για ένα σημείο πυρήνα.
    """
    # Βήμα 1: Εκτέλεση του DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(doc_vector)

    # Βήμα 2: Προετοιμασία για οπτικοποίηση με PCA (για >2 διαστάσεις)
    if doc_vector.shape[1] > 2:
        print("Τα δεδομένα έχουν >2 διαστάσεις. Εφαρμόζεται PCA για την οπτικοποίηση...")
        pca = PCA(n_components=2, random_state=42)
        plot_data = pca.fit_transform(doc_vector)
        xlabel = "Principal Component 1"
        ylabel = "Principal Component 2"
    else:
        plot_data = doc_vector
        xlabel = "Feature 1"
        ylabel = "Feature 2"

    # Βήμα 3: Δημιουργία του γραφήματος
    plt.figure(figsize=(12, 8))
    unique_labels = set(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col, marker, label = [0, 0, 0, 1], 'x', 'Θόρυβος (Noise)'
        else:
            marker, label = 'o', f'Cluster {k}'
            
        class_member_mask = (labels == k)
        xy = plot_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14 if marker == 'o' else 8, label=label)

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    plt.title(f'Οπτικοποίηση DBSCAN (βρέθηκαν {n_clusters_} clusters)\neps={eps}, min_samples={min_samples}', fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.show()
    
    return labels

# ===================================================================
# 3. MAIN EXECUTION FLOW
# ===================================================================

# --- ΒΗΜΑ Α: SVD & Δημιουργία Διανυσμάτων ---
print("--- ΒΗΜΑ Α: Εκτέλεση Singular Value Decomposition (SVD) ---")
vectorizer_svd = CountVectorizer()
doc_term_matrix = vectorizer_svd.fit_transform(texts_untouched).toarray()
U, Lambda, Vt = np.linalg.svd(doc_term_matrix, full_matrices=False)

# Δημιουργία των διανυσμάτων που θα χρησιμοποιηθούν για clustering
doc_concept_vectors = U @ np.diag(Lambda)
print(f"Δημιουργήθηκαν {doc_concept_vectors.shape[0]} διανύσματα εγγράφων με {doc_concept_vectors.shape[1]} έννοιες/διαστάσεις.\n")


# --- ΒΗΜΑ Β: Κανονικοποίηση Διανυσμάτων (Η ΔΙΟΡΘΩΣΗ) ---
print("--- ΒΗΜΑ Β: Κανονικοποίηση Διανυσμάτων (StandardScaler) ---")
print("Πριν την κανονικοποίηση, οι στήλες έχουν διαφορετικές κλίμακες (τυπ. αποκλίσεις):")
print(np.std(doc_concept_vectors, axis=0).round(2))

scaler = StandardScaler()
doc_vectors_normalized = scaler.fit_transform(doc_concept_vectors)
kmeans(doc_vectors_normalized)
print("\nΜετά την κανονικοποίηση, όλες οι στήλες έχουν την ίδια κλίμακα (τυπ. απόκλιση ~1.0):")
print(np.std(doc_vectors_normalized, axis=0).round(2))
print("-" * 50 + "\n")


# --- ΒΗΜΑ Γ: Εύρεση Βέλτιστου `eps` ---
print("--- ΒΗΜΑ Γ: Δημιουργία γραφήματος για την εύρεση του `eps` (k-distance graph) ---")
# Ορίζουμε το min_samples. Μια τιμή γύρω στο 5-10% του συνόλου των δεδομένων είναι καλή αρχή.
# Για ~70 έγγραφα, το 5 είναι μια λογική τιμή.
param_min_samples = 5 

# Υπολογισμός της απόστασης κάθε σημείου από τους k-πλησιέστερους γείτονές του
nbrs = NearestNeighbors(n_neighbors=param_min_samples).fit(doc_vectors_normalized)
distances, indices = nbrs.kneighbors(doc_vectors_normalized)

# Παίρνουμε την απόσταση του τελευταίου (k-οστού) γείτονα και ταξινομούμε
k_distance = np.sort(distances[:, param_min_samples-1], axis=0)

# Δημιουργία του γραφήματος
plt.figure(figsize=(10, 6))
plt.plot(k_distance)
plt.title(f'{param_min_samples}-Distance Graph (Γράφημα για την Εύρεση του `eps`)')
plt.xlabel("Έγγραφα (ταξινομημένα κατά απόσταση από τον k-οστό γείτονα)")
plt.ylabel(f"Απόσταση από τον {param_min_samples}-οστό γείτονα")
plt.grid(True, linestyle='--')
print("Παρατηρήστε το παρακάτω γράφημα. Η ιδανική τιμή για το `eps` είναι συνήθως στο σημείο 'αγκώνα' (elbow),")
print("δηλαδή στο σημείο όπου η καμπύλη αρχίζει να ανεβαίνει απότομα.")
plt.show()
print("-" * 50 + "\n")


# --- ΒΗΜΑ Δ: Εκτέλεση DBSCAN & Αξιολόγηση ---
print("--- ΒΗΜΑ Δ: Εκτέλεση DBSCAN Clustering ---")
# !!! ΣΗΜΑΝΤΙΚΟ: Αλλάξτε την παρακάτω τιμή 'param_eps' με βάση την τιμή
# που είδατε στον 'αγκώνα' του παραπάνω γραφήματος.
# Μια καλή αρχική τιμή για κανονικοποιημένα δεδομένα είναι συνήθως μεταξύ 1.5 και 4.0
param_eps = 12.075  # <--- ΑΛΛΑΞΤΕ ΑΥΤΗ ΤΗΝ ΤΙΜΗ ΒΑΣΕΙ ΤΟΥ ΓΡΑΦΗΜΑΤΟΣ

print(f"Εκτέλεση DBSCAN με παραμέτρους: eps={param_eps}, min_samples={param_min_samples}\n")
dbscan_labels = dbscan_clustering(doc_vectors_normalized, eps=param_eps, min_samples=param_min_samples)


# --- Αξιολόγηση Αποτελέσματος ---
print("\n" + "="*25)
print(" ΑΞΙΟΛΟΓΗΣΗ DBSCAN ")
print("="*25)

n_clusters_found = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise_points = list(dbscan_labels).count(-1)

print(f"Αλγόριθμος: DBSCAN")
print(f"Παράμετροι: eps={param_eps}, min_samples={param_min_samples}")
print(f"Αποτέλεσμα: Βρέθηκαν {n_clusters_found} clusters.")
print(f"Αριθμός εγγράφων που χαρακτηρίστηκαν ως 'θόρυβος' (outliers): {n_noise_points}\n")

# Ανάλυση Περιεχομένου των Clusters
doc_names = [f"Doc_{i+1} ({text[:35]}...)" for i, text in enumerate(texts_untouched)]
print(doc_names)
print("--- Περιεχόμενο των Clusters ---")
if n_clusters_found > 0:
    for cluster_id in sorted(set(dbscan_labels)):
        if cluster_id == -1:
            continue
        print(f"\n[ Cluster {cluster_id} ]")
        docs_in_cluster = [doc_names[i] for i, label in enumerate(dbscan_labels) if label == cluster_id]
        for doc in docs_in_cluster:
            print(f"  - {doc}")
else:
    print("Δεν βρέθηκε κανένα cluster με τις τρέχουσες παραμέτρους.")

print("\n--- Έγγραφα που είναι Θόρυβος (Outliers) ---")
if n_noise_points > 0:
    noise_docs = [doc_names[i] for i, label in enumerate(dbscan_labels) if label == -1]
    for doc in noise_docs:
        print(f"  - {doc}")
else:
    print("Κανένα έγγραφο δεν χαρακτηρίστηκε ως θόρυβος.")


# ===================================================================
# 5. AGGLOMERATIVE (HIERARCHICAL) CLUSTERING
# ===================================================================
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

print("\n" + "="*25)
print(" ΕΚΤΕΛΕΣΗ AGGLOMERATIVE CLUSTERING ")
print("="*25)

# -------------------------------------------------------------------
# ΒΗΜΑ 1: Δημιουργία και Οπτικοποίηση του Δενδρογράμματος
# -------------------------------------------------------------------
# Το δενδρόγραμμα μας δείχνει πώς τα clusters συγχωνεύονται σε κάθε βήμα.
# Είναι το καλύτερο εργαλείο για να αποφασίσουμε τον αριθμό των clusters (k).

print("--- Δημιουργία Δενδρογράμματος για την επιλογή του αριθμού των clusters (k) ---")

# Η μέθοδος 'ward' είναι μια καλή γενική επιλογή, καθώς προσπαθεί να ελαχιστοποιήσει
# τη διακύμανση (variance) μέσα σε κάθε cluster.
# Χρησιμοποιούμε πάντα τα κανονικοποιημένα δεδομένα.
linkage_matrix = linkage(doc_vectors_normalized, method='ward')

plt.figure(figsize=(15, 8))
plt.title('Δενδρόγραμμα Ιεραρχικής Ομαδοποίησης (Hierarchical Clustering Dendrogram)', fontsize=16)
plt.xlabel('Έγγραφα (Δείκτης)', fontsize=12)
plt.ylabel('Απόσταση (Distance)', fontsize=12)

# Δημιουργία του δενδρογράμματος
dendrogram(linkage_matrix,
           leaf_rotation=90.,  # περιστροφή των ετικετών για να χωράνε
           leaf_font_size=8.)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Πώς να διαβάσετε το Δενδρόγραμμα:
# 1. Κάθε φύλλο (κάτω μέρος) είναι ένα έγγραφο.
# 2. Οι κάθετες γραμμές δείχνουν τα clusters.
# 3. Οι οριζόντιες γραμμές δείχνουν τις συγχωνεύσεις των clusters.
# 4. Το ύψος της οριζόντιας γραμμής δείχνει την απόσταση/ανομοιότητα
#    μεταξύ των clusters που συγχωνεύθηκαν.
#
# ΠΩΣ ΝΑ ΕΠΙΛΕΞΕΤΕ ΤΟ 'k' (αριθμός clusters):
# - Φανταστείτε μια οριζόντια γραμμή που κόβει τις κάθετες γραμμές.

#   είναι ο αριθμός των clusters!
# - Κόψτε σε ένα ύψος όπου η απόσταση μεταξύ των συγχωνεύσεων είναι μεγάλη
#   (δηλαδή, εκεί που οι κάθετες γραμμές είναι μακριές).
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# ΒΗΜΑ 2: Εκτέλεση Clustering και Οπτικοποίηση για συγκεκριμένο k
# -------------------------------------------------------------------
# Αφού είδατε το δενδρόγραμμα, αποφασίστε έναν αριθμό clusters.
# Αλλάξτε την τιμή k_agglomerative παρακάτω.
k_agglomerative = 180 # <--- ΑΛΛΑΞΤΕ ΑΥΤΟ ΤΟΝ ΑΡΙΘΜΟ ΒΑΣΕΙ ΤΟΥ ΔΕΝΔΡΟΓΡΑΜΜΑΤΟΣ

print(f"\n--- Εκτέλεση Agglomerative Clustering για k={k_agglomerative} clusters ---")

# Δημιουργία και εκπαίδευση του μοντέλου
agglomerative_model = AgglomerativeClustering(n_clusters=k_agglomerative)
agglomerative_labels = agglomerative_model.fit_predict(doc_vectors_normalized)

# Οπτικοποίηση του αποτελέσματος (με PCA για 2D)
pca = PCA(n_components=2, random_state=42)
plot_data_agg = pca.fit_transform(doc_vectors_normalized)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(plot_data_agg[:, 0], plot_data_agg[:, 1], c=agglomerative_labels, cmap='viridis', s=60)
plt.title(f'Οπτικοποίηση Agglomerative Clustering για {k_agglomerative} Clusters', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(handles=scatter.legend_elements()[0],
           labels=[f'Cluster {i}' for i in range(k_agglomerative)],
           title="Clusters")
plt.grid(True)
plt.show()


# -------------------------------------------------------------------
# ΒΗΜΑ 3: Αξιολόγηση των Clusters
# -------------------------------------------------------------------
print("\n" + "="*25)
print(" ΑΞΙΟΛΟΓΗΣΗ AGGLOMERATIVE CLUSTERING ")
print("="*25)

print(f"Βρέθηκαν {k_agglomerative} clusters.\n")

# Ανάλυση Περιεχομένου των Clusters
doc_names = [f"Doc_{i+1} ({text[:35]}...)" for i, text in enumerate(texts_untouched)]

print("--- Περιεχόμενο των Clusters ---")
for cluster_id in range(k_agglomerative):
    print(f"\n[ Cluster {cluster_id} ]")
    docs_in_cluster = [doc_names[i] for i, label in enumerate(agglomerative_labels) if label == cluster_id]
    for doc in docs_in_cluster:
        print(f"  - {doc}")



