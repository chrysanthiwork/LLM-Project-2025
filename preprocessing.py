import os
import re
import unicodedata
import requests
import snowballstemmer
import enchant
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')

# Εγκατάσταση λεξικού αγγλικής
english_dict = enchant.Dict("en_US")
stemmer = snowballstemmer.stemmer('english')

# Συνάρτηση για αφαίρεση τόνων
def remove_accents(text):
    return ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )

# Συνάρτηση ανάγνωσης stopwords από URL
def read_stopwords_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        stopwords = response.text.splitlines()
        return [word.strip() for word in stopwords]
    else:
        print(f"Failed to retrieve the file. Status code: {response.status_code}")
        return []

# Αφαίρεση stopwords από κείμενο
def remove_stopwords(text, stopwords_list):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords_list]
    return ' '.join(filtered_words)

# Έλεγχος εγκυρότητας λέξης (υπάρχει σε λεξικό ή WordNet)
def is_valid_word(word):
    word = re.sub(r'\W+', '', word).strip().lower()
    return english_dict.check(word) or bool(wordnet.synsets(word))


# ============ Κύρια Επεξεργασία ============ #

root_folder = 'downloaded_articles'

documents = {}
texts = []
vocabulary = set()

# Διαβάζουμε όλα τα αρχεία
for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)
    if os.path.isdir(folder_path):
        articles = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    articles.append(content)
        documents[folder_name] = articles

# Διαβάζουμε stopwords
greek_stopwords = read_stopwords_from_url(
    'https://raw.githubusercontent.com/stopwords-iso/stopwords-el/refs/heads/master/stopwords-el.txt'
)
english_stopwords = read_stopwords_from_url(
    'https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt'
)

# Επεξεργασία κειμένων
texts = []
texts_untouched = []
vocabulary = set()
doc_labels = []  # 👈 κρατάει την κατηγορία του κάθε text

for category, articles in documents.items():  # 👈 χρησιμοποίησε και το category
    for article in articles:
        article = article.lower()
        article = re.sub(r'[^a-zA-ZΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώ\s]', '', article)
        article = remove_accents(article)
        article = remove_stopwords(article, english_stopwords)
        texts_untouched.append(article)
        words = article.split()

        processed_words = []
        for word in words:
            word = re.sub(r'\W+', '', word).strip()
            if is_valid_word(word):
                stemmed = stemmer.stemWords([word])[0]
                processed_words.append(stemmed)
                vocabulary.add(stemmed)
            else:
                print(f"Rejected: {word}")

        cleaned_text = ' '.join(processed_words)
        texts.append(cleaned_text)
        doc_labels.append(category)  # 👈 πρόσθεσε την κατηγορία

print(texts_untouched)
# Προβολή αποτελεσμάτων
'''print("\n--- Καθαρισμένα Κείμενα ---")
print(texts)

print("\n--- Μέγεθος Λεξιλογίου ---")
print(vocabulary)'''

__all__ = ['texts', 'documents', 'doc_labels']
