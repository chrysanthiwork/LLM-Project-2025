import os
import re

def bag_of_words_representation(document_tokens, vocabulary):
    """
    Converts a document into a binary representation based on the given vocabulary.
    Parameters:
        document_tokens (list): The document represented as a list of tokens.
        vocabulary (set or list): The vocabulary set/list.
    Returns:
        dict: A dictionary with vocabulary words as keys and binary values (1 or 0) indicating presence in the document.
    """
    bow_rep = {word: 1 if word in document_tokens else 0 for word in vocabulary}
    bow_rep = {k: bow_rep[k] for k in sorted(bow_rep)}
    return bow_rep

# First we should load all the articles and save them into a dictionary. The key is the name of the folder (the category)
# and the items will be all the texts in one list 
root_folder = 'downloaded_articles'

documents = {}

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

#Σύμφωνα με το 2ο κεφάλαιο, αρχικά θέλουμε να αφήσουμε μόνο τις λέξεις.
for key, articles in documents.items():
    cleaned_articles = [
        re.sub(r'[^a-zA-ZΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώ\s]', '', article)
        for article in articles
    ]
    documents[key] = cleaned_articles

#Στην συνέχεια, κάνουμε split και lower όλες τις λέξεις του κειμένου
# Split και lower κάθε cleaned article
for key, cleaned in documents.items():
    documents[key] = [
        article.lower().split()
        for article in cleaned
    ]
#print(documents.items())

# Δημιουργία του λεξικού (vocabulary set)
vocabulary = set()
for article_lists in documents.values():
    for article in article_lists:
        vocabulary.update(article)
    
#print(len(vocabulary))

#Πρώτη αναπαράσταση. Boolean Representation
bow_documents = {}
for key, article_lists in documents.items():
    bow_documents[key] = [
        bag_of_words_representation(article, vocabulary)
        for article in article_lists
    ]

print(bow_documents)