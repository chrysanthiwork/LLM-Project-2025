import os
import re
import unicodedata
import requests
import snowballstemmer


# Συνάρτηση για αφαίρεση τόνων από ελληνικό και αγγλικό κείμενο
def remove_accents(text):
    # Κανονικοποίηση σε μορφή NFD και αφαίρεση χαρακτήρων τόνων (Mn = Mark, Nonspacing)
    return ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )

# Function to read stopwords from a URL into a list
def read_stopwords_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        stopwords = response.text.splitlines()  # Split the response text into lines
        stopwords = [word.strip() for word in stopwords]  # Remove extra whitespace
        return stopwords
    else:
        print(f"Failed to retrieve the file. Status code: {response.status_code}")
        return []

def remove_stopwords(text, stopwords_list):
    # Tokenize the text into words
    words = text.split()
    # Filter out the stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords_list]
    # Return the filtered text as a string
    return ' '.join(filtered_words)

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
#print(texts)

greek_stopwords = read_stopwords_from_url('https://raw.githubusercontent.com/stopwords-iso/stopwords-el/refs/heads/master/stopwords-el.txt')
english_stopwords = read_stopwords_from_url('https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt')

#print(greek_stopwords)
#print(english_stopwords)

stemmer = snowballstemmer.stemmer('english')


texts = []

for articles in documents.values():
    for article in articles:
        # Ένωση λέξεων σε string
        text = ' '.join(article)

        # Αφαίρεση τόνων
        text = remove_accents(text)

        # Αφαίρεση stopwords
        text = remove_stopwords(text, english_stopwords)

        #TODO 
        #Αφαιρεση άχρηστων λέξεων από το vocabulary, πριν το stemming

        # Lemmatization/stemming
        words = text.split()
        stemmed_words = stemmer.stemWords(words)

        # Τελικό καθαρισμένο string
        cleaned_text = ' '.join(stemmed_words)

        # Προσθήκη στη λίστα
        texts.append(cleaned_text)

#print(texts)

vocabulary = set(word for articles in documents.values() for article in articles for word in article)

print(vocabulary)
print(len(vocabulary))

