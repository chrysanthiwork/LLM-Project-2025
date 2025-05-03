from preprocessing import documents,vocabulary

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

#Πρώτη αναπαράσταση. Boolean Representation
#δουλευει, απλα για το test θα δουμε τα 10 πρωτα

'''bow_documents = {}
for key, article_lists in documents.items():
    bow_documents[key] = [
        bag_of_words_representation(article, vocabulary)
        for article in article_lists
    ]'''

#test function bow
bow_documents = {}
for i, (key, article_lists) in enumerate(documents.items()):
    if i >= 1:
        break
    bow_documents[key] = [
        bag_of_words_representation(article, vocabulary)
        for article in article_lists
    ]

print(bow_documents)