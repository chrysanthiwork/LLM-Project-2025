import os
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

# === LOAD JSON ===
df = pd.read_json("cluster_sample.json")

# === CLEAN URLs ===
def clean_urls(nested_list):
    flat_list = [item for sublist in nested_list for item in sublist]
    return [url.replace("\\/", "/") for url in flat_list]

grouped_dict = df.groupby('label')['urls'].apply(lambda nested: clean_urls(nested)).to_dict()

# === CRAWLER FUNCTIONS ===

def sanitize_filename(s):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Αφαίρεση scripts, styles, meta, nav, footer κλπ
        for tag in soup(["script", "style", "meta", "nav", "footer", "noscript", "aside", "form", "iframe"]):
            tag.decompose()

        # Εστίαση στο κύριο περιεχόμενο: article, main, section, ή body
        candidates = soup.find_all(['article', 'main', 'section'])
        if not candidates:
            candidates = [soup.body] if soup.body else []

        # Μαζεύουμε το σημαντικό κείμενο
        text_parts = []
        for tag in candidates:
            paragraphs = tag.find_all(['p', 'h1', 'h2', 'h3', 'li'])  # μόνο λογικά block περιεχομένου
            for p in paragraphs:
                content = p.get_text(separator=' ', strip=True)
                if len(content.split()) > 5:  # Απόρριψη πολύ μικρών στοιχείων (π.χ. "ok", "read more")
                    text_parts.append(content)

        clean_text = '\n'.join(text_parts)
        return clean_text if clean_text.strip() else None

    except Exception as e:
        print(f" Σφάλμα στο URL: {url}\nΛεπτομέρειες: {e}")
        return None


def save_text_to_file(text, directory, filename):
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f" Αποθηκεύτηκε: {filepath}")

# === EXECUTION ===

base_dir = "downloaded_articles"

for label, urls in grouped_dict.items():
    print(f"\n Επεξεργασία κατηγορίας: {label}")
    label_dir = os.path.join(base_dir, sanitize_filename(label))
    
    for idx, url in enumerate(urls, start=1):
        print(f" ({idx}) {url}")
        text = extract_text_from_url(url)
        if text:
            filename = f"article_{idx}.txt"
            save_text_to_file(text, label_dir, filename)

print("Finished Downloading")