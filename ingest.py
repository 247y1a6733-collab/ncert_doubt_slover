import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "vector_cache.pkl")

def load_corpus():
    corpus = []
    meta = []
    for filename in os.listdir(os.path.join(os.path.dirname(__file__), "..", "data")):
        if not filename.endswith('.txt'):
            continue
        path = os.path.join(os.path.dirname(__file__), "..", "data", filename)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        # split into paragraphs by blank lines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        for p in paragraphs:
            corpus.append(p)
            meta.append({"source": filename, "paragraph": p})
    return corpus, meta

def build_vector_index():
    corpus, meta = load_corpus()
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({"vectorizer": vectorizer, "X": X, "meta": meta}, f)
    return vectorizer, X, meta

def load_index():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        return cache['vectorizer'], cache['X'], cache['meta']
    else:
        return build_vector_index()

if __name__ == "__main__":
    v, X, meta = build_vector_index()
    print("Built TF-IDF index with", X.shape[0], "paragraphs")
