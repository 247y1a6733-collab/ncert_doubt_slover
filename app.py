from flask import Flask, request, jsonify, send_from_directory
import os, pickle
from sklearn.metrics.pairwise import cosine_similarity
from ingest import load_index, build_vector_index
import numpy as np

app = Flask(__name__, static_folder='../frontend', static_url_path='/')

# Load or build index on startup
try:
    vectorizer, X, meta = load_index()
except Exception as e:
    vectorizer, X, meta = build_vector_index()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index1.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({"error":"No question provided"}), 400
    qv = vectorizer.transform([question])
    sims = cosine_similarity(qv, X).flatten()
    topk = sims.argsort()[::-1][:3]
    answers = []
    for idx in topk:
        answers.append({
            "score": float(sims[idx]),
            "source": meta[idx]['source'],
            "paragraph": meta[idx]['paragraph']
        })
    # choose best match
    best = answers[0]
    response_text = best['paragraph']
    return jsonify({"answer": response_text, "source": best['source'], "candidates": answers})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)
