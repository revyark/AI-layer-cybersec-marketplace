from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import tldextract
from scipy.sparse import hstack

app = Flask(__name__)

# -----------------------
# Custom tokenizer (must match training)
# -----------------------
def url_tokenizer(s):
    return re.split(r'[:/?=&._\-\.]+', s)

# -----------------------
# Load model + vectorizer + label encoder
# -----------------------
model, tfidf, le = joblib.load("malicious_url_detector.h5")

# -----------------------
# Lexical features
# -----------------------
def lexical_features(urls):
    out = []
    for u in urls:
        u = str(u)
        parsed = tldextract.extract(u)
        subdomain = parsed.subdomain
        domain = parsed.domain
        suffix = parsed.suffix
        features = {
            'len': len(u),
            'count_slash': u.count('/'),
            'count_dot': u.count('.'),
            'count_dash': u.count('-'),
            'count_at': u.count('@'),
            'count_qm': u.count('?'),
            'count_eq': u.count('='),
            'count_digits': sum(ch.isdigit() for ch in u),
            'num_subdomain_parts': 0 if not subdomain else len(subdomain.split('.')),
            'has_ip': 1 if re.match(r'^(http[s]?://)?\d+\.\d+\.\d+\.\d+', u) else 0,
            'tld_len': len(suffix),
            'domain_len': len(domain),
            'entropy': -sum(
                (u.count(c) / len(u)) * np.log2(u.count(c) / len(u) + 1e-9)
                for c in set(u)
            )
        }
        out.append(list(features.values()))
    return np.array(out)

# -----------------------
# Predict function
# -----------------------
def predict_url(url: str):
    lex = lexical_features([url])
    tfidf_vec = tfidf.transform([url])
    features = hstack([tfidf_vec, lex])

    pred = model.predict(features)[0]
    label = le.inverse_transform([pred])[0]
    return label

# -----------------------
# API Endpoint
# -----------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    prediction = predict_url(url)
    return jsonify({"prediction": prediction})

# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
