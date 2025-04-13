# File: TrendAnalysisAgent/app.py

from flask import Flask, request, jsonify
from src.preprocess import load_data, combine_fields, preprocess_documents
from src.topic_model import build_topic_model, get_topic_summary

app = Flask(__name__)

# Load and process data once at startup (in a real system you might update this dynamically)
DATA_PATH = "data/summaries.json"
data = load_data(DATA_PATH)
documents = combine_fields(data)
preprocessed_docs = preprocess_documents(documents)
topic_model, topics, probs = build_topic_model(preprocessed_docs)
topics_summary = get_topic_summary(topic_model)

@app.route("/topics", methods=["GET"])
def get_topics():
    """
    GET endpoint to retrieve the friendly topic summary.
    Optionally, a 'domain' parameter can be provided to filter results.
    """
    domain = request.args.get("domain")
    # For a real implementation, you would re-run the pipeline with domain filtering.
    # Here we simply return the precomputed topics summary.
    return jsonify(topics_summary)

@app.route("/documents", methods=["GET"])
def get_documents():
    """
    GET endpoint to retrieve document-level topic info.
    """
    df_json = topic_model.get_document_info(preprocessed_docs).to_json(orient="records")
    return jsonify({"documents": df_json})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
