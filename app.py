from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pdf_processing import extract_text_from_pdf
from search import search_documents
import os
from embeddings import get_embedding_from_llm
from pinecone import Pinecone, ServerlessSpec
from flask_cors import CORS

pinecone_api_key = os.environ['PINECONE_API_KEY']
pc = Pinecone(api_key=pinecone_api_key)
index_name = "case-study-index"

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name="case-study-index",
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index("case-study-index")


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    text = extract_text_from_pdf(file_path)
    embedding = get_embedding_from_llm(text)

    metadata = {
        'text': text
    }

    index.upsert([(file.filename, embedding, metadata)])

    return jsonify({"filename": filename, "message": "File uploaded and indexed successfully"}), 201


@app.route('/docs', methods=['GET'])
def search_docs():
    query = request.args.get('q', '')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = search_documents(query)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
