from embeddings import get_embedding_from_llm , generate_titles_from_snippet,extract_snippets
from pinecone import Pinecone, ServerlessSpec
import os
from sentence_transformers import SentenceTransformer

pinecone_api_key = os.environ['PINECONE_API_KEY']

pc = Pinecone(api_key=pinecone_api_key)
index_name = "case-study-index"
index = pc.Index(index_name)

# q = "How can we train our remote sales team in Asia"

model = SentenceTransformer('all-MiniLM-L6-v2')


def search_documents(query, num_results=5, num_snippets=5):
    query_embedding = get_embedding_from_llm(query)
    results = index.query(vector=query_embedding, top_k=num_results, include_metadata=True)
    search_results = []
    for match in results['matches']:
        text = match['metadata']['text']
        snippets = extract_snippets(text, query_embedding,num_snippets)
        titles = generate_titles_from_snippet(snippets, query)
        search_results.append({
            'filename': match['id'],
            'titles': titles
        })

    final_titles = []
    for result in search_results:
        final_titles.append(result['titles'][0])

    return search_results
