import os
from embeddings import get_embedding_from_llm
from pinecone import Pinecone, ServerlessSpec
import re

pinecone_api_key = os.environ['PINECONE_API_KEY']
index_name = "case-study-index"


def index_initial_files(folder_path):
    pc = Pinecone(api_key=pinecone_api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Replace with your model dimensions
            metric="cosine",  # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index("case-study-index")

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r",encoding="utf-8") as file:
                text = file.read()
                text = text.replace('\n', ' ')
                text = re.sub(r'\s+', ' ', text)

                embedding = get_embedding_from_llm(text)

                metadata = {
                    'text': text,
                }

                index.upsert([(filename, embedding, metadata)])


if __name__ == "__main__":
    initial_folder = "./samples"  # Folder name
    index_initial_files(initial_folder)
