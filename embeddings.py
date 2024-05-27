import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re


load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding_from_llm(text):
    return model.encode(text).tolist()


def clean_title(title):
    title = title.replace('?', '')
    title = title.replace('\n', '')
    title = re.sub(r'[^A-Za-z0-9 ]+', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()


def generate_titles_from_snippet(snippets, query):
    titles = []
    for snippet in snippets:
        prompt = (
            f"Generate a concise and descriptive title for the following text snippet, "
            f"considering the geographical context and relevance to the query:\n\n{snippet}\n\n"
            f"Query: {query}"
        )
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",  # Specify the model name
            prompt=prompt,
            max_tokens=10,
            n=1,
            stop=None,
            temperature=0.7,
        )

        title = response.choices[0].text.strip()
        cleaned_title = clean_title(title)
        titles.append(cleaned_title)
        return titles


def extract_snippets(text, query_embedding, num_snippets=5):
    sentences = text.split(". ")
    sentence_embeddings = model.encode(sentences)

    similarities = cosine_similarity([query_embedding], sentence_embeddings)[0]

    relevant_indices = similarities.argsort()[-num_snippets:][::-1]
    snippets = [sentences[idx] for idx in relevant_indices]

    return snippets
