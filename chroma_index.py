import chromadb
import json
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


db_client = chromadb.PersistentClient(path="./chromadb")

collection = db_client.get_or_create_collection(
    name="documents"
)

embeddings = []
documents = []
ids = []

with open("sample_document.txt", 'r') as file:
    document = file.read()

    chunks = document.split("\n\n")
    
    for i, chunk in enumerate(chunks):
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk,
        )

        embedding = response.embeddings[0].values

        embeddings.append(embedding)
        documents.append(chunk)
        ids.append(f"chunk_{i}")


collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
)

print(collection.count())