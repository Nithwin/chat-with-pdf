import json
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


chunk_embeddings = []

with open("sample_document.txt", 'r') as file:
    document = file.read()

    chunks = document.split("\n\n")
    
    for chunk in chunks:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk,
        )

        embedding = response.embeddings[0].values

        chunk_embeddings.append({
            "text": chunk,
            "embedding": embedding,
        })


with open("embeddings.json", "w") as file:
    json.dump(chunk_embeddings,file)

print("Embeddings saved....!")