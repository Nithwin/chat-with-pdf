from google import genai
from dotenv import load_dotenv
import os
import numpy as np
import json
load_dotenv()


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

api_key = os.getenv("GEMINI_API_KEY") 

client = genai.Client(api_key=api_key)


question = input("User: ")

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
)

question_embedding = response.embeddings[0].values

best_chunk = None
best_score = -1

with open("embeddings.json", "r") as file:
    chunk_embeddings = json.load(file)

for chunk in chunk_embeddings:
    similarity = cosine_similarity(chunk["embedding"], question_embedding)

    #print(f"Similarity: {similarity}")

    if similarity > best_score:
        best_score = similarity
        best_chunk = chunk["text"]


prompt = f"""
Answer the user's question only by using given context

Context:
{best_chunk}

Question:
{question}

If the answer was not present in the given context say
I don't have enough Information
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)

print("AI: ", response.text) 
