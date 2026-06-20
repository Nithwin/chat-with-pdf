import chromadb
from google import genai
from dotenv import load_dotenv
import os 

load_dotenv()

db_client = chromadb.PersistentClient(path="./chromadb")

collection = db_client.get_collection(
    name="documents"
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


question = input("User: ")

response = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
)

question_embedding = response.embeddings[0].values

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3,
)

retrived_chunks = results["documents"][0]

context = "\n\n".join(retrived_chunks)

prompt = f"""
Answer the user's question only by using the provided context.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I don't have enough information."
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)

print(response.text)