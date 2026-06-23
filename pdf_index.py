# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
import chromadb
import os

load_dotenv()

db_client = chromadb.PersistentClient(path="./chromadb")

db_client.delete_collection("documents")

collection = db_client.get_or_create_collection(
    name="documents"
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


reader = PdfReader("resume.pdf")

text = ""

for page in reader.pages:
    extracted_text = page.extract_text()

    if extracted_text:
        text += extracted_text + "\n"
    

chunks = [
    text[i:i+1000]
    for i in range(0, len(text), 500)
]


documents = []
embeddings = []
ids = []

for i, chunk in enumerate(chunks):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk,
    )

    embedding = response.embeddings[0].values

    documents.append(chunk)
    embeddings.append(embedding)
    ids.append(f"chunk_{i}")

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
)

print(collection.count())