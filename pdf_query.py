from dotenv import load_dotenv
from google import genai
import chromadb
import os

load_dotenv()

db_client = chromadb.PersistentClient(path="./chromadb")

collection = db_client.get_collection(
    name="documents"
)

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

while True:
    question = input("User >> ")

    if question in ["quit", "stop"]:
        print("Thanks for using me comeback soon!")
        break

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
    )

    question_embedding = response.embeddings[0].values

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
    )

    retrieved_chunks = results["documents"][0]

    print("\nRetrieved Chunks:")
    print("=" * 50)

    for chunk in retrieved_chunks:
        print(chunk)
        print("-" * 50)

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    Answer the user's question only using the provided context.

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

    print("\nAI:", response.text)