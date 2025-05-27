import time
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from pinecone import Pinecone, Index
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

start_time = time.time()

print("[INFO] Starting Flask app initialization...")
app = Flask(__name__)

print("[INFO] Loading environment variables from .env file...")
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY:
    print("[ERROR] PINECONE_API_KEY not found!")
if not OPENAI_API_KEY:
    print("[ERROR] OPENAI_API_KEY not found!")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print("[INFO] Initializing HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()
print("[INFO] Embeddings initialized.")

index_name = "medicalbot"
print(f"[INFO] Connecting to Pinecone index '{index_name}'...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)
print(f"[INFO] Pinecone index '{index_name}' ready.")

init_duration = time.time() - start_time
print(f"[INFO] App initialization completed in {init_duration:.2f} seconds.")

def retrieve_similar_chunks(query, k=3):
    print(f"[INFO] Retrieving top {k} similar chunks from Pinecone for query: {query}")
    query_emb = embeddings.embed_query(query)
    result = index.query(vector=query_emb, top_k=k, include_metadata=True)
    print(f"[INFO] Pinecone query completed. Number of matches: {len(result['matches'])}")
    # Extract text from metadata
    return [match['metadata']['text'] for match in result['matches'] if 'metadata' in match and 'text' in match['metadata']]

llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])
question_answer_chain = create_stuff_documents_chain(llm, prompt)
def run_rag_chain(user_input):
    print(f"[INFO] Running RAG chain for user input: {user_input}")
    # Retrieve top 3 similar chunks from Pinecone
    context_chunks = retrieve_similar_chunks(user_input, k=3)
    print(f"[INFO] Retrieved {len(context_chunks)} context chunks.")
    # Combine context for the LLM
    context = "\n".join(context_chunks)
    # Prepare the prompt with context
    prompt_with_context = prompt.format(input=user_input, context=context)
    print(f"[INFO] Sending prompt to LLM...")
    # Run the LLM with the prompt
    response = llm.invoke(prompt_with_context)
    print(f"[INFO] LLM response generated.")
    return response


@app.route("/")
def index_route():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = run_rag_chain(msg)
    return str(response)

if __name__ == '__main__':
    print("[INFO] Starting Flask web server...")
    print("[INFO] When you see '[INFO] * Running on', you can open the frontend at: http://127.0.0.1:8080/")
    server_start = time.time()
    app.run(host="0.0.0.0", port=8080, debug=True)
    total_runtime = time.time() - start_time
    print(f"[INFO] Program exited. Total runtime: {total_runtime:.2f} seconds.")
